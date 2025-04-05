import torch
from robustbench.data import load_cifar10
from robustbench.utils import load_model
from autoattack import AutoAttack
import foolbox as fb
import time
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import re
import io
from utils import save_and_plot_all_norms
from utils import parse_autoattacl_log

# Set up output directory
output_dir = ("./results")
os.makedirs(output_dir, exist_ok=True)

# Define epsilon for AutoAttack and FMN threshold
epsilon = 8 / 255

# Detect device (use GPU or MPS if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# List of models to evaluate
models = [
    "Carmon2019Unlabeled",
    "Wang2023Better_WRN-28-10",
    "Cui2023Decoupled_WRN-28-10",
    "Xu2023Exploring_WRN-28-10",
    "Rade2021Helper_extra"
]

# Number of test samples
n_examples = 100

print("Setup complete.")

# Load CIFAR-10 test samples
images, labels = load_cifar10(n_examples)
images, labels = images.clone().detach().to(device), labels.clone().detach().to(device)

attack_progress = {}
model_results = {}

fmn_attacks = {
    "Linf": fb.attacks.LInfFMNAttack(),
    "L2": fb.attacks.L2FMNAttack(),
    "L1": fb.attacks.L1FMNAttack(),
    "L0": fb.attacks.L0FMNAttack(),
}

print("Dataset loaded and storage initialized.")


for model_name in models:
    print(f"\n==== Evaluating model: {model_name} ====")

    # Load and prepare model
    model = load_model(model_name=model_name, dataset="cifar10", threat_model="Linf").to(device)
    model.eval()

    # Classification before attacks (Clean accuracy)
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)
    clean_accuracy = (predicted == labels).sum().item() / len(labels) * 100
    print(f"\nClean Accuracy: {clean_accuracy:.2f}%")
    # Print the true labels
    print(f"True labels: {labels}")

    ## AutoAttack

    # Initialize AutoAttack
    autoattack = AutoAttack(model, norm='Linf', eps=epsilon, version='standard', device=device)

    # Capture stdout while also printing progress
    old_stdout = sys.stdout
    sys.stdout = mystdout = io.StringIO()
    print("\n--- Starting AutoAttack ---")

    start_aa = time.time()
    adv_images_autoattack = autoattack.run_standard_evaluation(images, labels, bs=50)
    end_aa = time.time()

    sys.stdout = old_stdout  # Restore stdout
    aa_log = mystdout.getvalue()
    print(aa_log)  # Print the captured log for visibility

    # Get AutoAttack predicted labels and wrong indexes
    _, predicted_aa = model(adv_images_autoattack).max(1)
    print(f"Predicted labels AA: {predicted_aa}")
    correct = torch.sum(labels == predicted_aa)
    total = len(labels)
    print(f"Correct predictions AA: {correct}/{total} ({correct/total*100:.2f}%)")

    # Parse log and get attack progress
    attack_progress = parse_autoattack_log(aa_log, model, adv_images_autoattack, labels, model_name)

    # ----- Plotting AutoAttack Robust Accuracy Flow -----
    plt.figure(figsize=(10, 5))
    colors = ['blue', 'green', 'orange', 'purple', 'brown']
    for idx, (model_name, progress) in enumerate(attack_progress.items()):
        times, accs, steps = zip(*progress)
        plt.plot(times, accs, marker='o', linestyle='-', label=model_name, color=colors[idx % len(colors)])
        for t, acc, step in zip(times, accs, steps):
            plt.text(t, acc, f"{step}\n{acc:.1f}%", fontsize=10, verticalalignment='bottom', horizontalalignment='right')
    plt.xlabel("Cumulative Time (s)", fontsize=12)
    plt.ylabel("Robust Accuracy (%)", fontsize=12)
    plt.title("AutoAttack - Robust Accuracy Over Time", fontsize=14)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    autoattack_progress_path = os.path.join(output_dir, "autoattack_progress_all_models.png")
    plt.savefig(autoattack_progress_path)
    print(f"\nSaved AutoAttack progress plot (all models) to: {autoattack_progress_path}\n")
    plt.show()

    ## FMN

    # --- FMN Attacks ---
    fmodel = fb.PyTorchModel(model, bounds=(0, 1), device=device)
    fmn_results = {}
    adv_dict = {}

    for norm, attack in fmn_attacks.items():
        print(f"\n--- Starting FMN Attack ({norm}) on model {model_name} ---")

        # Capture stdout to track attack progress
        old_stdout = sys.stdout
        sys.stdout = mystdout = io.StringIO()

        start_fmn = time.time()
        _, adv_images, _ = attack(fmodel, images, labels, epsilons=None)
        end_fmn = time.time()

        sys.stdout = old_stdout  # Restore stdout
        fmn_log = mystdout.getvalue()
        print(fmn_log)  # Print captured FMN attack log
        print(f"Attack executed in {end_fmn-start_fmn:.2f} s")

      # Get FMN-predicted labels and wrong indexes
        _, predicted_fmn = model(adv_images).max(1)
        fmn_accuracy = torch.sum(labels == predicted_fmn).item() / len(labels) * 100
        perturbations = torch.norm((adv_images - images).reshape(adv_images.shape[0], -1), dim=1).cpu().numpy()

        fmn_results[norm] = {
            "accuracy": fmn_accuracy,
            "execution_time": end_fmn - start_fmn,
            "perturbations": perturbations
        }
        adv_dict[norm] = adv_images

        print(f"Predicted labels FMN ({norm}): {predicted_fmn}")
        correct = torch.sum(labels == predicted_fmn)
        total = len(labels)
        print(f"Robust Accuracy after FMN ({norm}): {correct}/{total} ({correct/total*100:.2f}%)")

    # --- Save and Plot Images for Each Norm ---
    save_and_plot_all_norms(model, model_name, images, labels, adv_images_autoattack, adv_dict, output_dir)

    model_results[model_name] = {
        "clean_accuracy": clean_accuracy,
        "fmn_results": fmn_results
    }

    # --- FMN-Linf Histogram of Maximum Perturbations ---
    start_fmn = time.time()
    _, adv_images_linf, _ = fmn_attacks["Linf"](fmodel, images, labels, epsilons=None)
    end_fmn = time.time()

    max_perturbations = []
    for i in range(len(images)):
        original_image = images[i].cpu().numpy()
        adversarial_image = adv_images_linf[i].cpu().numpy()
        perturbation = np.abs(adversarial_image - original_image)
        max_perturbations.append(np.max(perturbation))

    min_val = min(max_perturbations)
    max_val = max(max_perturbations)
    num_bins = n_examples
    bins = np.linspace(min_val, max_val, num_bins)
    if not np.any(np.isclose(bins, epsilon)):
        bins = np.sort(np.append(bins, epsilon))

    samples_within_epsilon = sum(p <= epsilon for p in max_perturbations)
    samples_outside_epsilon = len(max_perturbations) - samples_within_epsilon
    print(f"\nSamples with max perturbation <= {epsilon:.4f}: {samples_within_epsilon}")
    print(f"Samples with max perturbation > {epsilon:.4f}: {samples_outside_epsilon}\n")

    plt.figure(figsize=(12, 4))
    n, bins_used, patches = plt.hist(max_perturbations, bins=bins, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(x=epsilon, color='red', linestyle='--', label=f"Epsilon = {epsilon:.4f}")
    plt.xlabel("Maximum Perturbation (L_inf)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.title(f"Histogram of Maximum Perturbations - {model_name}", fontsize=14)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    fmn_linf_histogram_path = os.path.join(output_dir, f"{model_name}_fmn_linf_histogram.png")
    plt.savefig(fmn_linf_histogram_path)
    print(f"Saved FMN-Linf histogram for {model_name} to: {fmn_linf_histogram_path}")
    plt.show()

    linf_acc = 100 * torch.sum(labels == model(adv_images_linf).max(1)[1]).item() / len(labels)
    fmn_results["Linf"] = {
            "accuracy": linf_acc,
            "execution_time": end_fmn - start_fmn,
            "perturbations": np.array(max_perturbations)
        }

    model_results[model_name] = {
            "clean_accuracy": clean_accuracy,
            "fmn_results": fmn_results
        }
