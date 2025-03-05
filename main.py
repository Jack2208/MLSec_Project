import torch
from robustbench.data import load_cifar10
from robustbench.utils import load_model
from robustbench.eval import benchmark
import foolbox as fb
from autoattack import AutoAttack
import matplotlib.pyplot as plt
import numpy as np

# Detect device (MPS or CUDA)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Select model
model_name = "Carmon2019Unlabeled"
#model_name = "Wang2023Better_WRN-28-10"
#model_name = "Cui2023Decoupled_WRN-28-10"
#model_name = "Xu2023Exploring_WRN-28-10	"
#model_name = "Huang2022Revisiting_WRN-A4	"

# Evaluate the model with AutoAttack
print(f"\nEvaluating model: {model_name}")

# Load model and move to device (MPS/CUDA or CPU)
model = load_model(model_name=model_name, dataset="cifar10", threat_model="Linf").to(device)
model.eval()

# Load 50 CIFAR-10 test samples
images, labels = load_cifar10(n_examples=50)
images, labels = images.clone().detach().to(device), labels.clone().detach().to(device)

# Classification of the images without any attack
outputs = model(images)
_, predicted = torch.max(outputs, 1)
# True labels
print(f"True labels: {labels}")
# Predicted labels
print(f"Predicted labels: {predicted}")
# Accuracy of the model
correct = (predicted == labels).sum().item()
print(f"Correctly classified: {correct}/{len(labels)} ({correct/len(labels)*100:.2f}%)")

# --- Run AutoAttack Manually ---
print("\n**** AutoAttack ****\n")
autoattack = AutoAttack(model, norm='Linf', eps=8/255, version='standard', device=device)
adv_images_autoattack = autoattack.run_standard_evaluation(images, labels, bs=50)
# Predicted labels after AutoAttack
_, predicted_aa = model(adv_images_autoattack).max(1)
print(f"Predicted labels: {predicted_aa}")
# Compare true labels with predicted labels after AutoAttack
wrong_indexes_aa = torch.nonzero(labels != predicted_aa)
print(f"Wrong indexes AA: {wrong_indexes_aa}")

# Robustness of the model after AutoAttack
correct = torch.sum(labels == predicted_aa)
total = len(labels)
print(f"Correct predictions: {correct}/{total} ({correct/total*100:.2f}%)")
print(adv_images_autoattack[wrong_indexes_aa[0]].cpu().numpy().shape)


# --- Run FMN Attack (Foolbox) ---
print("\n**** FMN Attack ****\n")
fmodel = fb.PyTorchModel(model, bounds=(0, 1), device=device)
attack = fb.attacks.LInfFMNAttack()
#attack = fb.attacks.LinfProjectedGradientDescentAttack()
#attack = fb.attacks.LinfDeepFoolAttack()

raw, clipped, success = attack(fmodel, images, labels, epsilons=8/255)

# Predicted labels after FMN
_, predicted_fmn = model(clipped).max(1)
print(f"Predicted labels: {predicted_fmn}")
# Compare true labels with predicted labels after FMN
wrong_indexes_fmn = torch.nonzero(labels != predicted_fmn)
print(f"Wrong indexes FMN: {wrong_indexes_fmn}")

# Robustness of the model after FMN
correct = torch.sum(labels == predicted_fmn)
total = len(labels)
print(f"Correct predictions: {correct}/{total} ({correct/total*100:.2f}%)")
print(clipped[wrong_indexes_fmn[0]].cpu().numpy().shape)

# Samples misclassified by FMN but not by AA
wrong_fmn_not_aa = torch.nonzero((labels != predicted_fmn) & (labels == predicted_aa)).squeeze()
print(f"Wrongly classified by FMN but not by AA: {wrong_fmn_not_aa}")

# Samples misclassified by AA but not by FMN
wrong_aa_not_fmn = torch.nonzero((labels == predicted_fmn) & (labels != predicted_aa)).squeeze()
print(f"Wrongly classified by AA but not by FMN: {wrong_aa_not_fmn}")

# Samples that are classified differently by FMN and AA
diff = torch.nonzero(predicted_fmn != predicted_aa)
print(f"Different indexes: {diff}")

# True Labels, Predicted Labels after FMN, Predicted Labels after AA of the samples that are classified differently
for i in diff:
    print(f"Index: {i}")
    print(f"True label: {labels[i]}")
    print(f"Label after FMN: {predicted_fmn[i]}")
    print(f"Label after AA: {predicted_aa[i]}")
    print("\n")

# Plot images and perturbations of the samples that are classified differently
for i in diff:
    idx = i.item()
    true_label = labels[idx].item()
    label_fmn = predicted_fmn[idx].item()
    label_aa = predicted_aa[idx].item()

    original_image = images[idx].cpu().numpy().transpose(1, 2, 0)
    fmn_image = clipped[idx].cpu().numpy().transpose(1, 2, 0)
    aa_image = adv_images_autoattack[idx].cpu().numpy().transpose(1, 2, 0)
    perturbation_fmn = np.clip(fmn_image - original_image, 0, 1)
    perturbation_aa = np.clip(aa_image - original_image, 0, 1)

    fig, axs = plt.subplots(1, 5, figsize=(15, 3))
    axs[0].imshow(original_image)
    axs[0].set_title(f"Original\nLabel: {true_label}")
    axs[0].axis('off')

    axs[1].imshow(fmn_image)
    axs[1].set_title(f"FMN\nLabel: {label_fmn}")
    axs[1].axis('off')

    axs[2].imshow(perturbation_fmn * 10) # Scale the perturbation for better visualization
    axs[2].set_title("Perturbation FMN")
    axs[2].axis('off')

    axs[3].imshow(aa_image)
    axs[3].set_title(f"AA\nLabel: {label_aa}")
    axs[3].axis('off')

    axs[4].imshow(perturbation_aa * 10) # Scale the perturbation for better visualization
    axs[4].set_title("Perturbation AA")
    axs[4].axis('off')

    plt.show()
