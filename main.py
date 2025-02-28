import torch
from robustbench.data import load_cifar10
from robustbench.utils import load_model
from robustbench.eval import benchmark
import foolbox as fb
from autoattack import AutoAttack

# Detect device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Select model
model_name = "Carmon2019Unlabeled"
#model_name = "Wang2023Better_WRN-28-10"
#model_name = "Cui2023Decoupled_WRN-28-10"
#model_name = "Xu2023Exploring_WRN-28-10	"
#model_name = "Huang2022Revisiting_WRN-A4	"

# Evaluate each model with AutoAttack

print(f"\nEvaluating model: {model_name}")

# Load model and move to device (CUDA or CPU)
model = load_model(model_name=model_name, dataset="cifar10", threat_model="Linf").to(device)
model.eval()

# Load CIFAR-10 test samples
images, labels = load_cifar10(n_examples=50)
images, labels = images.clone().detach().to(device), labels.clone().detach().to(device)

print(f"Images shape: {images.shape}, Labels shape: {labels.shape}")
print(f"Images dtype: {images.dtype}, Labels dtype: {labels.dtype}")

# --- Run AutoAttack Manually ---
autoattack = AutoAttack(model, norm='Linf', eps=8/255, version='standard', device=device)
adv_images_autoattack = autoattack.run_standard_evaluation(images, labels, bs=50)

# Check which samples were misclassified
logits = model(adv_images_autoattack)
preds_autoattack = logits.argmax(dim=1)
autoattack_success = preds_autoattack != labels  # True if misclassified

# --- Run FMN Attack (Foolbox) ---
fmodel = fb.PyTorchModel(model, bounds=(0, 1), device =device)
attack = fb.attacks.LInfFMNAttack()
#attack = fb.attacks.LinfProjectedGradientDescentAttack()
adv_images_fmn, _, success = attack(fmodel, images, labels, epsilons=[8/255])

# Convert Foolbox attack results
fmn_success = success.squeeze(1)  # True if misclassified

# Calculate robustness percentages
fmn_robustness = 100 - (fmn_success.sum().item() / len(images)) * 100
print(f"Model Robustness after FMN Attack: {fmn_robustness:.2f}%")

# --- Compare Attacks ---
autoattack_misclassified = (autoattack_success).nonzero().squeeze().tolist()
fmn_misclassified = (fmn_success).nonzero().squeeze().tolist()
only_autoattack_misclassified = (autoattack_success & ~fmn_success).nonzero().squeeze().tolist()
only_fmn_misclassified = (fmn_success & ~autoattack_success).nonzero().squeeze().tolist()

print(f"\nSamples misclassified by AutoAttack: {autoattack_misclassified}")
print(f"Samples misclassified by FMN Attack: {fmn_misclassified}")
print(f"\nSamples misclassified only by AutoAttack: {only_autoattack_misclassified}")
print(f"Samples misclassified only by FMN Attack: {only_fmn_misclassified}")