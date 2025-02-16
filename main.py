import torch
from robustbench.data import load_cifar10
from robustbench.utils import load_model
from robustbench.eval import benchmark

# Detect device (use MPS if available, otherwise CPU)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Load 50 samples from CIFAR-10 test set
x_test, y_test = load_cifar10(n_examples=50)
x_test, y_test = x_test.to(device), y_test.to(device)  # Move data to device

# Select 5 models from the RobustBench leaderboard
model_names = [
    "Carmon2019Unlabeled",
    "Sehwag2021Proxy",
    "Gowal2021Improving_R18_ddpm_100m",
    "Rebuffi2021Fixing_70_16_cutmix_ddpm",
    "Huang2023Random"
]

# Evaluate each model with AutoAttack
for model_name in model_names:
    print(f"\nEvaluating model: {model_name}")

    # Load model and move to device (MPS or CPU)
    model = load_model(model_name=model_name, dataset="cifar10", threat_model="Linf").to(device)
    model.eval()

    # Run benchmark (removing 'attack' argument)
    clean_acc, adv_acc = benchmark(model, n_examples=50, dataset="cifar10", threat_model="Linf", eps=8 / 255, device=device, batch_size=25)

    print(f"Clean accuracy: {clean_acc:.1%}, Robust accuracy: {adv_acc:.1%}")

print("\nEvaluation complete.")