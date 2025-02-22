import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader

###############################
# Part 1: Hessian Sensitivity Analysis
###############################

def compute_hessian_sensitivity_dict(model, dataloader, device):
    """
    Compute a Hessian-based sensitivity score for each parameter in the model.
    For each parameter, we compute the first derivative (gradient) of a simple loss
    with respect to that parameter (with create_graph and retain_graph enabled),
    then compute the second derivative (diagonal of the Hessian) and sum its absolute values.
    """
    model.eval()
    sensitivity_scores = {}
    # Get a single batch of data
    images, labels = next(iter(dataloader))
    images, labels = images.to(device), labels.to(device)
    
    # Forward pass: use sum(outputs) as a simple scalar loss
    outputs = model(images)
    loss = torch.sum(outputs)
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            # Compute the gradient with retain_graph=True to keep the graph alive for subsequent backward passes
            grad = torch.autograd.grad(loss, param, create_graph=True, retain_graph=True)[0]
            # Compute the second derivative (diagonal of Hessian)
            hessian_diag = torch.autograd.grad(grad, param, grad_outputs=torch.ones_like(grad), retain_graph=True)[0]
            sensitivity = hessian_diag.abs().sum().item()
            sensitivity_scores[name] = sensitivity
    return sensitivity_scores


###############################
# Part 2: Super Weight Outlier Detection
###############################

def compute_weight_statistics(model):
    """
    Compute basic statistics (mean, std, max absolute value) for each weight tensor.
    Returns a dictionary mapping parameter names to their statistics.
    """
    stats = {}
    for name, param in model.named_parameters():
        if "weight" in name:
            weights = param.detach().cpu().numpy().flatten()
            stats[name] = {
                'mean': np.mean(weights),
                'std': np.std(weights),
                'max': np.max(np.abs(weights))
            }
    return stats

def identify_super_weights(model, z_threshold=2.5):
    """
    Identify super weights by computing the Z-score for every weight value based on its layer's statistics.
    Flags those weight indices where the absolute Z-score exceeds the z_threshold.
    Returns a dictionary mapping parameter names to arrays of indices that are considered outliers.
    """
    stats = compute_weight_statistics(model)
    super_weights = {}
    for name, param in model.named_parameters():
        if "weight" in name:
            weights = param.detach().cpu().numpy().flatten()
            mean_val = stats[name]['mean']
            std_val = stats[name]['std'] + 1e-8  # avoid division by zero
            z_scores = np.abs((weights - mean_val) / std_val)
            super_indices = np.where(z_scores > z_threshold)[0]
            if len(super_indices) > 0:
                super_weights[name] = super_indices
    return super_weights

###############################
# Part 3: Integrated Analysis
###############################

def main():
    # Set up device and data loader (using CIFAR-10 test set)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
       transforms.ToTensor(),
       transforms.Normalize((0.5,), (0.5,))
    ])
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=1, shuffle=True)
    
    # Load SqueezeNet and adjust for CIFAR-10 (10 classes)
    model = models.squeezenet1_0(weights=None)
    model.classifier[1] = nn.Conv2d(512, 10, kernel_size=(1, 1))
    model.load_state_dict(torch.load("data/squeezenet_cifar10.pth"))
    model.to(device)
    model.eval()
    
    # Compute Hessian-based sensitivity scores for each parameter
    sensitivity_scores = compute_hessian_sensitivity_dict(model, testloader, device)
    
    # Identify super weights (statistical outliers) using a Z-score threshold
    z_threshold = 2.5  # adjust this threshold as needed
    super_weight_indices = identify_super_weights(model, z_threshold=z_threshold)
    
    # Print integrated analysis for each parameter
    print("Integrated Analysis of Weight Importance:")
    for name in sensitivity_scores:
        print(f"Parameter: {name}")
        print(f"  Hessian Sensitivity Score: {sensitivity_scores[name]:.4f}")
        if name in super_weight_indices:
            print(f"  Super Weight Outlier Indices (Z-score > {z_threshold}): {super_weight_indices[name]}")
        else:
            print("  No super weight outliers detected.")
        print()
    
    # Optionally, plot a bar chart of sensitivity scores for visualization
    names = list(sensitivity_scores.keys())
    scores = [sensitivity_scores[n] for n in names]
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(names)), scores, color='skyblue')
    plt.xticks(range(len(names)), names, rotation=90)
    plt.xlabel("Parameter")
    plt.ylabel("Hessian Sensitivity Score")
    plt.title("Per-Parameter Sensitivity Scores")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
