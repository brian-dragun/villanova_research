import os
import torch
from squeezenet_train import train_model
from squeezenet_prune_model import prune_model
from squeezenet_evaluate_models import evaluate_model
from squeezenet_adversarial_test import test_adversarial_robustness
from squeezenet_robustness_test import apply_robustness_test

# New imports for integrated analysis:
from squeezenet_integrated_analysis import compute_hessian_sensitivity_dict
from squeezenet_super_weights import identify_super_weights

from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt

def run_integrated_analysis():
    print("\n🔍 **Step 6: Integrated Sensitivity and Super Weight Analysis**")
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
    
    # Identify super weights using a Z-score threshold (e.g., 2.5)
    z_threshold = 2.5
    super_weight_indices = identify_super_weights(model, z_threshold=z_threshold)
    
    # Print integrated analysis results
    print("\nIntegrated Analysis of Weight Importance:")
    for name in sensitivity_scores:
        print(f"Parameter: {name}")
        print(f"  Hessian Sensitivity Score: {sensitivity_scores[name]:.4f}")
        if name in super_weight_indices:
            print(f"  Super Weight Outlier Indices (Z-score > {z_threshold}): {super_weight_indices[name]}")
        else:
            print("  No super weight outliers detected.")
        print()
    
    # Optional: Plot a bar chart of sensitivity scores for visualization
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
    model_paths = {
        "original": "data/squeezenet_cifar10.pth",
        "pruned": "data/squeezenet_pruned.pth",
        "noisy": "data/squeezenet_noisy.pth"
    }
    
    # Step 1: Train or load the original model
    print("\n🚀 **Step 1: Training the Original Model**")
    if not os.path.exists(model_paths["original"]):
        train_model(model_paths["original"])
    else:
        print(f"✅ Pre-trained model found at {model_paths['original']} - Skipping training.")
    
    # Step 2: Prune the model
    print("\n🔍 **Step 2: Pruning the Model**")
    prune_model(model_paths["original"], model_paths["pruned"])
    
    # Step 3: Evaluate the models
    print("\n📊 **Step 3: Evaluating Model Performance**")
    evaluate_model(model_paths["original"])
    evaluate_model(model_paths["pruned"])
    evaluate_model(model_paths["noisy"])
    
    # Step 4: Apply robustness test (adding noise)
    print("\n🎭 **Step 4: Applying Robustness Test (Adding Noise)**")
    apply_robustness_test(model_paths["original"], model_paths["noisy"])
    
    # Step 5: Adversarial testing (FGSM attack)
    print("\n🛡 **Step 5: Adversarial Testing (FGSM Attack)**")
    test_adversarial_robustness(model_paths["original"])
    test_adversarial_robustness(model_paths["pruned"])
    test_adversarial_robustness(model_paths["noisy"])
    
    # Step 6: Integrated Sensitivity and Super Weight Analysis
    run_integrated_analysis()
    
    print("\n✅ **All steps completed successfully!**")
