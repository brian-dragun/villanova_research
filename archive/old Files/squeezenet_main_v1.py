import os
import torch
from squeezenet_train import train_model
from squeezenet_prune_model import prune_model
from squeezenet_evaluate_models import evaluate_model
from squeezenet_adversarial_test import test_adversarial_robustness
from squeezenet_robustness_test import apply_robustness_test  

# Define model paths
model_paths = {
    "original": "data/squeezenet_cifar10.pth",
    "pruned": "data/squeezenet_pruned.pth",
    "noisy": "data/squeezenet_noisy.pth"
}

# Ensure 'data' folder exists
if not os.path.exists("data"):
    os.makedirs("data")

print("\n🚀 **Step 1: Training the Original Model**")
if not os.path.exists(model_paths["original"]):
    train_model(model_paths["original"])
else:
    print(f"✅ Pre-trained model found at {model_paths['original']} - Skipping training.")

print("\n🔍 **Step 2: Pruning the Model**")
prune_model(model_paths["original"], model_paths["pruned"])

print("\n📊 **Step 3: Evaluating Model Performance**")
evaluate_model(model_paths["original"])
evaluate_model(model_paths["pruned"])
evaluate_model(model_paths["noisy"])

print("\n🎭 **Step 4: Applying Robustness Test (Adding Noise)**")
apply_robustness_test(model_paths["original"], model_paths["noisy"])

print("\n🛡 **Step 5: Adversarial Testing (FGSM Attack)**")
test_adversarial_robustness(model_paths["original"])
test_adversarial_robustness(model_paths["pruned"])
test_adversarial_robustness(model_paths["noisy"])

print("\n✅ **All steps completed successfully!**")
