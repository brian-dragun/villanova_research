import torch
import torchvision.models as models
import torch.nn as nn


prune_ratio = 0.05

# Load trained SqueezeNet model
model = models.squeezenet1_0(weights=None)
model.classifier[1] = torch.nn.Conv2d(512, 10, kernel_size=(1, 1))

# Load trained weights
model.load_state_dict(torch.load("data/squeezenet_cifar10.pth"))
model.eval()

# Compute sensitivity scores
def compute_sensitivity(model):
    sensitivity_scores = {}
    for name, param in model.named_parameters():
        if "weight" in name and param.requires_grad:
            sensitivity_scores[name] = param.abs().mean().item()
    return sensitivity_scores

# Prune model based on sensitivity scores
def prune_layerwise(model, sensitivity_scores, prune_ratio=0.05):
    for name, param in model.named_parameters():
        if "weight" in name and name in sensitivity_scores:
            threshold = torch.tensor(sensitivity_scores[name]).quantile(prune_ratio)
            mask = param.abs() >= threshold
            param.data.mul_(mask)
    return model


# Compute sensitivity scores
sensitivity_scores = compute_sensitivity(model)
print("Computed Sensitivity Scores:", sensitivity_scores)

# Keep essential layers while pruning
pruned_features = torch.nn.Sequential(*list(model.features.children())[:-2])  # Keep most features
pruned_classifier = torch.nn.Sequential(
    torch.nn.AdaptiveAvgPool2d((1, 1)),  # Ensures final layer shape
    torch.nn.Flatten(),
    torch.nn.Linear(512, 10)  # Maintains 10 output classes
)

# Create final pruned model
pruned_model = torch.nn.Sequential(pruned_features, pruned_classifier)
pruned_model = prune_layerwise(model, sensitivity_scores, prune_ratio)


# Save full pruned model
torch.save(pruned_model, "data/squeezenet_pruned.pth")
print("Model pruned based on sensitivity and saved as data/squeezenet_pruned.pth")
