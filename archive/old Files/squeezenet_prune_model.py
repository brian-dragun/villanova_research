import torch
import torchvision.models as models
import torch.nn as nn

def compute_sensitivity(model):
    """Computes sensitivity scores for each weight parameter."""
    sensitivity_scores = {}
    for name, param in model.named_parameters():
        if "weight" in name and param.requires_grad:
            sensitivity_scores[name] = param.abs().mean().item()
    return sensitivity_scores

def prune_layerwise(model, sensitivity_scores, prune_ratio=0.05):
    """Prunes model weights based on sensitivity scores in-place."""
    for name, param in model.named_parameters():
        if "weight" in name and name in sensitivity_scores:
            threshold = prune_ratio * sensitivity_scores[name]
            mask = param.abs() >= threshold
            # In-place multiplication keeps the original parameter names intact.
            param.data.mul_(mask)
    return model

def prune_model(model_path, pruned_model_path, prune_ratio=0.05):
    """Loads a trained SqueezeNet model, prunes its weights in-place, and saves its state dict."""
    # Load the original SqueezeNet model
    model = models.squeezenet1_0(weights=None)
    # Modify the classifier for CIFAR-10 (10 classes)
    model.classifier[1] = nn.Conv2d(512, 10, kernel_size=(1, 1))
    # Load the pretrained weights
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Compute sensitivity scores
    sensitivity_scores = compute_sensitivity(model)
    print("Computed Sensitivity Scores:", sensitivity_scores)

    # Prune the model in-place so that its architecture remains the same.
    model = prune_layerwise(model, sensitivity_scores, prune_ratio)

    # Save only the state dictionary, preserving the original key names.
    torch.save(model.state_dict(), pruned_model_path)
    print(f"✅ Model pruned based on sensitivity and saved as {pruned_model_path}")

# Run the pruning script if executed directly.
if __name__ == "__main__":
    prune_model("data/squeezenet_cifar10.pth", "data/squeezenet_pruned.pth")
