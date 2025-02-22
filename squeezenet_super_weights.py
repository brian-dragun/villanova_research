import torch
import torch.nn as nn
import numpy as np
from torchvision import models

def compute_weight_statistics(model):
    """
    Compute basic statistics for each weight tensor in the model.
    Returns a dictionary mapping parameter names to their mean, standard deviation, and maximum absolute value.
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
    Identify super weights using a Z-score threshold.
    
    For each weight tensor in the model, compute the Z-score for every weight value based on the tensor's mean and standard deviation.
    We flag those weights whose absolute Z-score exceeds the provided threshold.
    
    Returns a dictionary mapping parameter names to the indices of the weights that exceed the threshold.
    """
    stats = compute_weight_statistics(model)
    super_weights = {}
    for name, param in model.named_parameters():
        if "weight" in name:
            weights = param.detach().cpu().numpy().flatten()
            mean_val = stats[name]['mean']
            std_val = stats[name]['std'] + 1e-8  # avoid division by zero
            z_scores = np.abs((weights - mean_val) / std_val)
            # Identify indices where the Z-score exceeds the threshold
            super_indices = np.where(z_scores > z_threshold)[0]
            if len(super_indices) > 0:
                super_weights[name] = super_indices
    return super_weights

def main():
    # Load SqueezeNet and modify the classifier for CIFAR-10 (10 classes)
    model = models.squeezenet1_0(weights=None)
    model.classifier[1] = nn.Conv2d(512, 10, kernel_size=(1, 1))
    
    # Load pretrained weights (make sure the path is correct)
    model.load_state_dict(torch.load("data/squeezenet_cifar10.pth"))
    model.eval()

    # Identify super weights using a chosen Z-score threshold
    z_threshold = 2.5  # Adjust this threshold as needed for your analysis
    super_weights = identify_super_weights(model, z_threshold=z_threshold)

    # Print out the results
    print("Identified super weights (weights with absolute Z-score > {}):".format(z_threshold))
    for layer, indices in super_weights.items():
        print(f"Layer: {layer}")
        print(f"  Super weight indices: {indices}\n")

if __name__ == "__main__":
    main()
