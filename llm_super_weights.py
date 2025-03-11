import torch
import numpy as np
from transformers import AutoModelForCausalLM

def compute_weight_statistics(model):
    """
    Compute basic statistics for each weight tensor in the model.
    Returns a dictionary mapping parameter names to their mean, std, and maximum absolute value.
    """
    stats = {}
    for name, param in model.named_parameters():
        if "weight" in name and param.requires_grad:
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
    Returns a dictionary mapping parameter names to the indices of weights whose absolute Z-score exceeds the threshold.
    """
    stats = compute_weight_statistics(model)
    super_weights = {}
    for name, param in model.named_parameters():
        if "weight" in name and param.requires_grad:
            weights = param.detach().cpu().numpy().flatten()
            mean_val = stats[name]['mean']
            std_val = stats[name]['std'] + 1e-8  # avoid division by zero
            z_scores = np.abs((weights - mean_val) / std_val)
            indices = np.where(z_scores > z_threshold)[0]
            if len(indices) > 0:
                super_weights[name] = indices
    return super_weights

def main():
    MODEL_NAME = "meta-llama/Llama-2-7b"  # update as needed
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.eval()
    super_weights = identify_super_weights(model, z_threshold=2.5)
    print("Identified super weights (Z-score > 2.5):")
    for layer, indices in super_weights.items():
        print(f"Layer: {layer}, Super weight indices: {indices}")

if __name__ == "__main__":
    main()
