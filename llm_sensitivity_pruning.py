import torch

def compute_gradient_sensitivity(model, inputs, targets):
    """
    Computes the gradient-based sensitivity for each parameter in the model.
    Returns a dictionary mapping parameter names to their absolute gradient values.
    """
    model.zero_grad()
    # Create a copy of the inputs and override the 'labels' key.
    inputs_copy = dict(inputs)
    inputs_copy["labels"] = targets

    outputs = model(**inputs_copy)
    loss = outputs.loss
    loss.backward()  # Backpropagate to compute gradients

    sensitivity = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            # Use the absolute gradient as a sensitivity measure
            grad_abs = param.grad.detach().abs()
            sensitivity[name] = grad_abs
            # Print summary statistics for this parameter's gradient sensitivity
            print(f"[DEBUG] {name}: min={grad_abs.min().item():.4f}, "
                  f"max={grad_abs.max().item():.4f}, mean={grad_abs.mean().item():.4f}, "
                  f"std={grad_abs.std().item():.4f}")
    return sensitivity

def prune_by_sensitivity(model, sensitivity_dict, prune_ratio=0.01, sample_size=1000000):
    """
    Prunes parameters in the model based on their gradient sensitivity.
    For each parameter, compute a quantile threshold on the sensitivity values.
    If the tensor is very large, a random subset of size `sample_size` is used
    to compute the quantile threshold.
    """
    total_pruned = 0
    total_elements = 0
    for name, sens_tensor in sensitivity_dict.items():
        sens_flat = sens_tensor.view(-1)
        num_elements = sens_flat.numel()
        # If too many elements, sample a subset
        if num_elements > sample_size:
            indices = torch.randperm(num_elements)[:sample_size]
            sens_sample = sens_flat[indices]
        else:
            sens_sample = sens_flat

        threshold = torch.quantile(sens_sample.cpu(), prune_ratio)
        num_pruned = (sens_flat < threshold).sum().item()
        total_pruned += num_pruned
        total_elements += num_elements

        print(f"[DEBUG] Pruning {name}: threshold={threshold.item():.6f}, "
              f"pruning {num_pruned} / {num_elements} elements ({100 * num_pruned / num_elements:.2f}%)")

        # Retrieve the parameter from the model and apply the mask (on the same device)
        param = dict(model.named_parameters())[name]
        mask = (sens_tensor >= threshold).float().to(param.device)
        # Zero out weights below the sensitivity threshold (in-place)
        param.data.mul_(mask)

    print(f"[DEBUG] Total pruned weights: {total_pruned} / {total_elements} "
          f"({100 * total_pruned / total_elements:.2f}%)")
    return model
