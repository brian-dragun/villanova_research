import torch

def compute_gradient_sensitivity(model, inputs, targets=None):
    """
    Computes gradient-based sensitivity for each parameter.
    Assumes the `inputs` dict already includes the "labels" key.
    Returns a dictionary mapping parameter names to a flattened tensor of absolute gradients.
    """
    model.zero_grad()
    # Call the model with inputs; "labels" is already in inputs.
    outputs = model(**inputs)
    loss = outputs.loss
    loss.backward()
    
    sensitivity_dict = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            sensitivity_dict[name] = param.grad.detach().abs().view(-1)
    return sensitivity_dict

def prune_by_sensitivity(model, sensitivity_dict, prune_ratio=0.01):
    """
    For each parameter, sample the gradient sensitivity if it is too large,
    compute the quantile threshold, and zero out weights with absolute value
    below that threshold.
    
    prune_ratio: quantile for threshold (e.g. 0.01 for the bottom 1%).
    """
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.requires_grad and name in sensitivity_dict:
                sens = sensitivity_dict[name]
                # If the sensitivity tensor is huge, sample 1,000,000 values.
                if sens.numel() > 1_000_000:
                    indices = torch.randperm(sens.numel())[:1_000_000]
                    sens_sample = sens[indices]
                else:
                    sens_sample = sens
                # Compute threshold on the CPU to avoid device issues.
                threshold = torch.quantile(sens_sample.cpu(), prune_ratio)
                mask = torch.abs(param) >= threshold
                param.mul_(mask.to(param.device))
    return model
