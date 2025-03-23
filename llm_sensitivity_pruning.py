import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import MODEL_NAME, TEST_PROMPT

def compute_gradient_sensitivity(model, inputs, targets):
    model.zero_grad()
    outputs = model(**inputs)
    loss = outputs.loss if hasattr(outputs, "loss") and outputs.loss is not None else outputs.logits.mean()
    loss.backward(retain_graph=True)
    
    sensitivity_scores = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            # Save a copy of the absolute gradient values.
            sensitivity_scores[name] = param.grad.abs().detach().clone()
    return sensitivity_scores

def prune_by_sensitivity(model, sensitivity_scores, prune_ratio=0.05, sample_size=1000000):
    """
    Prune weights based on sensitivity scores. If the sensitivity tensor is huge,
    randomly sample a subset for quantile calculation.
    """
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in sensitivity_scores:
                sens_flat = sensitivity_scores[name].view(-1)
                # If the number of elements is huge, sample a subset for quantile calculation.
                if sens_flat.numel() > sample_size:
                    # Randomly permute indices and take the first 'sample_size' elements.
                    indices = torch.randperm(sens_flat.numel(), device=sens_flat.device)[:sample_size]
                    sens_sample = sens_flat[indices]
                    threshold = torch.quantile(sens_sample.cpu(), prune_ratio)
                else:
                    threshold = torch.quantile(sens_flat.cpu(), prune_ratio)
                # Move threshold back to the original device
                threshold = threshold.to(sens_flat.device)
                mask = sensitivity_scores[name] >= threshold
                param.data.mul_(mask)
    return model

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model.to(device)
    model.train()  # Enable gradients
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    
    inputs = tokenizer(TEST_PROMPT, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    targets = inputs["input_ids"]
    
    print("Computing gradient sensitivity scores...")
    sens_scores = compute_gradient_sensitivity(model, inputs, targets)
    
    prune_ratio = 0.05  # Prune the lowest 5% sensitive weights per parameter
    print(f"Pruning weights with the lowest {prune_ratio*100}% sensitivity...")
    model = prune_by_sensitivity(model, sens_scores, prune_ratio=prune_ratio)
    
    pruned_model_path = "data/llm_sensitivity_pruned"
    model.save_pretrained(pruned_model_path)
    print(f"Pruned model saved to {pruned_model_path}")
