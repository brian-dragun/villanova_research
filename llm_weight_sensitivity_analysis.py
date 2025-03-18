import os
import math
import copy
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import MODEL_NAME, TEST_PROMPT, EPSILON

# --- Evaluation Helper ---
def evaluate_perplexity(model, tokenizer, prompt):
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)
    outputs = model(**inputs, labels=inputs.input_ids)
    loss = outputs.loss.item()
    perplexity = math.exp(loss)
    return perplexity

# --- Layer Ablation Experiment ---
def layer_ablation_experiment(model, tokenizer, prompt, layers_to_ablate):
    """
    Zeroes out the weights for each specified layer (by parameter name) and returns
    the perplexity after ablation.
    """
    baseline = evaluate_perplexity(model, tokenizer, prompt)
    results = {}
    for layer_name in layers_to_ablate:
        param = dict(model.named_parameters()).get(layer_name, None)
        if param is None:
            print(f"Layer {layer_name} not found.")
            continue
        backup = param.detach().clone()
        with torch.no_grad():
            param.zero_()
        new_perplexity = evaluate_perplexity(model, tokenizer, prompt)
        results[layer_name] = new_perplexity
        with torch.no_grad():
            param.copy_(backup)
    return baseline, results

# --- Weight Scaling Experiment ---
def weight_scaling_experiment(model, tokenizer, prompt, layer_name, scaling_factors):
    """
    Multiply the weights of a given layer by different scaling factors and compute
    the perplexity for each factor.
    """
    param = dict(model.named_parameters()).get(layer_name, None)
    if param is None:
        raise ValueError(f"Layer {layer_name} not found.")
    baseline = evaluate_perplexity(model, tokenizer, prompt)
    results = {}
    backup = param.detach().clone()
    for factor in scaling_factors:
        with torch.no_grad():
            param.mul_(factor)
        new_perplexity = evaluate_perplexity(model, tokenizer, prompt)
        results[factor] = new_perplexity
        with torch.no_grad():
            param.copy_(backup)
    return baseline, results

# --- Fisher Information Experiment ---
def fisher_information_experiment(model, tokenizer, prompt, num_samples=20):
    """
    Compute an approximate diagonal Fisher Information for each parameter by averaging
    over several forward/backward passes using the same prompt.
    """
    model.eval()
    fisher_info = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            fisher_info[name] = torch.zeros_like(param)
    loss_fn = torch.nn.CrossEntropyLoss()
    for _ in range(num_samples):
        inputs = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)
        model.zero_grad()
        outputs = model(**inputs, labels=inputs.input_ids)
        loss = outputs.loss
        loss.backward()
        for name, param in model.named_parameters():
            if param.grad is not None:
                fisher_info[name] += param.grad.detach() ** 2
    for name in fisher_info:
        fisher_info[name] /= num_samples
    return fisher_info

# --- Main function to run experiments ---
def main():
    # You can either use a single prompt from config or extend this list.
    # For example, you could define test_prompts = [TEST_PROMPT, "Another prompt", ...]
    test_prompts = [TEST_PROMPT]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using model: {MODEL_NAME}")
    print(f"Using test prompt(s): {test_prompts}")
    
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    print("### Baseline Perplexities ###")
    for prompt in test_prompts:
        ppl = evaluate_perplexity(model, tokenizer, prompt)
        print(f"Prompt: '{prompt}' -> Perplexity: {ppl:.2f}")
    
    # Run layer ablation experiment on selected layers using the first prompt.
    ablation_prompt = test_prompts[0]
    layers_to_ablate = [
        "transformer.wte.weight",        # Input embeddings
        "transformer.h.0.mlp.c_fc.weight"  # Example layer in the first block's feedforward network
    ]
    baseline_ablation, ablation_results = layer_ablation_experiment(model, tokenizer, ablation_prompt, layers_to_ablate)
    print(f"\nBaseline perplexity on prompt '{ablation_prompt}': {baseline_ablation:.2f}")
    print("Layer Ablation Results:")
    for layer, ppl in ablation_results.items():
        print(f"Layer: {layer} -> Perplexity after ablation: {ppl:.2f}")
    
    # Run weight scaling experiment on a selected layer.
    scaling_prompt = test_prompts[0]
    scaling_factors = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
    layer_name = "transformer.h.0.mlp.c_fc.weight"  # Change this as needed.
    baseline_scale, scaling_results = weight_scaling_experiment(model, tokenizer, scaling_prompt, layer_name, scaling_factors)
    print(f"\nBaseline perplexity on prompt '{scaling_prompt}': {baseline_scale:.2f}")
    print(f"Weight Scaling Results for layer {layer_name}:")
    for factor, ppl in scaling_results.items():
        print(f"Scaling factor: {factor} -> Perplexity: {ppl:.2f}")
    
    # Compute Fisher Information for the first prompt.
    fisher_info = fisher_information_experiment(model, tokenizer, test_prompts[0])
    print("\nAverage Fisher Information (per parameter):")
    for name, fi in fisher_info.items():
        avg_fi = fi.mean().item()
        print(f"Parameter: {name} -> Average FI: {avg_fi:.4e}")

if __name__ == "__main__":
    main()
