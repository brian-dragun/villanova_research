import torch
import math
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import MODEL_NAME, TEST_PROMPT

def evaluate_perplexity(model, tokenizer, prompt):
    model.eval()
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss.item()
    return math.exp(loss)

def layer_ablation_experiment(model, tokenizer, prompt, layers_to_ablate):
    """
    Zero out weights for each layer in layers_to_ablate, measure perplexity.
    """
    baseline = evaluate_perplexity(model, tokenizer, prompt)
    results = {}
    
    param_dict = dict(model.named_parameters())
    for layer_name in layers_to_ablate:
        if layer_name not in param_dict:
            print(f"Layer '{layer_name}' not found in model; skipping.")
            continue
        
        param = param_dict[layer_name]
        backup = param.detach().clone()
        with torch.no_grad():
            param.zero_()
        new_ppl = evaluate_perplexity(model, tokenizer, prompt)
        results[layer_name] = new_ppl
        with torch.no_grad():
            param.copy_(backup)
    
    return baseline, results

def main():
    print(f"Using model: {MODEL_NAME}")
    test_prompts = [TEST_PROMPT]

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Print all param names (optional debug)
    # for n, p in model.named_parameters():
    #     print(n)
    
    # Decide layer names depending on LLaMA vs GPT
    model_name_lower = MODEL_NAME.lower()
    if "llama" in model_name_lower:
        # LLaMA 2 uses something like "model.embed_tokens.weight", "model.layers.0.mlp.gate_proj.weight"
        # Example: let's ablate "model.layers.0.mlp.gate_proj.weight"
        layers_to_ablate = ["model.layers.0.mlp.gate_proj.weight"]
    else:
        # GPT style
        layers_to_ablate = ["transformer.h.0.mlp.c_fc.weight"]

    # 1) Layer ablation experiment
    ablation_prompt = test_prompts[0]
    baseline_ablation, ablation_results = layer_ablation_experiment(model, tokenizer, ablation_prompt, layers_to_ablate)
    print(f"\nBaseline perplexity on prompt '{ablation_prompt}': {baseline_ablation:.2f}")
    print("Layer Ablation Results:")
    for layer, ppl in ablation_results.items():
        print(f"Layer: {layer} -> Perplexity after ablation: {ppl:.2f}")

    # 2) Possibly do scaling experiment, fisher info, etc.
    # ...

if __name__ == "__main__":
    main()
