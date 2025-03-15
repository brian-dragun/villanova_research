import torch
import torch.autograd as autograd
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "meta-llama/Llama-2-7b"  # update as needed

def compute_hessian_sensitivity(model, input_text):
    """
    Computes Hessian-based sensitivity scores for LLM weights given an input text.
    """
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model(**inputs)
    # Use model loss if available; otherwise, a mean over logits as a proxy
    loss = outputs.loss if hasattr(outputs, "loss") and outputs.loss is not None else outputs.logits.mean()
    
    grads = autograd.grad(loss, model.parameters(), create_graph=True)
    
    sensitivity_scores = {}
    for (name, param), grad in zip(model.named_parameters(), grads):
        if grad is not None:
            hessian_diag = autograd.grad(grad, param, grad_outputs=torch.ones_like(grad), retain_graph=True)[0]
            sensitivity_scores[name] = hessian_diag.abs().sum().item()
    return sensitivity_scores

def plot_sensitivity(sensitivity_scores):
    names = list(sensitivity_scores.keys())
    scores = [sensitivity_scores[name] for name in names]
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(names)), scores)
    plt.xticks(range(len(names)), names, rotation=90)
    plt.xlabel("Parameter")
    plt.ylabel("Hessian Sensitivity Score")
    plt.title("Sensitivity Scores for LLM Weights")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    test_text = "The quick brown fox jumps over the lazy dog."
    scores = compute_hessian_sensitivity(model, test_text)
    print("Hessian Sensitivity Scores:")
    for name, score in scores.items():
        print(f"{name}: {score:.4f}")
    plot_sensitivity(scores)
