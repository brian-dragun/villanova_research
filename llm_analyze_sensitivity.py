import torch
import torch.autograd as autograd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from config import MODEL_NAME, TEST_PROMPT

def compute_hessian_sensitivity(model, input_text, device=torch.device("cpu")):
    """
    Computes Hessian-based sensitivity scores for LLM weights given an input text.
    Uses the slow tokenizer (use_fast=False) to avoid tiktoken conversion issues.
    """
    model.eval()
    # Force slow tokenizer to avoid tiktoken conversion issues.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    inputs = tokenizer(input_text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    outputs = model(**inputs)
    # Use model loss if available; otherwise, use the mean of logits as a proxy.
    loss = outputs.loss if hasattr(outputs, "loss") and outputs.loss is not None else outputs.logits.mean()
    
    # Compute first-order gradients.
    grads = autograd.grad(loss, model.parameters(), create_graph=True)
    
    sensitivity_scores = {}
    for (name, param), grad in zip(model.named_parameters(), grads):
        if grad is not None:
            # Compute the diagonal of the Hessian for the parameter.
            hessian_diag = autograd.grad(
                grad, param, 
                grad_outputs=torch.ones_like(grad), 
                retain_graph=True, 
                allow_unused=True
            )[0]
            if hessian_diag is not None:
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
    plt.savefig("llm_diagram_sensitivity_plot.png")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model.to(device)
    test_text = TEST_PROMPT
    sensitivity_scores = compute_hessian_sensitivity(model, test_text, device=device)
    print("Hessian Sensitivity Scores:")
    for name, score in sensitivity_scores.items():
        print(f"{name}: {score:.4f}")
    plot_sensitivity(sensitivity_scores)
