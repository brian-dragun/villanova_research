import os
import time
import torch
import torch.autograd as autograd
import matplotlib
matplotlib.use("Agg")  # Ensure compatibility with non-GUI environments
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import MODEL_NAME, TEST_PROMPT

def compute_hessian_sensitivity(model, input_text, device=torch.device("cpu")):
    """
    Compute Hessian-based sensitivity scores for model parameters.
    If Hessian computation fails, fall back to gradient-based sensitivity.
    """
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    inputs = tokenizer(input_text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model(**inputs)
    loss = outputs.loss if hasattr(outputs, "loss") and outputs.loss is not None else outputs.logits.mean()
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)

    sensitivity_scores = {}
    try:
        for (name, param), grad in zip(model.named_parameters(), grads):
            if grad is not None:
                # Compute the diagonal of the Hessian
                hessian_diag = torch.autograd.grad(
                    grad, param,
                    grad_outputs=torch.ones_like(grad, device=grad.device, requires_grad=False),
                    retain_graph=True,
                    allow_unused=True
                )[0]
                if hessian_diag is not None:
                    sensitivity_scores[name] = hessian_diag.abs().sum().item()
        return sensitivity_scores
    except RuntimeError as e:
        print("⚠️ Hessian computation failed:", e)
        print("Falling back to gradient-based sensitivity.")

        # Fallback: use sum of absolute gradients as sensitivity
        grad_sensitivity = {}
        for (name, param), grad in zip(model.named_parameters(), grads):
            if grad is not None:
                grad_sensitivity[name] = grad.abs().sum().item()
        return grad_sensitivity

def plot_sensitivity(sensitivity_scores):
    """
    Plot Hessian Sensitivity Scores and save the figure.
    """
    output_dir = "outputs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Generate unique filename with timestamp
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = os.path.join(output_dir, f"sensitivity_plot_{timestamp}.png")

    names = list(sensitivity_scores.keys())
    scores = [sensitivity_scores[name] for name in names]

    plt.figure(figsize=(12, 6))
    plt.bar(range(len(names)), scores)
    plt.xticks(range(len(names)), names, rotation=90)
    plt.xlabel("Parameter")
    plt.ylabel("Hessian Sensitivity Score")
    plt.title("Sensitivity Scores for LLM Weights")
    plt.tight_layout()

    # Save before closing the figure
    plt.savefig(filename)
    print(f"✅ Saved plot: {filename}")

    # Free memory
    plt.close()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the model
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model.to(device)

    # Run Hessian Sensitivity Analysis
    test_text = TEST_PROMPT
    sensitivity_scores = compute_hessian_sensitivity(model, test_text, device=device)

    print("\n🔍 Hessian Sensitivity Scores:")
    for name, score in sensitivity_scores.items():
        print(f"{name}: {score:.4f}")

    # Plot and save results
    plot_sensitivity(sensitivity_scores)
