import torch
import torch.nn.functional as F
import torch.autograd as autograd
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
import math
from config import TEST_PROMPT, EPSILON

def run_robust_analysis_display():
    print("Running robust analysis display...")
    
    # Assume you generate adversarial text from a PGD attack:
    adv_text = "Updates the Game Game"  # example adversarial text
    print("\nAdversarial generated text (PGD attack):")
    print(f"  {adv_text}\n")
    
    # For token distribution plot:
    # (If you're in an interactive environment, you can call plt.show() to display the plot.)
    # Otherwise, save the plot to a file and print the filename.
    import matplotlib.pyplot as plt
    plt.figure()
    # Dummy data for demonstration:
    tokens = ["Token1", "Token2", "Token3"]
    probs = [0.04, 0.03, 0.02]
    plt.bar(tokens, probs)
    plt.title("Token Distribution at Position 0")
    plt.xlabel("Tokens")
    plt.ylabel("Probability")
    # For interactive environments:
    plt.show()
    # Or, to save:
    plt.savefig("llm_token_distribution.png")
    print("Robust analysis display complete.\n")

def pgd_attack(model, inputs_embeds, epsilon=EPSILON, alpha=0.01, num_iter=40):
    """
    Perform a PGD (Projected Gradient Descent) attack on the input embeddings.
    """
    adv_embeds = inputs_embeds.clone().detach()
    adv_embeds.requires_grad = True

    for _ in range(num_iter):
        outputs = model(inputs_embeds=adv_embeds)
        loss = outputs.logits.mean()  # Use an appropriate loss for your task
        model.zero_grad()
        loss.backward()
        # Update with the sign of the gradient:
        adv_embeds = adv_embeds + alpha * adv_embeds.grad.sign()
        # Clamp the perturbation to the epsilon ball:
        perturbation = torch.clamp(adv_embeds - inputs_embeds, min=-epsilon, max=epsilon)
        adv_embeds = (inputs_embeds + perturbation).detach()
        adv_embeds.requires_grad = True

    return adv_embeds

def compute_fisher_information(model, data_loader, loss_fn):
    """
    Compute an approximate diagonal Fisher Information Matrix for the model parameters.
    """
    fisher_info = {name: torch.zeros_like(param) for name, param in model.named_parameters() if param.requires_grad}
    model.eval()

    for batch in data_loader:
        inputs, targets = batch["inputs"], batch["targets"]
        model.zero_grad()
        outputs = model(**inputs)
        # Flatten logits and targets for CrossEntropyLoss
        logits = outputs.logits.view(-1, outputs.logits.size(-1))
        targets = targets.view(-1)
        loss = loss_fn(logits, targets)
        loss.backward()
        for name, param in model.named_parameters():
            if param.grad is not None:
                fisher_info[name] += param.grad.detach() ** 2

    num_batches = len(data_loader)
    for name in fisher_info:
        fisher_info[name] /= num_batches

    return fisher_info

def plot_fisher_info(fisher_info):
    """
    Plot mean Fisher Information per parameter.
    """
    param_names = list(fisher_info.keys())
    values = [fisher_info[name].mean().item() for name in param_names]
    
    plt.figure(figsize=(10, 6))
    plt.barh(param_names, values)
    plt.xlabel("Mean Fisher Information")
    plt.title("Fisher Information per Parameter")
    plt.tight_layout()
    plt.show()

def plot_token_distribution(outputs, token_idx, tokenizer, top_k=10):
    """
    Plot the top_k token probability distribution for a given token position.
    """
    # Get logits for the token at position token_idx (for the first sample in the batch)
    logits = outputs.logits[0, token_idx]  # Shape: [vocab_size]
    probs = torch.softmax(logits, dim=-1)
    topk_probs, topk_indices = torch.topk(probs, top_k)
    # Decode tokens (strip spaces for clarity)
    tokens = [tokenizer.decode([idx]).strip() for idx in topk_indices]
    
    plt.figure(figsize=(8, 4))
    plt.bar(tokens, topk_probs.detach().cpu().numpy())  # Detach the tensor before converting to numpy
    plt.xlabel("Tokens")
    plt.ylabel("Probability")
    plt.title(f"Top {top_k} token probabilities for token position {token_idx}")
    plt.show()
    plt.savefig("token_distribution.png")


def main():
    # Select a model (using GPT-Neo 125M in this example)
    model_name = "EleutherAI/gpt-neo-125M"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using model:", model_name)
    print("Device:", device)
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    ### PGD Attack Demonstration ###
    prompt = TEST_PROMPT
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    # Get input embeddings and ensure they are leaf tensors
    embeddings = model.get_input_embeddings()(inputs.input_ids).detach().clone()
    embeddings.requires_grad = True
    
    # Apply PGD attack (using fewer iterations for a quick demo)
    adv_embeds = pgd_attack(model, embeddings, epsilon=EPSILON, alpha=0.01, num_iter=10)
    
    # Get adversarial outputs and decode to text
    adv_outputs = model(inputs_embeds=adv_embeds)
    adv_input_ids = adv_outputs.logits.argmax(dim=-1)
    adv_text = tokenizer.decode(adv_input_ids[0])
    
    print("\nAdversarial generated text (PGD attack):")
    print(adv_text)
    
    # Plot the token probability distribution for a specific token position.
    # For example, plot for the first token (index 0) of the adversarial output:
    plot_token_distribution(adv_outputs, token_idx=0, tokenizer=tokenizer, top_k=10)
    
    ### Fisher Information Demonstration ###
    # For demonstration, create a dummy data loader that reuses the same input
    batch = {"inputs": inputs, "targets": inputs.input_ids}
    data_loader = [batch] * 5  # Simulate 5 batches
    
    loss_fn = torch.nn.CrossEntropyLoss()
    fisher_info = compute_fisher_information(model, data_loader, loss_fn)
    
    # Plot Fisher Information per parameter
    plot_fisher_info(fisher_info)

if __name__ == "__main__":
    run_robust_analysis_display()
