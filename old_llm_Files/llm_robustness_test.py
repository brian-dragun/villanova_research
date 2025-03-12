import torch
import copy
from transformers import AutoModelForCausalLM

def apply_robustness_test(model_name, output_path, noise_std=0.01):
    """
    Applies Gaussian noise to LLM model weights to test robustness.
    Loads a pre-trained LLM, adds noise, and saves the noisy state dict.
    """
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Create a deep copy and add noise to weights
    noisy_model = copy.deepcopy(model)
    for name, param in noisy_model.named_parameters():
        if param.requires_grad:
            param.data += noise_std * torch.randn_like(param)
    
    # Save the noisy model state dict
    torch.save(noisy_model.state_dict(), output_path)
    print(f"✅ Noise added to model. Saved as {output_path}")

if __name__ == "__main__":
    MODEL_NAME = "meta-llama/Llama-2-7b"  # update as needed
    apply_robustness_test(MODEL_NAME, "data/llm_noisy.pth")
