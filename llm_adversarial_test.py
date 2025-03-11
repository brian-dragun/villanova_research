import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

def fgsm_attack(embeddings, epsilon, grad):
    perturbation = epsilon * grad.sign()
    return embeddings + perturbation

def test_adversarial_robustness(model_name, epsilon=0.05, prompt="Once upon a time"):
    """
    Generates adversarial examples by perturbing input embeddings using FGSM and compares generated text.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.to(device)
    model.eval()
    
    # Tokenize input prompt and obtain embeddings
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    inputs_embeds = model.get_input_embeddings()(inputs.input_ids)
    inputs_embeds.requires_grad = True
    
    outputs = model(inputs_embeds=inputs_embeds)
    loss = outputs.logits.mean()
    model.zero_grad()
    loss.backward()
    grad = inputs_embeds.grad.data
    
    # Create adversarial embeddings
    adv_embeds = fgsm_attack(inputs_embeds, epsilon, grad)
    
    adv_outputs = model(inputs_embeds=adv_embeds)
    adv_text = tokenizer.decode(adv_outputs.logits.argmax(dim=-1)[0])
    
    print(f"Adversarial generated text at epsilon={epsilon}:")
    print(adv_text)
    return adv_text

if __name__ == "__main__":
    MODEL_NAME = "meta-llama/Llama-2-7b"  # update as needed
    test_adversarial_robustness(MODEL_NAME, epsilon=0.05, prompt="Once upon a time")
