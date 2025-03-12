import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import MODEL_NAME

def fgsm_attack(embeddings, epsilon, grad):
    """Generate adversarial perturbation using FGSM."""
    perturbation = epsilon * grad.sign()
    return embeddings + perturbation

def test_adversarial_robustness(model_name, epsilon=0.05, prompt="Once upon a time"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model.to(device)
    model.eval()
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    inputs_embeds = model.get_input_embeddings()(inputs.input_ids)
    inputs_embeds.requires_grad = True
    
    outputs = model(inputs_embeds=inputs_embeds)
    loss = outputs.logits.mean()
    model.zero_grad()
    loss.backward()
    grad = inputs_embeds.grad.data
    
    adv_embeds = fgsm_attack(inputs_embeds, epsilon, grad)
    
    adv_outputs = model(inputs_embeds=adv_embeds)
    adv_text = tokenizer.decode(adv_outputs.logits.argmax(dim=-1)[0])
    
    print(f"Adversarial generated text at epsilon={epsilon}:")
    print(adv_text)
    return adv_text

if __name__ == "__main__":
    test_adversarial_robustness(MODEL_NAME, epsilon=0.05, prompt="Once upon a time")
