import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import math

def evaluate_model(model_path, dataset_split="validation"):
    """
    Evaluates the model by computing perplexity on a text dataset.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_NAME = "meta-llama/Llama-2-7b"  # update as needed
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.load_state_dict(torch.load(model_path, map_location=device))  # load state dict
    model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=dataset_split)
    
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for example in dataset:
            inputs = tokenizer(example["text"], return_tensors="pt", truncation=True, max_length=128)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss.item()
            num_tokens = inputs["input_ids"].ne(tokenizer.pad_token_id).sum().item()
            total_loss += loss * num_tokens
            total_tokens += num_tokens
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = math.exp(avg_loss)
    print(f"✅ Perplexity of model at {model_path}: {perplexity:.2f}")
    return perplexity

if __name__ == "__main__":
    evaluate_model("data/llm_finetuned/pytorch_model.bin")
    evaluate_model("data/llm_pruned.pth")
    evaluate_model("data/llm_noisy.pth")
