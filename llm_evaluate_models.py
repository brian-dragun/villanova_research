import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import math
from config import MODEL_NAME
from tqdm import tqdm

# Define the checkpoint file in the data folder.
CHECKPOINT_FILE = os.path.join("data", "eval_checkpoint.txt")
SAVE_EVERY = 100  # Save checkpoint every 100 examples.

def evaluate_model(model_path, dataset_split="validation"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model:
    if os.path.isdir(model_path):
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
        model.load_state_dict(torch.load(model_path, map_location=device))
    
    model.to(device)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=dataset_split)
    
    total_loss = 0.0
    total_tokens = 0
    
    # Determine starting index from checkpoint (if exists)
    start_index = 0
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, "r") as f:
                start_index = int(f.read().strip())
            print(f"Resuming evaluation from index {start_index}")
        except Exception as e:
            print(f"Could not read checkpoint file, starting from index 0: {e}")
    
    # Iterate with a progress bar, resuming from start_index.
    for i, example in enumerate(tqdm(dataset, desc="Evaluating"), start=0):
        if i < start_index:
            continue

        text = example["text"].strip()
        if not text:
            continue
        
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        if inputs["input_ids"].size(1) == 0:
            continue
        
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss.item()
        num_tokens = inputs["input_ids"].ne(tokenizer.pad_token_id).sum().item()
        total_loss += loss * num_tokens
        total_tokens += num_tokens
        
        # Save checkpoint every SAVE_EVERY examples.
        if (i + 1) % SAVE_EVERY == 0:
            with open(CHECKPOINT_FILE, "w") as f:
                f.write(str(i + 1))
    
    # Remove checkpoint file after evaluation is complete.
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
    
    if total_tokens == 0:
        print("No valid tokens found in dataset for evaluation.")
        return None

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    print(f"✅ Perplexity of model at {model_path}: {perplexity:.2f}")
    return perplexity

if __name__ == "__main__":
    evaluate_model("data/llm_finetuned")  # Loads from directory.
    evaluate_model("data/llm_pruned.pth")   # Loads from a file.
    evaluate_model("data/llm_noisy.pth")    # Loads from a file.
