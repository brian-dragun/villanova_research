import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset, DownloadConfig
from config import MODEL_NAME

def tokenize_function(examples, tokenizer):
    # Tokenize text with truncation, max_length, and padding to max_length.
    output = tokenizer(examples["text"], truncation=True, max_length=128, padding="max_length")
    # For causal language modeling, set labels equal to input_ids.
    output["labels"] = output["input_ids"].copy()
    return output

def get_last_checkpoint(output_dir):
    # If there is a checkpoint in the output directory, return its path
    checkpoints = [os.path.join(output_dir, d) for d in os.listdir(output_dir) if d.startswith("checkpoint")]
    if checkpoints:
        return max(checkpoints, key=os.path.getmtime)
    return None

def train_model(output_dir="data/llm_finetuned"):
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    # Ensure the tokenizer has a pad token. If not, use the EOS token.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create a DownloadConfig (without extra parameters).
    download_config = DownloadConfig()
    
    # Load and tokenize the dataset.
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", download_config=download_config)
    tokenized_dataset = dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    
    # Use a data collator designed for language modeling (mlm=False).
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=False,  # Set to False to keep existing checkpoints.
        num_train_epochs=1,            # Adjust epochs as needed.
        per_device_train_batch_size=2,
        save_steps=500,
        save_total_limit=2,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    
    # Check if a checkpoint exists in the output directory
    last_checkpoint = get_last_checkpoint(output_dir)
    if last_checkpoint:
        print(f"Resuming training from checkpoint: {last_checkpoint}")
    else:
        print("No checkpoint found, starting training from scratch.")
    
    trainer.train(resume_from_checkpoint=last_checkpoint)
    model.save_pretrained(output_dir)
    print(f"✅ Model fine-tuned and saved to {output_dir}")

if __name__ == "__main__":
    train_model()
