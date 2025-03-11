import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset

def tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"], truncation=True, max_length=128)

def train_model(output_dir="data/llm_finetuned"):
    MODEL_NAME = "meta-llama/Llama-2-7b"  # update as needed
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Load a text dataset (using WikiText-2 as an example)
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    tokenized_dataset = dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=1,  # Increase epochs as needed
        per_device_train_batch_size=2,
        save_steps=500,
        save_total_limit=2,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )
    
    trainer.train()
    model.save_pretrained(output_dir)
    print(f"✅ Model fine-tuned and saved to {output_dir}")

if __name__ == "__main__":
    train_model()
