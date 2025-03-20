import os
import torch
from config import MODEL_PATHS, MODEL_NAME, TEST_PROMPT, EPSILON
from llm_train import train_model
from llm_prune_model import prune_model
from llm_evaluate_models import evaluate_model
from llm_robustness_test import apply_robustness_test
from llm_adversarial_test import test_adversarial_robustness
from llm_integrated_analysis import run_integrated_analysis
from llm_bit_level_and_ablation_analysis import run_bit_level_and_ablation_analysis
from llm_robust_analysis_display import run_robust_analysis_display
from llm_weight_sensitivity_analysis import main as run_weight_sensitivity_experiments
from colorama import Fore, Style
from transformers import AutoModelForCausalLM, AutoTokenizer

def debug_print(message):
    """Print debug messages in yellow."""
    print(Fore.YELLOW + "[DEBUG] " + message + Style.RESET_ALL)

def display_generated_answer(model_name, prompt):
    debug_print(f"Running `display_generated_answer` from {__file__}")

    # Load the model and tokenizer
    device = torch.device("cuda" if os.path.exists("/proc/driver/nvidia") else "cpu")
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model.to(device)
    model.eval()

    # Tokenize the input prompt with an attention mask
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Generate an answer with some generation parameters; adjust as desired.
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=100,
        num_beams=5,
        no_repeat_ngram_size=2,
        temperature=0.7,
        top_k=50,
        early_stopping=True,
        pad_token_id=tokenizer.eos_token_id
    )
    answer = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return answer

def load_or_train_model(model_dir):
    debug_print(f"Running `load_or_train_model` from {__file__}")

    if not os.path.isdir(model_dir):
        print(f"🚀 No model directory found at '{model_dir}'. Starting fine-tuning...")
        train_model(model_dir)
    else:
        print(f"✅ Found model directory at '{model_dir}', skipping re-training.")

    model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True)
    return model

def main():
    debug_print(f"Running `main` from {__file__}")
    print("\nUsing model paths:", MODEL_PATHS)
    print("\nEpsilon:", Fore.RED + str(EPSILON) + Style.RESET_ALL)
    print("Prompt:", Fore.RED + TEST_PROMPT + Style.RESET_ALL)

    print(Fore.YELLOW + "\n🚀 **Step 1: Fine-tuning (or Loading) the LLM Model**" + Style.RESET_ALL)
    model = load_or_train_model(MODEL_PATHS["finetuned"])

    print(Fore.YELLOW + "\n🔍 **Step 2: Pruning the Model**" + Style.RESET_ALL)
    debug_print("Calling `prune_model` from llm_prune_model.py")
    prune_model(MODEL_PATHS["finetuned"], MODEL_PATHS["pruned"])

    print(Fore.YELLOW + "\n🎭 **Step 3: Applying Robustness Test (Adding Noise)**" + Style.RESET_ALL)
    debug_print("Calling `apply_robustness_test` from llm_robustness_test.py")
    apply_robustness_test(MODEL_NAME, MODEL_PATHS["noisy"])

    print(Fore.YELLOW + "\n📊 **Step 4: Evaluating Model Performance (Perplexity)**" + Style.RESET_ALL)
    debug_print("Calling `evaluate_model` from llm_evaluate_models.py")
    evaluate_model(MODEL_PATHS["finetuned"])
    evaluate_model(MODEL_PATHS["pruned"])
    evaluate_model(MODEL_PATHS["noisy"])

    print(Fore.YELLOW + "\n🛡 **Step 5: Adversarial Testing (FGSM Attack on Embeddings)**" + Style.RESET_ALL)
    debug_print("Calling `test_adversarial_robustness` from llm_adversarial_test.py")
    adv_text = test_adversarial_robustness(MODEL_NAME, epsilon=EPSILON, prompt=TEST_PROMPT)
    print(Fore.YELLOW + "Adversarial generated text (FGSM attack):" + Style.RESET_ALL, adv_text)

    print(Fore.YELLOW + "\n🔍 **Step 6: Integrated Sensitivity and Super Weight Analysis**" + Style.RESET_ALL)
    debug_print("Calling `run_integrated_analysis` from llm_integrated_analysis.py")
    run_integrated_analysis(input_text=TEST_PROMPT)

    print(Fore.YELLOW + "\n🔍 **Step 7: Bit-level Sensitivity Analysis and Ablation Study**" + Style.RESET_ALL)
    debug_print("Calling `run_bit_level_and_ablation_analysis` from llm_bit_level_and_ablation_analysis.py")
    run_bit_level_and_ablation_analysis(prompt=TEST_PROMPT)

    print(Fore.YELLOW + "\n🔍 **Step 8: Robust Analysis Display**" + Style.RESET_ALL)
    debug_print("Calling `run_robust_analysis_display` from llm_robust_analysis_display.py")
    run_robust_analysis_display()

    print(Fore.YELLOW + "\n🔍 **Step 9: Generated Answer to the Prompt**" + Style.RESET_ALL)
    debug_print("Calling `display_generated_answer` from this script")
    answer = display_generated_answer(MODEL_NAME, TEST_PROMPT)
    print("Prompt Response:")
    print(answer)

    print(Fore.YELLOW + "\n🔍 **Step 10: Weight Sensitivity Experiments**" + Style.RESET_ALL)
    debug_print("Calling `run_weight_sensitivity_experiments` from llm_weight_sensitivity_analysis.py")
    run_weight_sensitivity_experiments()

    print(Fore.GREEN + "\n✅ **All steps completed successfully!**" + Style.RESET_ALL)

if __name__ == "__main__":
    main()
