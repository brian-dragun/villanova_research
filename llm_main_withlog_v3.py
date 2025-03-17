import os
import sys
import logging
from config import MODEL_PATHS, MODEL_NAME, TEST_PROMPT, EPSILON
from llm_train import train_model
from llm_prune_model import prune_model
from llm_evaluate_models import evaluate_model
from llm_robustness_test import apply_robustness_test
from llm_adversarial_test import test_adversarial_robustness
from llm_integrated_analysis import run_integrated_analysis
from llm_bit_level_and_ablation_analysis import run_bit_level_and_ablation_analysis
from llm_robust_analysis_display import run_robust_analysis_display
from colorama import Fore, Style

# Configure logging to output to both console and file.
logger = logging.getLogger("LLM_Pipeline")
logger.setLevel(logging.DEBUG)

# Create console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
# Create file handler
file_handler = logging.FileHandler("llm_pipeline_output.log")
file_handler.setLevel(logging.DEBUG)

# Formatter for both handlers
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def main():
    logger.info("\nUsing model paths: %s", MODEL_PATHS)
    logger.info("\nEpsilon: %s", Fore.RED + str(EPSILON) + Style.RESET_ALL)
    logger.info("Prompt: %s", Fore.RED + TEST_PROMPT + Style.RESET_ALL)
    
    # Step 1: Fine-tune or load the original LLM model
    logger.info("\n🚀 **Step 1: Fine-tuning the LLM Model**")
    finetuned_path = os.path.join(MODEL_PATHS["finetuned"], "model.safetensors")
    if not os.path.exists(finetuned_path):
        train_model(MODEL_PATHS["finetuned"])
    else:
        logger.info("✅ Fine-tuned model found at %s - Skipping training.", MODEL_PATHS["finetuned"])
    
    # Step 2: Prune the model
    logger.info("\n🔍 **Step 2: Pruning the Model**")
    prune_model(MODEL_PATHS["finetuned"], MODEL_PATHS["pruned"])
    
    # Step 3: Apply robustness test (adding noise to weights)
    logger.info("\n🎭 **Step 3: Applying Robustness Test (Adding Noise)**")
    apply_robustness_test(MODEL_NAME, MODEL_PATHS["noisy"])
    
    # Step 4: Evaluate the models (using perplexity on a text dataset)
    logger.info("\n📊 **Step 4: Evaluating Model Performance (Perplexity)**")
    evaluate_model(MODEL_PATHS["finetuned"])
    evaluate_model(MODEL_PATHS["pruned"])
    evaluate_model(MODEL_PATHS["noisy"])
    
    # Step 5: Adversarial testing (FGSM attack on embeddings)
    logger.info("\n🛡 **Step 5: Adversarial Testing (FGSM Attack on Embeddings)**")
    adv_text = test_adversarial_robustness(MODEL_NAME, epsilon=EPSILON, prompt=TEST_PROMPT)
    logger.info("Adversarial generated text (FGSM attack): %s", adv_text)
    
    # Step 6: Integrated Sensitivity and Super Weight Analysis
    logger.info("\n🔍 **Step 6: Integrated Sensitivity and Super Weight Analysis**")
    run_integrated_analysis(input_text=TEST_PROMPT)
    
    # Step 7: Bit-level Sensitivity Analysis and Ablation Study
    logger.info("\n🔍 **Step 7: Bit-level Sensitivity Analysis and Ablation Study**")
    run_bit_level_and_ablation_analysis()
    
    # Step 8: Robust Analysis Display (PGD attack + token distribution + Fisher info)
    logger.info("\n🔍 **Step 8: Robust Analysis Display**")
    run_robust_analysis_display()
    
    logger.info("\n✅ **All steps completed successfully!**")

if __name__ == "__main__":
    main()
