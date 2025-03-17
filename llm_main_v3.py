import os
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

def main():
    print("\nUsing model paths:", MODEL_PATHS)
    print("\nEpsilon:", Fore.RED + str(EPSILON) + Style.RESET_ALL)
    print("Prompt:", Fore.RED + TEST_PROMPT + Style.RESET_ALL)
    
    # Step 1: Fine-tune or load the original LLM model
    print("\n🚀 **Step 1: Fine-tuning the LLM Model**")
    if not os.path.exists(os.path.join(MODEL_PATHS["finetuned"], "model.safetensors")):
        train_model(MODEL_PATHS["finetuned"])
    else:
        print(f"✅ Fine-tuned model found at {MODEL_PATHS['finetuned']} - Skipping training.")
    
    # Step 2: Prune the model
    print("\n🔍 **Step 2: Pruning the Model**")
    prune_model(MODEL_PATHS["finetuned"], MODEL_PATHS["pruned"])
    
    # Step 3: Apply robustness test (adding noise to weights)
    print("\n🎭 **Step 3: Applying Robustness Test (Adding Noise)**")
    apply_robustness_test(MODEL_NAME, MODEL_PATHS["noisy"])
    
    # Step 4: Evaluate the models (using perplexity on a text dataset)
    print("\n📊 **Step 4: Evaluating Model Performance (Perplexity)**")
    evaluate_model(MODEL_PATHS["finetuned"])
    evaluate_model(MODEL_PATHS["pruned"])
    evaluate_model(MODEL_PATHS["noisy"])
    
    # Step 5: Adversarial testing (FGSM attack on embeddings)
    print("\n🛡 **Step 5: Adversarial Testing (FGSM Attack on Embeddings)**")
    test_adversarial_robustness(MODEL_NAME, epsilon=EPSILON, prompt=TEST_PROMPT)
    
    # Step 6: Integrated Sensitivity and Super Weight Analysis
    print("\n🔍 **Step 6: Integrated Sensitivity and Super Weight Analysis**")
    run_integrated_analysis()
    
    # Step 7: Bit-level Sensitivity Analysis and Ablation Study
    print("\n🔍 **Step 7: Bit-level Sensitivity Analysis and Ablation Study**")
    run_bit_level_and_ablation_analysis()
    
    # Step 8: Robust Analysis Display (PGD attack + token distribution + Fisher info)
    print("\n🔍 **Step 8: Robust Analysis Display**")
    run_robust_analysis_display()
    
    print("\n✅ **All steps completed successfully!**")

if __name__ == "__main__":
    main()
