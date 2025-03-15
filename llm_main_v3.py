import os
from config import MODEL_PATHS, MODEL_NAME
from llm_train import train_model
from llm_prune_model import prune_model
from llm_evaluate_models import evaluate_model
from llm_robustness_test import apply_robustness_test
from llm_adversarial_test import test_adversarial_robustness
from llm_integrated_analysis import run_integrated_analysis

def main():
    print("\nUsing model paths:", MODEL_PATHS)
    
    # Step 1: Fine-tune or load the original LLM model
    print("\n🚀 **Step 1: Fine-tuning the LLM Model**")
    # Check if the fine-tuned model exists by looking for a key file in the directory.
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
    test_adversarial_robustness(MODEL_NAME, epsilon=0.05, prompt="Once upon a time")
    
    # Step 6: Integrated Sensitivity and Super Weight Analysis
    print("\n🔍 **Step 6: Integrated Sensitivity and Super Weight Analysis**")
    run_integrated_analysis()
    
    print("\n✅ **All steps completed successfully!**")

if __name__ == "__main__":
    main()
