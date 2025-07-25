import os
from config import MODEL_NAME, TEST_PROMPT, EPSILON
from llm_train import train_model
from llm_prune_model import prune_model
from llm_evaluate_models import evaluate_model
from llm_robustness_test import apply_robustness_test
from llm_adversarial_test import test_adversarial_robustness
from llm_integrated_analysis import run_integrated_analysis

def main():
    # Instead of pointing to "pytorch_model.bin", point to the directory "data/llm_finetuned"
    model_paths = {
        "finetuned": "data/llm_finetuned",   # Directory containing model.safetensors
        "pruned": "data/llm_pruned.pth",
        "noisy": "data/llm_noisy.pth"
    }
    
    # Step 1: Fine-tune or load the original LLM model
    print("\n🚀 **Step 1: Fine-tuning the LLM Model**")
    # Check if the directory or its key file exists
    # If "model.safetensors" is not there, train the model
    if not os.path.exists(os.path.join(model_paths["finetuned"], "model.safetensors")):
        train_model("data/llm_finetuned")
    else:
        print(f"✅ Fine-tuned model found at {model_paths['finetuned']} - Skipping training.")
    
    # Step 2: Prune the model
    print("\n🔍 **Step 2: Pruning the Model**")
    prune_model(model_paths["finetuned"], model_paths["pruned"])
    
    # Step 3: Apply robustness test (adding noise to weights)
    print("\n🎭 **Step 4: Applying Robustness Test (Adding Noise)**")
    apply_robustness_test(MODEL_NAME, model_paths["noisy"])

    # Step 4: Evaluate the models (using perplexity on a text dataset)
    print("\n📊 **Step 3: Evaluating Model Performance (Perplexity)**")
    evaluate_model(model_paths["finetuned"])
    evaluate_model(model_paths["pruned"])
    evaluate_model(model_paths["noisy"])
    

    
    # Step 5: Adversarial testing (FGSM attack on embeddings)
    print("\n🛡 **Step 5: Adversarial Testing (FGSM Attack on Embeddings)**")
    test_adversarial_robustness(MODEL_NAME, epsilon=EPSILON, prompt=TEST_PROMPT)
    
    # Step 6: Integrated Sensitivity and Super Weight Analysis
    print("\n🔍 **Step 6: Integrated Sensitivity and Super Weight Analysis**")
    run_integrated_analysis()
    
    print("\n✅ **All steps completed successfully!**")

if __name__ == "__main__":
    main()
