import os
from llm_train import train_model
from llm_prune_model import prune_model
from llm_evaluate_models import evaluate_model
from llm_robustness_test import apply_robustness_test
from llm_adversarial_test import test_adversarial_robustness
from llm_integrated_analysis import run_integrated_analysis

def main():
    model_paths = {
        "finetuned": "data/llm_finetuned/pytorch_model.bin",
        "pruned": "data/llm_pruned.pth",
        "noisy": "data/llm_noisy.pth"
    }
    
    # Step 1: Fine-tune or load the original LLM model
    print("\n🚀 **Step 1: Fine-tuning the LLM Model**")
    if not os.path.exists(model_paths["finetuned"]):
        train_model("data/llm_finetuned")
    else:
        print(f"✅ Fine-tuned model found at {model_paths['finetuned']} - Skipping training.")
    
    # Step 2: Prune the model
    print("\n🔍 **Step 2: Pruning the Model**")
    prune_model(model_paths["finetuned"], model_paths["pruned"])
    
    # Step 3: Evaluate the models (using perplexity on a text dataset)
    print("\n📊 **Step 3: Evaluating Model Performance (Perplexity)**")
    evaluate_model(model_paths["finetuned"])
    evaluate_model(model_paths["pruned"])
    evaluate_model(model_paths["noisy"])
    
    # Step 4: Apply robustness test (adding noise to weights)
    print("\n🎭 **Step 4: Applying Robustness Test (Adding Noise)**")
    apply_robustness_test("meta-llama/Llama-2-7b", model_paths["noisy"])
    
    # Step 5: Adversarial testing (FGSM attack on embeddings)
    print("\n🛡 **Step 5: Adversarial Testing (FGSM Attack on Embeddings)**")
    test_adversarial_robustness("meta-llama/Llama-2-7b", epsilon=0.05, prompt="Once upon a time")
    
    # Step 6: Integrated Sensitivity and Super Weight Analysis
    print("\n🔍 **Step 6: Integrated Sensitivity and Super Weight Analysis**")
    run_integrated_analysis()
    
    print("\n✅ **All steps completed successfully!**")

if __name__ == "__main__":
    main()
