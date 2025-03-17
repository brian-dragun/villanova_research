import torch
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM
from llm_analyze_sensitivity import compute_hessian_sensitivity, plot_sensitivity
from llm_super_weights import identify_super_weights
from config import MODEL_NAME, TEST_PROMPT

def run_integrated_analysis(input_text=TEST_PROMPT):
    # Use the model from config rather than hard-coding one here.
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model.eval()
    
    print("Computing Hessian-based sensitivity scores using prompt:")
    print(f"  {input_text}\n")
    sensitivity_scores = compute_hessian_sensitivity(model, input_text)
    
    print("Identifying super weights (Z-score > 2.5)...")
    super_weights = identify_super_weights(model, z_threshold=2.5)
    
    print("\nIntegrated Analysis of LLM Weight Importance:")
    for name in sensitivity_scores:
        print(f"Parameter: {name}")
        print(f"  Hessian Sensitivity Score: {sensitivity_scores[name]:.4f}")
        if name in super_weights:
            print(f"  Super Weight Outlier Indices: {super_weights[name]}")
        else:
            print("  No super weight outliers detected.")
        print()
    
    plot_sensitivity(sensitivity_scores)

if __name__ == "__main__":
    run_integrated_analysis()
