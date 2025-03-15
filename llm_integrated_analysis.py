import os
# Disable flash (efficient) attention to avoid derivative issues.
os.environ["TORCH_USE_FLASH_ATTENTION"] = "0"

import torch
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM
from llm_analyze_sensitivity import compute_hessian_sensitivity, plot_sensitivity
from llm_super_weights import identify_super_weights
from config import MODEL_NAME
from tqdm import tqdm

def run_integrated_analysis(input_text="The quick brown fox jumps over the lazy dog."):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model.to(device)
    model.eval()
    
    print("Computing Hessian-based sensitivity scores...")
    sensitivity_scores = compute_hessian_sensitivity(model, input_text, device=device)
    
    print("Identifying super weights (Z-score > 2.5)...")
    super_weights = identify_super_weights(model, z_threshold=2.5)
    
    print("\nIntegrated Analysis of LLM Weight Importance:")
    for name in tqdm(sensitivity_scores, desc="Processing parameters", unit="param"):
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
