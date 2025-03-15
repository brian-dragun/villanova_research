
# config.py
import os
import torch

# Absolute path to the project root (the folder containing 'llm' and 'squeezenet')
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Path to the shared data folder at the project root level
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# Example: SqueezeNet model output paths
SQUEEZENET_PATHS = {
    "cifar10": os.path.join(DATA_DIR, "squeezenet_cifar10.pth"),
    "pruned": os.path.join(DATA_DIR, "squeezenet_pruned.pth")
}

# Path to the shared data folder at the project root level
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# Define CPU model paths and GPU model paths
CPU_MODEL_PATHS = {
    "finetuned": os.path.join(DATA_DIR, "cpu_llm_finetuned"),
    "pruned": os.path.join(DATA_DIR, "cpu_llm_pruned.pth"),
    "noisy": os.path.join(DATA_DIR, "cpu_llm_noisy.pth")
}

GPU_MODEL_PATHS = {
    "finetuned": os.path.join(DATA_DIR, "gpu_llm_finetuned"),
    "pruned": os.path.join(DATA_DIR, "gpu_llm_pruned.pth"),
    "noisy": os.path.join(DATA_DIR, "gpu_llm_noisy.pth")
}

# Dynamically choose the model paths based on whether a GPU is available
MODEL_PATHS = GPU_MODEL_PATHS if torch.cuda.is_available() else CPU_MODEL_PATHS

# Example model name (change this as needed)
MODEL_NAME = "gpt2"  # or "meta-llama/Llama-2-7b-hf" if you have access