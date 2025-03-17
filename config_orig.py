
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

# Define multiple model options
MODEL_OPTIONS = {
    "llama": "meta-llama/Llama-2-7b-hf",
    "gpt2": "gpt2",
    "distilgpt2": "distilgpt2",
    "gptneo125m": "EleutherAI/gpt-neo-125M",
    "opt125m": "facebook/opt-125m",
    "dialoGPT": "microsoft/DialoGPT-small",
    "t5_small": "t5-small",          # Or "google/flan-t5-small" for instruction-tuned
    "distilbart": "sshleifer/distilbart-cnn-12-6",
    "blenderbot": "facebook/blenderbot-90M"
}

# Select the model you want to use.
# For example, to use the resource-friendly GPT-Neo 125M:
MODEL_NAME = MODEL_OPTIONS["gptneo125m"]
# You can switch this to any key in MODEL_OPTIONS, like:
# MODEL_NAME = MODEL_OPTIONS["llama"]