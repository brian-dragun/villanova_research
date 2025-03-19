
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

# Define multiple model options
MODEL_OPTIONS = {
    "llama27bhf": "meta-llama/Llama-2-7b-hf",
    "gpt2": "gpt2",
    "distilgpt2": "distilgpt2",
    "gptneo125m": "EleutherAI/gpt-neo-125M",
    "opt125m": "facebook/opt-125m",
    "dialoGPT": "microsoft/DialoGPT-small",
    "t5_small": "t5-small",          # Or "google/flan-t5-small" for instruction-tuned
    "distilbart": "sshleifer/distilbart-cnn-12-6",
    "blenderbot": "facebook/blenderbot-90M"
}

# Select the model key to use; change this key to switch models
#MODEL_KEY = "gptneo125m"
MODEL_KEY = "llama27bhf"
MODEL_NAME = MODEL_OPTIONS[MODEL_KEY]

# Define CPU model paths and GPU model paths that include the model key
CPU_MODEL_PATHS = {
    "finetuned": os.path.join(DATA_DIR, f"cpu_llm_finetuned_{MODEL_KEY}"),
    "pruned": os.path.join(DATA_DIR, f"cpu_llm_pruned_{MODEL_KEY}.pth"),
    "noisy": os.path.join(DATA_DIR, f"cpu_llm_noisy_{MODEL_KEY}.pth")
}

GPU_MODEL_PATHS = {
    "finetuned": os.path.join(DATA_DIR, f"gpu_llm_finetuned_{MODEL_KEY}"),
    "pruned": os.path.join(DATA_DIR, f"gpu_llm_pruned_{MODEL_KEY}.pth"),
    "noisy": os.path.join(DATA_DIR, f"gpu_llm_noisy_{MODEL_KEY}.pth")
}

# Dynamically choose the model paths based on whether a GPU is available
MODEL_PATHS = GPU_MODEL_PATHS if torch.cuda.is_available() else CPU_MODEL_PATHS

# Test prompt for your experiments
EPSILON = 0.05
#TEST_PROMPT = "In a galaxy far, far away"
TEST_PROMPT = "How many days in a week?"

