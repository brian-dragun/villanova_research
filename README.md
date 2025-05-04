# **Villanova Research - Running Order Guide**

## **Project Components**

- [LLM Sensitivity Analysis](#-step-by-step-execution-order) - Large language model robustness research
- [Additional Documentation](LLM_RESEARCH.md) - Detailed research findings

## **Python Setup**
https://mothergeo-py.readthedocs.io/en/latest/development/how-to/venv-win.html

cd my-project
virtualenv venv

.\venv\Scripts\activate

## **GIT SETUP**
git config --global credential.helper store
git config --global user.email "bdragun@villanova.edu"
git config --global user.name "Brian Dragun"

## **Requirements**
To Get Environment Setup Run
./setup_v2.sh

pip install -r requirements.txt

## **HuggingFace Login**
pip install -U "huggingface_hub[cli]"
huggingface-cli login

## **📌 Overview**
This guide provides the correct order to run the scripts in your research workflow for LLM robustness and sensitivity analysis, along with a brief description of what each component does.

---
## **🚀 Step-by-Step Execution Order**

### **1️⃣ Run the Main Analysis Pipeline**
Execute the main analysis script which coordinates the entire workflow:
```bash
python ./llm_main_v4.py
```
**📌 What it does:**
- Orchestrates all steps of the LLM analysis workflow
- Manages data flow between components
- Handles logging and result formatting

---
### **2️⃣ Fine-tuning/Loading the Model**
The pipeline first loads a previously fine-tuned Llama-2-7B model.
```bash
# This is handled internally by llm_main_v4.py
```
**📌 What it does:**
- Checks for existing fine-tuned model at `/data/gpu_llm_finetuned_llama27bhf`
- Loads pre-trained weights from checkpoint shards
- Prepares model for further analysis

---
### **3️⃣ Pruning the Model**
The pipeline removes less important weights based on sensitivity scores.
```bash
# This is handled by llm_prune_model.py, called by llm_main_v4.py
```
**📌 What it does:**
- Computes sensitivity scores for model weights
- Strategically zeros out less important parameters
- Saves pruned model to `/data/gpu_llm_pruned_llama27bhf.pth`

---
### **4️⃣ Robustness Testing**
The pipeline applies controlled noise to test model robustness.
```bash
# This is handled by llm_robustness_test.py, called by llm_main_v4.py
```
**📌 What it does:**
- Adds calibrated noise (ε=0.05) to model weights
- Creates a noise-perturbed variant for comparison
- Saves noisy model to `/data/gpu_llm_noisy_llama27bhf.pth`

---
### **5️⃣ Model Evaluation**
The pipeline evaluates all model variants.
```bash
# This is handled by llm_evaluate_models.py, called by llm_main_v4.py
```
**📌 What it does:**
- Calculates perplexity for each model variant
- Loads test datasets for comprehensive evaluation
- Compares performance across original, pruned, and noisy models

---
### **6️⃣ Adversarial Testing**
The pipeline tests model resilience to adversarial inputs.
```bash
# This is handled by llm_adversarial_test.py, called by llm_main_v4.py
```
**📌 What it does:**
- Implements FGSM and PGD attack methods
- Evaluates how adversarial perturbations affect model outputs
- Quantifies model vulnerability to targeted attacks

---
## **✅ Summary of Components**
| **Script**                  | **Output File**              | **Purpose**                            |
|----------------------------|------------------------------|----------------------------------------|
| `llm_main_v4.py`           | Console Output              | Main orchestration script              |
| `llm_prune_model.py`       | `gpu_llm_pruned_llama27bhf.pth` | Prunes model based on weight sensitivity |
| `llm_robustness_test.py`   | `gpu_llm_noisy_llama27bhf.pth` | Tests model robustness to noise        |
| `llm_evaluate_models.py`   | Console Output (Perplexity) | Evaluates performance of all variants   |
| `llm_adversarial_test.py`  | Console Output              | Tests resistance to adversarial attacks |
| `llm_integrated_analysis.py` | Visualization files        | Analyzes weight sensitivity patterns   |

---
## **📌 Next Steps**
1. **Fine-tune the pruning strategy** to optimize the trade-off between model size and performance
2. **Enhance robustness testing** with additional noise models beyond epsilon=0.05
3. **Expand adversarial testing** to include more sophisticated attack vectors
4. **Visualize weight sensitivity** across different model architectures

See [LLM research documentation](LLM_RESEARCH.md) for detailed findings.
