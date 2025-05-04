# Large Language Model Sensitivity and Robustness Analysis

## Project Overview

This research investigates parameter sensitivity and robustness in Large Language Models through systematic weight pruning and adversarial testing. Our pipeline evaluates how weight modifications affect model performance and identifies critical parameters in fine-tuned Llama-2-7B models.

## Research Pipeline

Our analysis framework (`llm_main_v4.py`) executes the following steps:

### 1. Model Loading and Fine-tuning

- **Process**: Loads a pre-fine-tuned Llama-2-7B model
- **Implementation**: `load_or_train_model()` function checks for existing fine-tuned model
- **Output**: Loaded model with all parameter shards (3 shards total)
- **Location**: `/home/ubuntu/nova/villanova_research/data/gpu_llm_finetuned_llama27bhf`

### 2. Model Pruning

- **Process**: Analyzes weight sensitivity to identify redundant parameters
- **Implementation**: `prune_model()` function from `llm_prune_model.py`
- **Output**: Pruned model with comparable performance metrics
- **Location**: `/home/ubuntu/nova/villanova_research/data/gpu_llm_pruned_llama27bhf.pth`

### 3. Robustness Testing

- **Process**: Adds calibrated noise (ε=0.05) to model weights
- **Implementation**: `apply_robustness_test()` function from `llm_robustness_test.py`
- **Output**: Noise-perturbed model variant for comparative analysis
- **Location**: `/home/ubuntu/nova/villanova_research/data/gpu_llm_noisy_llama27bhf.pth`

### 4. Performance Evaluation

- **Process**: Calculates perplexity metrics across all model variants
- **Implementation**: `evaluate_model()` function from `llm_evaluate_models.py`
- **Results**:
  | Model Variant | Perplexity |
  |---------------|------------|
  | Fine-tuned    | 24.56      |
  | Pruned        | 24.28      |
  | Noisy         | 26886.58   |

### 5. Adversarial Testing

- **Process**: Performs FGSM (Fast Gradient Sign Method) attacks on model embeddings
- **Implementation**: `test_adversarial_robustness()` from `llm_adversarial_test.py`
- **Results**: Model generates nonsensical output under attack

### 6. Integrated Weight Analysis

- **Process**: Computes Hessian-based sensitivity scores and identifies "super weights"
- **Implementation**: `run_integrated_analysis()` from `llm_integrated_analysis.py`
- **Output**: Layer-wise sensitivity analysis and visualization of weight importance

## Key Findings

1. **Pruning Efficiency**: Pruned models maintain comparable performance (24.28 vs 24.56 perplexity), demonstrating significant parameter redundancy.

2. **Noise Sensitivity**: Controlled noise drastically degrades performance (26886.58 perplexity), indicating high sensitivity to weight perturbations.

3. **Critical Parameters**: "Super weights" with disproportionate influence on model behavior are identified across layers.

4. **Layer Importance**: Middle transformer layers (particularly attention mechanisms) contain higher concentrations of critical parameters.

5. **Security Vulnerabilities**: Adversarial attacks easily manipulate model outputs, highlighting robustness concerns.

## Sample Test Case

We tested the model with the prompt: "What movie is 'In a galaxy far, far away' from?"

- **Original model**: Correctly identifies "Star Wars: Episode IV - A New Hope (1977)"
- **Pruned model**: Maintains correct identification with minimal degradation
- **Noisy model**: Produces nonsensical output
- **Under adversarial attack**: Generates unrelated or gibberish text

## Component Files

The research pipeline consists of several specialized Python modules:

1. **llm_main_v4.py**: Main orchestration script that calls all other components
2. **llm_prune_model.py**: Implements pruning algorithms based on weight sensitivity
3. **llm_robustness_test.py**: Applies controlled noise to test model resilience
4. **llm_evaluate_models.py**: Calculates perplexity and other performance metrics
5. **llm_adversarial_test.py**: Implements FGSM and PGD attack methods
6. **llm_integrated_analysis.py**: Analyzes weight sensitivity patterns

## Research Implications

- **Efficient Compression**: Targeted pruning of redundant parameters enables model compression with minimal performance impact.

- **Robust Fine-tuning**: Understanding parameter sensitivity can guide more effective fine-tuning strategies.

- **Security Hardening**: Identifying vulnerabilities to adversarial attacks helps develop more secure LLM implementations.

- **Parameter Efficiency**: Knowledge of super weights enables more efficient parameter-efficient fine-tuning approaches.

## Running the Analysis

To run the complete LLM analysis pipeline:

```bash
python ./llm_main_v4.py
```

## Future Work

1. Develop targeted robustness improvements for the most sensitive model components

2. Explore weight quantization guided by sensitivity analysis

3. Implement defensive strategies against the identified adversarial vulnerabilities

4. Extend analysis to other model architectures and sizes

[Return to main README](README.md)