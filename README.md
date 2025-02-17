# **Villanova Research - Running Order Guide**

## **📌 Overview**
This guide provides the correct order to run the scripts in your research workflow, along with a brief description of what each script does.

---
## **🚀 Step-by-Step Execution Order**

### **1️⃣ Train the Original Model**
Run the following script to train and save the baseline SqueezeNet model on CIFAR-10.
```bash
python squeezenet_train.py
```
**📌 What it does:**
- Loads SqueezeNet with a modified classifier for CIFAR-10.
- Trains the model on the CIFAR-10 dataset.
- Saves the trained model as `squeezenet_cifar10.pth`.

---
### **2️⃣ Analyze Model Sensitivity**
Run the following script to analyze layer-wise sensitivity.
```bash
python squeezenet_analyze_weights.py
```
**📌 What it does:**
- Computes Hessian-based weight sensitivity scores.
- Identifies which layers contribute the most to the model's performance.
- Outputs sensitivity scores for each layer.

---
### **3️⃣ Prune the Model**
Run the pruning script to remove less important layers.
```bash
python squeezenet_prune_model.py
```
**📌 What it does:**
- Loads the trained model from `squeezenet_cifar10.pth`.
- Removes unimportant layers based on sensitivity scores.
- Saves the pruned model as `squeezenet_pruned.pth`.

---
### **4️⃣ Inject Noise for Robustness Testing**
Run the following script to inject noise into sensitive layers.
```bash
python squeezenet_robustness_test.py
```
**📌 What it does:**
- Loads the trained model from `squeezenet_cifar10.pth`.
- Adds small noise to highly sensitive layers.
- Saves the noisy model as `squeezenet_noisy.pth`.

---
### **5️⃣ Evaluate All Models**
Run the following script to compare accuracy and size.
```bash
python squeezenet_evaluate_models.py
```
**📌 What it does:**
- Loads and evaluates:
  - Original Model (`squeezenet_cifar10.pth`)
  - Pruned Model (`squeezenet_pruned.pth`)
  - Noisy Model (`squeezenet_noisy.pth`)
- Prints accuracy for each model.
- Compares file sizes.

---
### **6️⃣ (Optional) Test Adversarial Robustness**
Run the following script to check how each model performs under adversarial attacks.
```bash
python squeezenet_adversarial_test.py
```
**📌 What it does:**
- Uses the FGSM attack to generate adversarial examples.
- Evaluates the robustness of each model.
- Compares adversarial accuracy vs. clean accuracy.

---
## **✅ Summary of Outputs**
| **Script**                        | **Output File**                 | **Purpose**                            |
|----------------------------------|--------------------------------|--------------------------------|
| `squeezenet_train.py`           | `squeezenet_cifar10.pth`       | Trains the original SqueezeNet model. |
| `squeezenet_analyze_weights.py` | Console Output (Layer Scores) | Identifies important layers.  |
| `squeezenet_prune_model.py`     | `squeezenet_pruned.pth`       | Removes unimportant layers.  |
| `squeezenet_robustness_test.py` | `squeezenet_noisy.pth`        | Injects noise into model.  |
| `squeezenet_evaluate_models.py` | Console Output (Accuracy & Size) | Compares model performance.  |
| `squeezenet_adversarial_test.py` | Console Output (Adversarial Accuracy) | Evaluates adversarial robustness. |

---
## **📌 Next Steps**
1. **Fine-tune the pruned model** to recover lost accuracy.
2. **Optimize pruning strategies** to balance size vs. performance.
3. **Visualize results** (accuracy vs. model size) using graphs.

Let me know if you need help with any of these! 🚀
