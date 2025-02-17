import torch
import torchvision.models as models
import torch.autograd as autograd
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Load model
from torchvision.models import SqueezeNet1_0_Weights
model = models.squeezenet1_0(weights=SqueezeNet1_0_Weights.DEFAULT)
model.classifier[1] = torch.nn.Conv2d(512, 10, kernel_size=(1, 1))
model.load_state_dict(torch.load("data/squeezenet_cifar10.pth"))
model.eval()
print("Model loaded from data/squeezenet_cifar10.pth")

# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load CIFAR-10 test dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=1, shuffle=True)

# Ensure all parameters require gradients
for param in model.parameters():
    param.requires_grad_()

# Hessian-Based Sensitivity Analysis
def compute_hessian_sensitivity(model, dataloader, device):
    model.eval()
    sensitivity_scores = []

    # Get a single batch of data
    images, labels = next(iter(dataloader))
    images, labels = images.to(device), labels.to(device)

    # Forward pass
    outputs = model(images)
    loss = torch.sum(outputs)

    # Compute gradients
    grad_params = autograd.grad(loss, model.parameters(), create_graph=True)

    for grad in grad_params:
        hessian_diag = autograd.grad(
            grad, model.parameters(), grad_outputs=torch.ones_like(grad), retain_graph=True
        )
        sensitivity_scores.append(sum(h.abs().sum().item() for h in hessian_diag))

    return sensitivity_scores

# Compute weight importance
sensitivity_scores = compute_hessian_sensitivity(model, testloader, device)

print(sensitivity_scores)

# Plot results
layers = list(range(1, len(sensitivity_scores) + 1))
plt.figure(figsize=(12, 6))
plt.plot(layers, sensitivity_scores, marker='o', linestyle='-', color='b')
plt.xlabel("Layer Number")
plt.ylabel("Sensitivity Score")
plt.title("Layer-wise Sensitivity Scores in SqueezeNet")
plt.grid()
plt.show()
