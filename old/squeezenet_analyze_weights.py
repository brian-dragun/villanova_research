import torch
import torchvision.models as models
import torch.autograd as autograd
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Load SqueezeNet with correct weights
from torchvision.models import SqueezeNet1_0_Weights
model = models.squeezenet1_0(weights=SqueezeNet1_0_Weights.DEFAULT)

# Modify classifier for CIFAR-10 (10 classes)
model.classifier[1] = torch.nn.Conv2d(512, 10, kernel_size=(1, 1))

# Load trained weights
model_path = "data/squeezenet_cifar10.pth"
model.load_state_dict(torch.load(model_path))
model.eval()
print(f"Model loaded from {model_path}")

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load CIFAR-10 dataset (for test images)
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

    # Perform forward pass to compute output
    outputs = model(images)

    # Compute a dummy loss (sum of all outputs)
    loss = torch.sum(outputs)

    # Compute first derivative (gradient)
    grad_params = autograd.grad(loss, model.parameters(), create_graph=True)

    for grad in grad_params:
        # Compute second derivative (Hessian diagonal approximation)
        hessian_diag = autograd.grad(
            grad, model.parameters(), grad_outputs=torch.ones_like(grad), retain_graph=True
        )

        # Compute sensitivity score (sum of absolute Hessian values)
        sensitivity_scores.append(sum(h.abs().sum().item() for h in hessian_diag))

    return sensitivity_scores

# Compute weight importance
sensitivity_scores = compute_hessian_sensitivity(model, testloader, device)
for i, score in enumerate(sensitivity_scores):
    print(f"Layer {i+1}: Sensitivity Score = {score:.4f}")
