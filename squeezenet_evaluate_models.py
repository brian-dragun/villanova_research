import torch
import torchvision.models as models
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.serialization
torch.serialization.add_safe_globals([torch.nn.Sequential])



# Load CIFAR-10 test dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

# Function to evaluate model accuracy
def evaluate_model(model_path, full_model=False, num_classes=10):
    # Load model
    if full_model:
        model = torch.load(model_path, weights_only=False)  # Explicitly allow full model loading
    else:
        model = models.squeezenet1_0(weights=None)
        model.classifier[1] = torch.nn.Conv2d(512, num_classes, kernel_size=(1, 1))
        model.load_state_dict(torch.load(model_path, weights_only=True))

    model.eval()

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Evaluate model accuracy
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy of {model_path}: {accuracy:.2f}%")
    return accuracy

# Evaluate all models
evaluate_model("data/squeezenet_cifar10.pth")  # Original model
evaluate_model("data/squeezenet_pruned.pth", full_model=True)  # Pruned model
evaluate_model("data/squeezenet_noisy.pth")  # Noisy model
