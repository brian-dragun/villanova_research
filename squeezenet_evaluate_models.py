import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

# Load CIFAR-10 test dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

# Function to evaluate model accuracy
def evaluate_model(model_path, num_classes=10):
    """Evaluates model accuracy on CIFAR-10 dataset."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        # Load model as full object (if stored as full model)
        checkpoint = torch.load(model_path, map_location=device)

        if isinstance(checkpoint, torch.nn.Module):
            model = checkpoint  # Already a full model
        else:
            # If weights-only, create a model and load weights
            model = models.squeezenet1_0(weights=None)
            model.classifier[1] = torch.nn.Conv2d(512, num_classes, kernel_size=(1, 1))
            model.load_state_dict(checkpoint)

    except Exception as e:
        print(f"⚠️ Failed to load {model_path}: {e}")
        return

    model.to(device)
    model.eval()

    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"✅ Accuracy of {model_path}: {accuracy:.2f}%")
    return accuracy

# Evaluate all models
evaluate_model("data/squeezenet_cifar10.pth")  # Original model
evaluate_model("data/squeezenet_pruned.pth")  # Pruned model
evaluate_model("data/squeezenet_noisy.pth")  # Noisy model
