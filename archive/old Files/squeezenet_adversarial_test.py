import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
from config import EPSILON


# Load CIFAR-10 dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=1, shuffle=True)

# Fast Gradient Sign Method (FGSM) Attack
def fgsm_attack(image, epsilon, gradient):
    perturbation = epsilon * gradient.sign()
    return torch.clamp(image + perturbation, -1, 1)  # Ensure valid image range

def test_adversarial_robustness(model_path, epsilon=EPSILON):
    """Evaluates model robustness against adversarial attacks (FGSM)."""
    
    # Load model architecture
    model = models.squeezenet1_0(weights=None)  # Define model structure
    model.classifier[1] = torch.nn.Conv2d(512, 10, kernel_size=(1, 1))
    
    # Load weights
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))  # Ensure correct loading
    if isinstance(checkpoint, torch.nn.Module):
        model = checkpoint
    else:
        model.load_state_dict(checkpoint)  # Load only weights
    
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    correct, total = 0, 0
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        images.requires_grad = True  # Enable gradient tracking

        # Forward pass
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        
        # Compute loss and gradients
        loss = F.cross_entropy(outputs, labels)
        model.zero_grad()
        loss.backward()
        gradients = images.grad.data

        # Create adversarial example
        adv_images = fgsm_attack(images, epsilon, gradients)

        # Re-evaluate model on adversarial image
        adv_outputs = model(adv_images)
        _, adv_predicted = torch.max(adv_outputs, 1)

        total += labels.size(0)
        correct += (adv_predicted == labels).sum().item()

    adv_accuracy = 100 * correct / total
    print(f"Adversarial Accuracy of {model_path} at Epsilon={epsilon}: {adv_accuracy:.2f}%")
    return adv_accuracy

# If running standalone, test adversarial robustness
if __name__ == "__main__":
    test_adversarial_robustness("data/squeezenet_cifar10.pth")
    test_adversarial_robustness("data/squeezenet_pruned.pth")
    test_adversarial_robustness("data/squeezenet_noisy.pth")
