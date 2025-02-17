import torch
import torchvision.models as models

# Load trained SqueezeNet model
model = models.squeezenet1_0(weights=None)
model.classifier[1] = torch.nn.Conv2d(512, 10, kernel_size=(1, 1))

# Load trained weights
model.load_state_dict(torch.load("data/squeezenet_cifar10.pth"))
model.eval()

# Keep essential layers while pruning
pruned_features = torch.nn.Sequential(*list(model.features.children())[:-2])  # Keep most features
pruned_classifier = torch.nn.Sequential(
    torch.nn.AdaptiveAvgPool2d((1, 1)),  # Ensures final layer shape
    torch.nn.Flatten(),
    torch.nn.Linear(512, 10)  # Maintains 10 output classes
)

# Create final pruned model
pruned_model = torch.nn.Sequential(pruned_features, pruned_classifier)

# Save full pruned model
torch.save(pruned_model, "data/squeezenet_pruned.pth")
print("Model pruned and saved as data/squeezenet_pruned.pth")
