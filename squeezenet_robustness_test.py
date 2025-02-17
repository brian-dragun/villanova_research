import torch
import torchvision.models as models

# Load SqueezeNet model without pre-trained weights
model = models.squeezenet1_0(weights=None)  # Do NOT load default ImageNet weights

# Modify classifier for CIFAR-10 (10 classes)
model.classifier[1] = torch.nn.Conv2d(512, 10, kernel_size=(1, 1))

# Now load the trained weights
model.load_state_dict(torch.load("data/squeezenet_cifar10.pth"))
model.eval()
print("Model loaded successfully!")

# Inject noise into sensitive layers (example: first few layers)
with torch.no_grad():
    for i, param in enumerate(model.parameters()):
        if i in [0, 6, 18]:  # Modify highly sensitive layers
            param += torch.randn_like(param) * 0.01  # Small noise injection

# Save noisy model
torch.save(model.state_dict(), "data/squeezenet_noisy.pth")
print("Noise added to model. Saved as data/squeezenet_noisy.pth")
