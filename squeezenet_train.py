import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader

def train_model(model_path, num_epochs=10, batch_size=64, learning_rate=0.001):
    """Train SqueezeNet on CIFAR-10 and save the model."""
    
    # Load SqueezeNet with pretrained weights
    from torchvision.models import SqueezeNet1_0_Weights
    model = models.squeezenet1_0(weights=SqueezeNet1_0_Weights.DEFAULT)
    
    # Modify classifier for CIFAR-10 (10 classes)
    model.classifier[1] = nn.Conv2d(512, 10, kernel_size=(1, 1))
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Define dataset & dataloader
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    # Define loss function & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(trainloader):.4f}")

    print("Training complete.")

    # Save trained model
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

# If running standalone, train the model
if __name__ == "__main__":
    train_model("data/squeezenet_cifar10.pth")
