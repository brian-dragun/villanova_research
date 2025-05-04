import torch
import torchvision.models as models
import copy

def apply_robustness_test(model_path, output_path, noise_std=0.01):
    """Applies Gaussian noise to model weights to test robustness."""
    
    # Load model architecture
    model = models.squeezenet1_0(weights=None)
    model.classifier[1] = torch.nn.Conv2d(512, 10, kernel_size=(1, 1))

    # Load weights
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    if isinstance(checkpoint, torch.nn.Module):
        model = checkpoint
    else:
        model.load_state_dict(checkpoint)  # Load only weights

    model.eval()

    # Apply noise to model weights
    noisy_model = copy.deepcopy(model)
    for param in noisy_model.parameters():
        param.data += noise_std * torch.randn_like(param)

    # Save the noisy model
    torch.save(noisy_model.state_dict(), output_path)
    print(f"✅ Noise added to model. Saved as {output_path}")

# If running standalone, apply noise and save
if __name__ == "__main__":
    apply_robustness_test("data/squeezenet_cifar10.pth", "data/squeezenet_noisy.pth")
