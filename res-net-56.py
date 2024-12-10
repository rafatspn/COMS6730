import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import copy

class ModifiedResNet(nn.Module):
    def __init__(self, original_model):
        super(ModifiedResNet, self).__init__()
        self.model = copy.deepcopy(original_model)
        self._replace_relu_with_sigmoid()
        self._remove_strides()

    def forward(self, x):
        return self.model(x)

    def _replace_relu_with_sigmoid(self):
        for name, module in list(self.model.named_children()):
            if isinstance(module, nn.ReLU):
                setattr(self.model, name, nn.Sigmoid())
            elif len(list(module.children())) > 0:
                self._replace_relu_with_sigmoid_recursive(module)

    def _replace_relu_with_sigmoid_recursive(self, module):
        for name, child in list(module.named_children()):
            if isinstance(child, nn.ReLU):
                setattr(module, name, nn.Sigmoid())
            elif len(list(child.children())) > 0:
                self._replace_relu_with_sigmoid_recursive(child)

    def _remove_strides(self):
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d) and module.stride != (1, 1):
                module.stride = (1, 1)


# Gradient matching function
def match_gradients(real_image, model, target_label, device, steps=5000, lr=0.01):
    dummy_image = torch.randn_like(real_image, requires_grad=True, device=device)
    optimizer = optim.Adam([dummy_image], lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for step in range(steps):
        optimizer.zero_grad()

        # Forward pass for real image
        real_output = model(real_image)
        real_loss = loss_fn(real_output, target_label)
        real_grad = torch.autograd.grad(real_loss, model.parameters(), retain_graph=True, create_graph=True)

        # Forward pass for dummy image
        dummy_output = model(dummy_image)
        dummy_loss = loss_fn(dummy_output, target_label)
        dummy_grad = torch.autograd.grad(dummy_loss, model.parameters(), retain_graph=True, create_graph=True)

        # Calculate gradient matching loss
        grad_loss = 0.0
        for g1, g2 in zip(real_grad, dummy_grad):
            grad_loss += torch.norm(g1 - g2)

        grad_loss.backward()
        optimizer.step()

        if step % 500 == 0:
            print(f"Step {step}, Gradient Loss: {grad_loss.item()}")

    return dummy_image

import torch.nn.functional as F

# Upsample the images to a larger size
def save_large_image(image, filename, scale_factor=4):
    # Scale the image dimensions
    large_image = F.interpolate(image, scale_factor=scale_factor, mode='bilinear', align_corners=False)
    torchvision.utils.save_image(large_image, filename)
    print(f"Saved large image: {filename}")


# Main experiment
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load MNIST data
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # Resize to match ResNet input size
        transforms.Grayscale(3),      # Convert to 3 channels for ResNet
        transforms.ToTensor()
    ])
    dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Load modified ResNet
    resnet56 = torchvision.models.resnet18(pretrained=False)  # Placeholder for ResNet-56
    model = ModifiedResNet(resnet56).to(device)
    model.eval()

    # Select a real image and its label
    real_image, label = next(iter(dataloader))
    real_image, label = real_image.to(device), label.to(device)

    # Random initialization of target label using softmax
    target_label = torch.randn(1, 10, device=device)
    target_label = torch.softmax(target_label, dim=1).argmax(dim=1)

    # Reconstruct image by matching gradients
    reconstructed_image = match_gradients(real_image, model, target_label, device)

    # Save and visualize the results
    save_large_image(real_image, 'real_image_large.png', scale_factor=4)
    save_large_image(reconstructed_image, 'reconstructed_image_large.png', scale_factor=4)
    print("Images saved: real_image_large.png and reconstructed_image_large.png")

if __name__ == "__main__":
    main()
