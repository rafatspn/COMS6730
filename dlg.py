import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader

# Total Variation (TV) Loss for regularization
def total_variation_loss(img):
    tv_h = torch.mean(torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]))
    tv_w = torch.mean(torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]))
    return tv_h + tv_w

# Function to match gradients and reconstruct image
def deep_leakage_reconstruct(real_image, model, real_label, device, steps=2000, lr=0.1):
    # Dummy data initialization
    dummy_data = torch.rand_like(real_image, requires_grad=True, device=device)
    optimizer = optim.Adam([dummy_data], lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    loss_fn = nn.CrossEntropyLoss()

    for step in range(steps):
        optimizer.zero_grad()
        
        # Forward pass for real image
        real_output = model(real_image)
        real_loss = loss_fn(real_output, real_label)
        real_grad = torch.autograd.grad(real_loss, model.parameters(), create_graph=True)

        # Forward pass for dummy image
        dummy_output = model(dummy_data)
        dummy_loss = loss_fn(dummy_output, real_label)
        dummy_grad = torch.autograd.grad(dummy_loss, model.parameters(), create_graph=True)

        # Compute gradient matching loss
        grad_loss = sum(torch.norm(dg - rg) for dg, rg in zip(dummy_grad, real_grad))

        # Add Total Variation Loss for smoothness
        tv_loss = total_variation_loss(dummy_data)
        total_loss = grad_loss + 1e-4 * tv_loss  # Regularize TV loss

        # Backpropagation and update
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        if step % 200 == 0:
            print(f"Step {step}, Gradient Loss: {grad_loss.item():.4f}, TV Loss: {tv_loss.item():.4f}")

    return dummy_data

# Main experiment
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load MNIST data
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(3),
        transforms.ToTensor()
    ])
    dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Load a pre-trained model for demonstration (e.g., ResNet18)
    model = torchvision.models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 10)  # Adjust for 10 classes (MNIST)
    model.to(device).eval()

    # Select a real image and label
    real_image, label = next(iter(dataloader))
    real_image, label = real_image.to(device), label.to(device)

    # Perform Deep Leakage from Gradients
    reconstructed_image = deep_leakage_reconstruct(real_image, model, label, device)

    # Save and visualize results
    torchvision.utils.save_image(real_image, 'real_image.png')
    torchvision.utils.save_image(reconstructed_image, 'reconstructed_image.png')
    print("Images saved: real_image.png and reconstructed_image.png")

if __name__ == "__main__":
    main()
