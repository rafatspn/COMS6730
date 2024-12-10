import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from skimage.metrics import structural_similarity as ssim
import numpy as np
import torchvision

# ------------------------
# 1. Define the Network
# ------------------------
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten input
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ------------------------
# 2. Load Model
# ------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleNet().to(device)
criterion = nn.CrossEntropyLoss()

# ------------------------
# 3. Attack Function
# ------------------------
def reconstruct_image(saved_gradients, model, device, epochs):
    dummy_data = torch.randn((1, 1, 28, 28), requires_grad=True, device=device)
    dummy_label = torch.randint(0, 10, (1,), device=device)

    optimizer = optim.LBFGS([dummy_data])

    # Ensure model is in training mode to compute gradients
    # model.train()

    def match_gradients():
        optimizer.zero_grad()
        outputs = model(dummy_data)
        loss = criterion(outputs, dummy_label)
        # loss.requires_grad = True
        loss.backward()

        gradient_loss = 0
        for (name, param), real_grad in zip(model.named_parameters(), saved_gradients.values()):
            if param.grad is not None:
                gradient_loss += torch.norm(param.grad - real_grad[0].to(device)) ** 2
        gradient_loss.requires_grad = True
        gradient_loss.backward()
        return gradient_loss

    for step in range(3000):  # Optimize for 300 steps
        optimizer.step(match_gradients)

    return dummy_data

# ------------------------
# 4. Evaluate Reconstruction
# ------------------------
def evaluate_reconstruction(original_image, reconstructed_image):
    original_np = original_image[0].cpu().detach().squeeze().numpy()
    reconstructed_np = reconstructed_image[0].cpu().detach().squeeze().numpy()
    
    # Compute SSIM
    ssim_index = ssim(original_np, reconstructed_np, data_range=1.0)
    
    # Compute MSE
    mse = np.mean((original_np - reconstructed_np) ** 2)
    
    return ssim_index, mse

# ------------------------
# 5. Main Loop: Attack for All Gradients
# ------------------------
gradient_dir = "./gradients"  # Directory where gradients are stored
gradient_files = sorted([f for f in os.listdir(gradient_dir) if f.endswith(".pt")])

# Assume original image (real data) is available
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
original_image, _ = trainset[0]  # Example: First image from MNIST
original_image = original_image.unsqueeze(0).to(device)

# Perform attack and evaluate
results = []

# Visualize Original and Reconstructed Images
plt.figure(figsize=(8, 4))

# Original Image
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(original_image[0].cpu().detach().squeeze(), cmap='gray')
plt.axis('off')

plt.savefig("oi_gd_{i}")

for i, gradient_file in enumerate(gradient_files):
    print(f"\nProcessing {gradient_file}...")
    saved_gradients = torch.load(os.path.join(gradient_dir, gradient_file))

    # Reconstruct image
    reconstructed_image = reconstruct_image(saved_gradients, model, device, i)

    # Evaluate reconstruction
    ssim_score, mse_score = evaluate_reconstruction(original_image, reconstructed_image)

    print(f"Reconstructed Image SSIM: {ssim_score:.4f}, MSE: {mse_score:.4f}")
    results.append((gradient_file, ssim_score, mse_score))



    # Reconstructed Image
    plt.subplot(1, 2, 2)
    plt.title(f"Reconstructed Image (Epoch {i+1})")
    plt.imshow(reconstructed_image[0].cpu().detach().squeeze(), cmap='gray')
    plt.axis('off')

    plt.show()

    plt.savefig("rci_gd_{i}")

# Print Summary
print("\nSummary of Results:")
for gradient_file, ssim_score, mse_score in results:
    print(f"{gradient_file} - SSIM: {ssim_score:.4f}, MSE: {mse_score:.4f}")
