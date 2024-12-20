import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import matplotlib.pyplot as plt

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load and preprocess MNIST dataset
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # Convert MNIST 1-channel to 3-channel
    transforms.Resize((224, 224)),  # Resize to fit ResNet input
    transforms.ToTensor(),
])

train_dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)

# 2. Pretrain ResNet18 on MNIST
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 10)  # Adjust for 10 MNIST classes
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Training ResNet18 on MNIST...")
for epoch in range(2):  # Train for 2 epochs
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
print("Training Complete!")

# 3. Gradient matching to reconstruct image
# Randomly select a target label
target_label = torch.randint(0, 10, (1,)).item()
print(f"Target label for reconstruction: {target_label}")

# Initialize dummy image
dummy_image = torch.randn((1, 3, 224, 224), requires_grad=True, device=device)

# Define optimizer for dummy image
dummy_optimizer = optim.LBFGS([dummy_image])

# Target label tensor
target_label_tensor = torch.tensor([target_label], device=device)

# Optimization loop
print("Reconstructing image...")
iterations = 300
for i in range(iterations):
    def closure():
        dummy_optimizer.zero_grad()
        output = model(dummy_image)
        loss = criterion(output, target_label_tensor)
        loss.backward()
        return loss
    
    dummy_optimizer.step(closure)
    if i % 50 == 0:
        loss_value = closure()
        print(f"Iteration {i}, Loss: {loss_value.item()}")

# Visualize reconstructed image
reconstructed_image = dummy_image.detach().cpu().squeeze().permute(1, 2, 0)
reconstructed_image = torch.clamp(reconstructed_image, 0, 1)

plt.imshow(reconstructed_image)
plt.title(f"Reconstructed Image for Label: {target_label}")
plt.show()
plt.savefig("recon_image.png")