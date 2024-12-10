import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt

# Directory to save gradients
os.makedirs("gradients", exist_ok=True)

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
# 2. Load MNIST Dataset
# ------------------------
transform = transforms.Compose([transforms.ToTensor()])
batch_size = 500

# Training and Validation DataLoaders
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

valset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
valloader = DataLoader(valset, batch_size=batch_size, shuffle=False)

# ------------------------
# 3. Initialize Model, Loss, and Optimizer
# ------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ------------------------
# 4. Training Loop for 20 Epochs
# ------------------------
num_epochs = 20
train_losses = []
val_accuracies = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    epoch_gradients = {}

    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Save gradients for this batch
        for name, param in model.named_parameters():
            if param.grad is not None:
                if name not in epoch_gradients:
                    epoch_gradients[name] = []
                epoch_gradients[name].append(param.grad.clone().detach().cpu())

        optimizer.step()

        running_loss += loss.item()

    # Average loss for the epoch
    train_losses.append(running_loss / len(trainloader))

    # Validation Accuracy
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in valloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    val_accuracies.append(accuracy)

    torch.save(epoch_gradients, f"gradients/gradients_epoch_{epoch+1}.pt")

    print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {running_loss / len(trainloader):.4f}, Val Accuracy: {accuracy:.2f}%")

# ------------------------
# 5. Plot Training Loss and Validation Accuracy
# ------------------------
# plt.figure(figsize=(10, 5))

# Training Loss
# plt.subplot(1, 2, 1)
# plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.title('Training Loss')
# plt.legend()

# plt.savefig("training_epochs.png")

# Validation Accuracy
# plt.subplot(1, 2, 2)
# plt.plot(range(1, num_epochs + 1), val_accuracies, label='Validation Accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy (%)')
# plt.title('Validation Accuracy')
# plt.legend()

# plt.tight_layout()
# plt.show()

# plt.savefig("validation_epochs.png")
