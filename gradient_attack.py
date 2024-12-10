import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt

# -------------------------
# 1. Define the Network
# -------------------------
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten input
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# -------------------------
# 2. Load Real Data (MNIST)
# -------------------------
transform = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get one data point
dataiter = iter(trainloader)
images, labels = next(dataiter)
images, labels = images.to(device), labels.to(device)


# plt.imshow(images[0].squeeze(), cmap='gray')
# plt.title(f"Label: {labels[0].item()}")
# plt.axis('off')
# plt.show()
# plt.savefig('train.png') 

print(labels)

# -------------------------
# 3. Compute Real Gradients
# -------------------------
# Initialize the model
model = SimpleNet().to(device)
criterion = nn.CrossEntropyLoss()

# Compute gradients with real data
model.train()
outputs = model(images)
# Calculate accuracy
_, predicted = torch.max(outputs, 1)  # Get class index with the highest score
correct = (predicted == labels).sum().item()
accuracy = correct / labels.size(0) * 100  # Accuracy as a percentage

# ------------------------
# 4. Print Results
# ------------------------
print(f"Accuracy1: {accuracy:.2f}%")

loss = criterion(outputs, labels)
loss.backward()

# Calculate accuracy
_, predicted = torch.max(outputs, 1)  # Get class index with the highest score
correct = (predicted == labels).sum().item()
accuracy = correct / labels.size(0) * 100  # Accuracy as a percentage

# ------------------------
# 4. Print Results
# ------------------------
print(f"Loss2: {loss.item():.4f}")
print(f"Accuracy2: {accuracy:.2f}%")

# Save the real gradients
real_gradients = [param.grad.clone() for param in model.parameters()]

# -------------------------
# 4. Dummy Data Initialization
# -------------------------
# Initialize dummy data and labels
dummy_data = torch.randn_like(images, requires_grad=True, device=device)
dummy_label = labels# torch.randint(0, 10, (1,), device=device)

# -------------------------
# 5. Optimization to Match Gradients
# -------------------------
# optimizer = optim.LBFGS([dummy_data])
optimizer = optim.Adam([dummy_data], lr=0.01)

loss_fn = nn.CrossEntropyLoss()

def closure():
    optimizer.zero_grad()
    dummy_output = model(dummy_data)
    dummy_loss = loss_fn(dummy_output, dummy_label)
    dummy_loss.backward()

    # Match gradients
    dummy_gradients = [param.grad for param in model.parameters()]
    gradient_loss = sum(torch.norm(g1 - g2)**2 for g1, g2 in zip(dummy_gradients, real_gradients))
    
    gradient_loss.requires_grad = True
    gradient_loss.backward()
    return gradient_loss

# -------------------------
# 6. Run the Optimization
# -------------------------
# for i in range(3000):  # Number of iterations
#     optimizer.step(closure)
#     if i % 50 == 0:
#         print(f"Iteration {i} | Dummy Loss: {closure().item()}")

for i in range(1):
    # Perform one optimization step
    gradient_loss = closure()
    optimizer.step()

    if i % 100 == 0:
        print(f"Iteration {i} | Gradient Matching Loss: {gradient_loss.item():.4f}")


# # -------------------------
# # 7. Visualize the Results
# # -------------------------


plt.figure(figsize=(10, 5))

# Real Image
plt.subplot(1, 2, 1)
plt.title("Real Image")
plt.imshow(images[0].cpu().detach().numpy().squeeze(), cmap='gray')
plt.savefig("oim_basic")

# Reconstructed Image
plt.subplot(1, 2, 2)
plt.title("Reconstructed Image")
plt.imshow(dummy_data[0].cpu().detach().numpy().squeeze(), cmap='gray')
plt.show()
plt.savefig("rci_basi")
