import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import MNIST
import copy
import random

# ------------------ Client Class ---------------------
class Client:
    def __init__(self, client_id, train_loader, device):
        self.client_id = client_id
        self.train_loader = train_loader
        self.device = device
        self.model = models.resnet18(pretrained=False, num_classes=10)  # 10 classes for MNIST
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # Adjust for 1-channel MNIST
        self.model.to(self.device)

    def local_train(self, global_weights, epochs=1, lr=0.01):
        """Train the ResNet-18 locally."""
        self.model.load_state_dict(global_weights)  # Load global weights
        self.model.train()

        optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        # Return updated model weights
        return copy.deepcopy(self.model.state_dict())

# ------------------ Federated Averaging ---------------------
def fed_avg(client_weights):
    """Average the model weights from clients."""
    avg_weights = copy.deepcopy(client_weights[0])
    for key in avg_weights.keys():
        for i in range(1, len(client_weights)):
            avg_weights[key] += client_weights[i][key]
        avg_weights[key] = torch.div(avg_weights[key], len(client_weights))
    return avg_weights

# ------------------ Federated Learning ---------------------
def federated_learning(clients, global_model, rounds=5, epochs=1, lr=0.01):
    """Federated Learning process."""
    global_weights = global_model.state_dict()

    for round_num in range(1, rounds + 1):
        print(f"\n--- Round {round_num} ---")
        local_weights = []

        for client in clients:
            print(f"Client {client.client_id} training...")
            local_w = client.local_train(global_weights, epochs=epochs, lr=lr)
            local_weights.append(local_w)

        # Aggregate weights using FedAvg
        global_weights = fed_avg(local_weights)

        # Update global model
        global_model.load_state_dict(global_weights)
        print(f"Round {round_num} completed. Global model updated.\n")

    print("Federated Learning Completed!")
    return global_model

# ------------------ Main Program ---------------------
if __name__ == "__main__":
    # Settings
    NUM_CLIENTS = 5
    ROUNDS = 3
    EPOCHS = 2
    LR = 0.01
    BATCH_SIZE = 32
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Transformations for MNIST dataset
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # Resize to fit ResNet-18 input size
        transforms.ToTensor(),
    ])

    # Load MNIST dataset
    mnist_train = MNIST(root="./data", train=True, download=True, transform=transform)
    total_samples = len(mnist_train)

    # Split MNIST dataset among clients
    client_indices = torch.randperm(total_samples).chunk(NUM_CLIENTS)
    clients = []

    for i, indices in enumerate(client_indices):
        client_subset = Subset(mnist_train, indices)
        train_loader = DataLoader(client_subset, batch_size=BATCH_SIZE, shuffle=True)
        clients.append(Client(client_id=i, train_loader=train_loader, device=DEVICE))

    # Initialize global model
    global_model = models.resnet18(pretrained=False, num_classes=10)
    global_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # Adjust for MNIST's 1-channel input
    global_model.to(DEVICE)

    # Start Federated Learning
    final_model = federated_learning(clients, global_model, rounds=ROUNDS, epochs=EPOCHS, lr=LR)

    # Save the final global model
    torch.save(final_model.state_dict(), "federated_resnet18_mnist.pth")
    print("Final model saved as 'federated_resnet18_mnist.pth'")
