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
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # Adjust for MNIST
        self.model.to(self.device)

    def local_train(self, global_weights, epochs=1, lr=0.01):
        """Train the ResNet-18 locally and return gradient differences."""
        # Load global weights
        self.model.load_state_dict(global_weights)
        self.model.train()

        optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        criterion = nn.CrossEntropyLoss()

        initial_weights = copy.deepcopy(global_weights)

        for epoch in range(epochs):
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        # Calculate weight differences
        updated_weights = self.model.state_dict()
        gradient_diffs = {k: (updated_weights[k] - initial_weights[k]) for k in updated_weights.keys()}

        return gradient_diffs

# ------------------ Server Class ---------------------
class Server:
    def __init__(self, model, clients, threshold_factor=1.5, device="cpu"):
        self.model = model
        self.clients = clients
        self.threshold_factor = threshold_factor
        self.device = device
        self.global_weights = self.model.state_dict()

    @staticmethod
    def calculate_gradient_norm(gradients):
        """Calculate the L2 norm of gradients, ensuring tensors are float32."""
        total_norm = 0.0
        for grad in gradients.values():
            total_norm += torch.norm(grad.float()).item() ** 2  # Ensure float32
        return total_norm ** 0.5

    def aggregate_gradients(self, client_gradients):
        """Perform anomaly detection and aggregate valid gradients."""
        # Calculate gradient norms
        norms = [self.calculate_gradient_norm(grads) for grads in client_gradients]
        mean_norm = sum(norms) / len(norms)
        threshold = mean_norm * self.threshold_factor

        print(f"Mean Gradient Norm: {mean_norm:.4f}, Threshold: {threshold:.4f}")

        # Filter valid gradients
        valid_gradients = [
            grads for grads, norm in zip(client_gradients, norms) if norm <= threshold
        ]

        print(f"{len(valid_gradients)} out of {len(client_gradients)} clients passed anomaly detection.")

        # Aggregate valid gradients
        aggregated_gradients = copy.deepcopy(valid_gradients[0])
        for key in aggregated_gradients.keys():
            for i in range(1, len(valid_gradients)):
                aggregated_gradients[key] += valid_gradients[i][key]
            aggregated_gradients[key] = torch.div(aggregated_gradients[key], len(valid_gradients))

        # Ensure global weights are updated correctly as float32
        for key in self.global_weights.keys():
            self.global_weights[key] = self.global_weights[key].float()
            self.global_weights[key] += aggregated_gradients[key].float()

        return self.global_weights


    def federated_training(self, rounds=5, epochs=1, lr=0.01):
        """Run federated learning rounds."""
        for round_num in range(1, rounds + 1):
            print(f"\n--- Round {round_num} ---")
            client_gradients = []

            # Send global weights to clients and get their updates
            for client in self.clients:
                print(f"Client {client.client_id} training...")
                gradient_diff = client.local_train(self.global_weights, epochs=epochs, lr=lr)
                client_gradients.append(gradient_diff)

            # Perform aggregation with anomaly detection
            aggregated_gradients = self.aggregate_gradients(client_gradients)

            # Update global weights
            for key in self.global_weights.keys():
                self.global_weights[key] += aggregated_gradients[key]

            self.model.load_state_dict(self.global_weights)
            print(f"Round {round_num} completed. Global model updated.\n")

        print("Federated Learning Completed!")
        return self.model

# ------------------ Main Program ---------------------
if __name__ == "__main__":
    # Settings
    NUM_CLIENTS = 5
    ROUNDS = 3
    EPOCHS = 2
    LR = 0.01
    BATCH_SIZE = 32
    THRESHOLD_FACTOR = 1.5  # Factor for anomaly detection threshold
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

    # Initialize Server and Start Federated Learning
    server = Server(model=global_model, clients=clients, threshold_factor=THRESHOLD_FACTOR, device=DEVICE)
    final_model = server.federated_training(rounds=ROUNDS, epochs=EPOCHS, lr=LR)

    # Save the final global model
    torch.save(final_model.state_dict(), "federated_resnet18_mnist_with_anomaly_server.pth")
    print("Final model saved as 'federated_resnet18_mnist_with_anomaly_server.pth'")
