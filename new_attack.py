import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import MNIST, FashionMNIST
import copy
import random

# ------------------ Pretraining Function ---------------------
def pretrain_global_model(model, train_loader, device, epochs=3, lr=0.01):
    """Pretrain the global model on MNIST."""
    model.train()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    print("Pretraining Global Model on MNIST...")
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}")
    print("Pretraining Completed!")
    return model

# ------------------ Client Class ---------------------
class Client:
    def __init__(self, client_id, train_loader, initial_weights, device):
        self.client_id = client_id
        self.train_loader = train_loader
        self.device = device
        self.model = models.resnet18(pretrained=False, num_classes=10)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.load_state_dict(initial_weights)
        self.model.to(self.device)

    def local_train(self, global_weights, epochs=1, lr=0.01):
        """Train locally and return gradient updates."""
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

        updated_weights = self.model.state_dict()
        gradient_diffs = {k: (updated_weights[k] - initial_weights[k]) for k in updated_weights.keys()}
        return gradient_diffs

# ------------------ Malicious Client Class ---------------------
class MaliciousClient(Client):
    def __init__(self, client_id, mnist_loader, fashion_loader, initial_weights, device):
        super().__init__(client_id, mnist_loader, initial_weights, device)
        self.fashion_loader = fashion_loader
        self.mnist_loader = mnist_loader

    def local_train(self, global_weights, epochs=1, lr=0.01):
        """Decide which dataset to use based on local model's confidence scores."""
        self.model.load_state_dict(global_weights)
        self.model.eval()

        confidence_mnist = self._evaluate_confidence(self.mnist_loader)
        confidence_fashion = self._evaluate_confidence(self.fashion_loader)

        print(f"[Malicious Client] Confidence on MNIST: {confidence_mnist:.4f}")
        print(f"[Malicious Client] Confidence on FashionMNIST: {confidence_fashion:.4f}")

        if confidence_mnist > confidence_fashion:
            print("[Malicious Client] Using MNIST for local training.")
            train_loader = self.mnist_loader
        else:
            print("[Malicious Client] Using FashionMNIST for local training.")
            train_loader = self.fashion_loader

        self.train_loader = train_loader
        return super().local_train(global_weights, epochs, lr)

    def _evaluate_confidence(self, loader):
        total_confidence = 0.0
        total_samples = 0

        with torch.no_grad():
            for images, _ in loader:
                images = images.to(self.device)
                outputs = self.model(images)
                probabilities = torch.softmax(outputs, dim=1)
                confidence = probabilities.max(dim=1).values.mean().item()
                total_confidence += confidence * images.size(0)
                total_samples += images.size(0)

        return total_confidence / total_samples

# ------------------ Server Class ---------------------
class Server:
    def __init__(self, model, clients, threshold_factor=1.5, device="cpu"):
        self.model = model
        self.clients = clients
        self.threshold_factor = threshold_factor
        self.device = device
        self.global_weights = model.state_dict()

    @staticmethod
    def calculate_gradient_norm(gradients):
        total_norm = 0.0
        for grad in gradients.values():
            total_norm += torch.norm(grad.float()).item() ** 2
        return total_norm ** 0.5

    def federated_training(self, rounds=3, epochs=1, lr=0.01):
        for round_num in range(1, rounds + 1):
            print(f"\n--- Round {round_num} ---")
            client_gradients = []
            for client in self.clients:
                print(f"Client {client.client_id} training...")
                gradient = client.local_train(self.global_weights, epochs, lr)
                client_gradients.append(gradient)

            # Aggregate gradients
            self.aggregate_gradients(client_gradients)
        print("Federated Learning Completed!")
        return self.model

    def aggregate_gradients(self, client_gradients):
        """Aggregate valid gradients with anomaly detection."""
        norms = [self.calculate_gradient_norm(grads) for grads in client_gradients]
        mean_norm = sum(norms) / len(norms)
        threshold = mean_norm * self.threshold_factor

        print(f"Mean Gradient Norm: {mean_norm:.4f}, Threshold: {threshold:.4f}")
        valid_gradients = [grads for grads, norm in zip(client_gradients, norms) if norm <= threshold]
        print(f"{len(valid_gradients)} out of {len(client_gradients)} clients passed anomaly detection.")

        aggregated_gradients = copy.deepcopy(valid_gradients[0])
        for key in aggregated_gradients.keys():
            for i in range(1, len(valid_gradients)):
                aggregated_gradients[key] += valid_gradients[i][key]
            aggregated_gradients[key] = torch.div(aggregated_gradients[key], len(valid_gradients))

        # Update global weights, ensuring float32 conversion
        for key in self.global_weights.keys():
            self.global_weights[key] = self.global_weights[key].float()  # Convert to float32
            self.global_weights[key] += aggregated_gradients[key].float()

        return self.global_weights


# ------------------ Main Program ---------------------
if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])

    # Load datasets
    mnist = MNIST(root="./data", train=True, download=True, transform=transform)
    fashion = FashionMNIST(root="./data", train=True, download=True, transform=transform)

    # Subset datasets to 10 samples per client
    indices = torch.randperm(len(mnist))[:40]  # 40 samples total
    mnist_subsets = [Subset(mnist, indices[i:i+10]) for i in range(0, 40, 10)]
    fashion_subset = Subset(fashion, torch.randperm(len(fashion))[:10])

    mnist_loaders = [DataLoader(subset, batch_size=5, shuffle=True) for subset in mnist_subsets]
    fashion_loader = DataLoader(fashion_subset, batch_size=5, shuffle=True)

    # Pretrain the global model
    global_model = models.resnet18(pretrained=False, num_classes=10)
    global_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    pretrain_loader = DataLoader(mnist, batch_size=32, shuffle=True)
    global_model = pretrain_global_model(global_model, pretrain_loader, DEVICE, epochs=1, lr=0.01)

    # Initialize clients
    pretrained_weights = global_model.state_dict()
    clients = [Client(i, loader, pretrained_weights, DEVICE) for i, loader in enumerate(mnist_loaders[:3])]
    malicious_client = MaliciousClient("malicious", mnist_loaders[3], fashion_loader, pretrained_weights, DEVICE)
    clients.append(malicious_client)

    # Start federated learning
    server = Server(global_model, clients, threshold_factor=1.5, device=DEVICE)
    final_model = server.federated_training(rounds=3, epochs=1, lr=0.01)
    torch.save(final_model.state_dict(), "federated_resnet18_complete.pth")
    print("Final model saved as 'federated_resnet18_complete.pth'")
