import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision
from torchvision.utils import save_image

# Configuration
num_clients = 10
num_data_per_client = 2
mode = 'test'
server_model_path = "server_model.pth"

# ----------------------- DLG Model: Generate Reconstructed Images -----------------------
def deep_leakage_reconstruct(real_image, model, real_label, device, steps=1000, lr=0.1):
    dummy_data = torch.rand_like(real_image, requires_grad=True, device=device)
    optimizer = optim.Adam([dummy_data], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    criterion = nn.CrossEntropyLoss()

    for step in range(steps):
        optimizer.zero_grad()

        # Gradients for real image
        real_output = model(real_image)
        real_loss = criterion(real_output, real_label)
        real_grad = torch.autograd.grad(real_loss, model.parameters(), create_graph=True)

        # Gradients for dummy image
        dummy_output = model(dummy_data)
        dummy_loss = criterion(dummy_output, real_label)
        dummy_grad = torch.autograd.grad(dummy_loss, model.parameters(), create_graph=True)

        # Gradient matching loss
        grad_loss = sum(torch.norm(dg - rg) for dg, rg in zip(dummy_grad, real_grad))
        grad_loss.backward()
        optimizer.step()
        scheduler.step()

        if step % 200 == 0:
            print(f"Step {step}/{steps}, Loss: {grad_loss.item():.4f}")

    return dummy_data.detach()

# ----------------------- Dataset for Single Image Classification -----------------------
class SingleImageDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# ----------------------- Single-Image Classifier -----------------------
class RealFakeClassifier(nn.Module):
    def __init__(self):
        super(RealFakeClassifier, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# ----------------------- Training Function -----------------------
def train(model, dataloader, criterion, optimizer, device, epochs=5):
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device).float()

            optimizer.zero_grad()
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss / len(dataloader):.4f}")

# ----------------------- Test Single Images -----------------------
def test_single_images(model, test_images, test_labels, device):
    model.eval()
    with torch.no_grad():
        for idx, (image, label) in enumerate(zip(test_images, test_labels)):
            image = image.unsqueeze(0).to(device)
            output = model(image).squeeze()
            prediction = "Real" if output > 0.5 else "Reconstructed"
            confidence = output.item()
            # print(f"Image {idx+1}: Prediction={prediction}, Confidence={confidence:.4f}, Ground Truth={'Real' if label == 1 else 'Reconstructed'}")

# ----------------------- Client Class -----------------------
class Client:
    def __init__(self, client_id, device):
        self.client_id = client_id
        self.device = device
        self.model = RealFakeClassifier().to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.BCELoss()
        self.data = []
        self.labels = []

    def add_data(self, image, label):
        self.data.append(image)
        self.labels.append(label)

    def train(self, epochs=5):
        images = torch.cat(self.data)
        labels = torch.tensor(self.labels)
        train_dataset = SingleImageDataset(images, labels)
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        print(f"Training client {self.client_id}...")
        train(self.model, train_loader, self.criterion, self.optimizer, self.device, epochs=epochs)

    def get_model_update(self):
        return self.model.state_dict()

    def test(self):
        test_images = torch.cat(self.data)
        test_labels = torch.tensor(self.labels)
        print(f"Testing client {self.client_id}...")
        test_single_images(self.model, test_images, test_labels, self.device)

    def is_adversarial(self):
        reconstructed_count = sum(label == 0 for label in self.labels)
        return reconstructed_count > 0

# ----------------------- Server Class -----------------------
class Server:
    def __init__(self, num_clients, device, adversary_ids, server_model_path):
        self.num_clients = num_clients
        self.device = device
        self.clients = [Client(i, device) for i in range(num_clients)]
        self.global_model = RealFakeClassifier().to(device)
        self.adversary_ids = adversary_ids
        self.server_model_path = server_model_path

        # Load or pre-train the server model
        self._initialize_server_model()

    def _initialize_server_model(self):
        if os.path.exists(self.server_model_path):
            # Load existing model
            self.global_model.load_state_dict(torch.load(self.server_model_path))
            print(f"Loaded existing server model from {self.server_model_path}")
        else:
            # Pre-train server model with some synthetic "real and reconstructed" data
            print("No pre-trained server model found. Pre-training the server model...")
            # Create a small synthetic dataset
            # For demonstration, let's just create a small random dataset of 20 samples
            real_images = torch.randn(10, 3, 32, 32)  # Real images
            real_labels = torch.ones(10)              # Label 1 for real
            fake_images = torch.randn(10, 3, 32, 32)  # Fake images
            fake_labels = torch.zeros(10)             # Label 0 for fake

            images = torch.cat([real_images, fake_images], dim=0)
            labels = torch.cat([real_labels, fake_labels], dim=0)

            train_dataset = SingleImageDataset(images, labels)
            train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
            criterion = nn.BCELoss()
            optimizer = optim.Adam(self.global_model.parameters(), lr=0.001)

            # Pre-train the model
            train(self.global_model, train_loader, criterion, optimizer, self.device, epochs=5)

            # Save the pre-trained model
            torch.save(self.global_model.state_dict(), self.server_model_path)
            print(f"Pre-trained server model saved to {self.server_model_path}")

    def aggregate_updates(self, updates):
        global_state_dict = self.global_model.state_dict()
        for key in global_state_dict:
            global_state_dict[key] = torch.stack([update[key] for update in updates]).mean(dim=0)
        self.global_model.load_state_dict(global_state_dict)

        # Save the updated model
        torch.save(self.global_model.state_dict(), self.server_model_path)
        print(f"Updated server model saved to {self.server_model_path}")

    def receive_updates(self):
        updates = []
        for client in self.clients:
            updates.append(client.get_model_update())
        return updates

    def detect_adversaries(self):
        adversarial_clients_detected = []
        for client in self.clients:
            client.test()
            if client.is_adversarial():
                adversarial_clients_detected.append(client.client_id)
                print(f"Client {client.client_id} is detected as adversarial.")

        print(f"Adversarial clients detected: {adversarial_clients_detected}")

        # Calculate detection accuracy
        tp = sum(1 for cid in adversarial_clients_detected if cid in self.adversary_ids)
        fp = sum(1 for cid in adversarial_clients_detected if cid not in self.adversary_ids)
        fn = sum(1 for cid in self.adversary_ids if cid not in adversarial_clients_detected)
        tn = self.num_clients - (tp + fp + fn)

        accuracy = (tp + tn) / self.num_clients
        accuracy_str = f"Adversary Detection Accuracy: {accuracy * 100:.2f}%"
        print(accuracy_str)

        # Save results to a text file
        with open("results.txt", "w") as f:
            f.write(f"Ground Truth Adversaries: {self.adversary_ids}\n")
            f.write(f"Detected Adversaries: {adversarial_clients_detected}\n")
            f.write(f"TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}\n")
            f.write(accuracy_str + "\n")

# ----------------------- Main Function -----------------------
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define known adversarial clients
    adversary_ids = [2,3,7,9]  # ground truth list of adversaries

    # Load pre-trained ResNet18 for gradient leakage
    model_resnet = torchvision.models.resnet18(pretrained=False)
    model_resnet.fc = nn.Linear(model_resnet.fc.in_features, 10)
    model_resnet.to(device).eval()

    # Load MNIST dataset
    transform = transforms.Compose([transforms.Resize((32, 32)), transforms.Grayscale(3), transforms.ToTensor()])
    dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Initialize server (will load or pre-train and save global model)
    server = Server(num_clients, device, adversary_ids, server_model_path)

    # Distribute data to clients
    client_data = [[] for _ in range(num_clients)]
    client_labels = [[] for _ in range(num_clients)]

    for i, (real_image, label) in enumerate(dataloader):
        if i >= num_clients * num_data_per_client:
            break
        client_id = i // num_data_per_client
        real_image, label = real_image.to(device), label.to(device)

        if client_id in adversary_ids:
            fake_image = deep_leakage_reconstruct(real_image, model_resnet, label, device)
            client_data[client_id].append(fake_image.cpu())
            client_labels[client_id].append(0)  # Fake
        else:
            client_data[client_id].append(real_image.cpu())
            client_labels[client_id].append(1)  # Real

    for client_id in range(num_clients):
        for image, lbl in zip(client_data[client_id], client_labels[client_id]):
            server.clients[client_id].add_data(image, lbl)

    # Train clients locally
    for client in server.clients:
        client.train(epochs=5)

    # Clients share their updates with the server
    updates = server.receive_updates()

    # Server aggregates the updates, updates global model, and saves it
    server.aggregate_updates(updates)

    # Detect adversarial clients and compute detection accuracy
    server.detect_adversaries()

if __name__ == "__main__":
    main()
