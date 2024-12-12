import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision
from torchvision.utils import save_image
import random

# ----------------------- Existing DLG Model -----------------------
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

# ----------------------- Federated Client -----------------------
class FederatedClient:
    def __init__(self, id, dataloader, model, adversarial=False):
        self.id = id
        self.dataloader = dataloader
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.adversarial = adversarial
        self.local_gradients = None
        self.reconstructed_image = None
        self.label = None

    def local_train(self, epochs=1):
        criterion = nn.CrossEntropyLoss()
        self.model.train()

        for epoch in range(epochs):
            total_loss = 0.0
            for images, labels in self.dataloader:
                images, labels = images.to(device), labels.to(device)
                self.optimizer.zero_grad()

                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()

                total_loss += loss.item()
                self.local_gradients = [p.grad.data.clone() for p in self.model.parameters()]
                self.label = labels[0].unsqueeze(0)  # Single label for simplicity

                # Adversarial behavior: Reconstruct the image if client is adversarial
                if self.adversarial:
                    self.reconstructed_image = deep_leakage_reconstruct(images, self.model, labels, device)
                    print(f"Adversarial Client {self.id} reconstructed an image")

                self.optimizer.step()
        print(f"Client {self.id} finished training: Loss = {total_loss / len(self.dataloader):.4f}")

    def upload_gradients(self):
        return self.local_gradients

# ----------------------- Federated Server -----------------------
class FederatedServer:
    def __init__(self, global_model, num_clients):
        self.global_model = global_model
        self.num_clients = num_clients
        self.real_fake_classifier = RealFakeClassifier().to(device)
        self.real_fake_classifier.load_state_dict(torch.load("real_fake_classifier.pth", map_location=device))

    def aggregate_gradients(self, client_gradients):
        with torch.no_grad():
            for global_param, grads in zip(self.global_model.parameters(), zip(*client_gradients)):
                global_param.grad = torch.mean(torch.stack(grads), dim=0)

    def detect_adversary(self, clients):
        print("\nDetecting Adversarial Clients...")
        correct_detections = 0
        total_adversarial = 0
        detection_logs = []
        for client in clients:
            if client.adversarial and client.reconstructed_image is not None:
                total_adversarial += 1
                image = client.reconstructed_image.unsqueeze(0).to(device)
                output = self.real_fake_classifier(image).squeeze()
                prediction = "Real" if output > 0.5 else "Reconstructed"
                confidence = output.item()
                ground_truth = "Reconstructed"
                correct = prediction == ground_truth
                if correct:
                    correct_detections += 1
                detection_logs.append((client.id, ground_truth, prediction, confidence, correct))
                print(f"Client {client.id}: Detection - {prediction} (Confidence: {confidence:.4f}), Ground Truth: {ground_truth}")
        
        # Accuracy
        detection_accuracy = correct_detections / total_adversarial if total_adversarial > 0 else 0.0
        print(f"Adversarial Detection Accuracy: {detection_accuracy * 100:.2f}%")
        print("Detailed Logs:")
        for log in detection_logs:
            print(f"Client {log[0]} | Ground Truth: {log[1]} | Prediction: {log[2]} | Confidence: {log[3]:.4f} | Correct: {log[4]}")

# ----------------------- RealFakeClassifier -----------------------
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

# ----------------------- Main Function -----------------------
def main():
    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data Preparation
    transform = transforms.Compose([transforms.Resize((32, 32)), transforms.Grayscale(3), transforms.ToTensor()])
    dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)

    # Define number of samples per client
    n = 2  # Define how many samples per client
    subsets = torch.utils.data.random_split(dataset, [n]*5 + [len(dataset) - n*5])[:5]

    # Initialize Global Model and Clients
    global_model = torchvision.models.resnet18(pretrained=False)
    global_model.fc = nn.Linear(global_model.fc.in_features, 10)
    global_model.to(device)

    clients = []
    for i, subset in enumerate(subsets):
        dataloader = DataLoader(subset, batch_size=8, shuffle=True)
        adversarial = True if i in random.sample(range(5), 2) else False  # Randomly assign adversarial clients
        clients.append(FederatedClient(i, dataloader, global_model, adversarial=adversarial))
        print(f"Client {i}: {'Adversarial' if adversarial else 'Honest'}")

    server = FederatedServer(global_model, num_clients=5)

    # Federated Training Round
    for round_num in range(1, 4):  # 3 rounds for demonstration
        print(f"\n### Federated Round {round_num} ###")
        client_gradients = []
        for client in clients:
            client.local_train(epochs=1)
            client_gradients.append(client.upload_gradients())
        server.aggregate_gradients(client_gradients)
        print("Server aggregated gradients")

    # Adversary Detection
    server.detect_adversary(clients)

if __name__ == "__main__":
    main()
