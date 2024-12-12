import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms
import torchvision
import random

# Configuration
num_clients = 10
num_data_per_client = 3
mode = 'test'
server_model_path = "server_model.pth"

# -------------------- DLG Reconstruction Function --------------------
def deep_leakage_reconstruct(real_image, model, real_label, device, steps=200, lr=0.1):
    dummy_data = torch.rand_like(real_image, requires_grad=True, device=device)
    optimizer = optim.Adam([dummy_data], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    criterion = nn.CrossEntropyLoss()

    for step in range(steps):
        optimizer.zero_grad()
        real_output = model(real_image)
        real_loss = criterion(real_output, real_label)
        real_grad = torch.autograd.grad(real_loss, model.parameters(), create_graph=True)

        dummy_output = model(dummy_data)
        dummy_loss = criterion(dummy_output, real_label)
        dummy_grad = torch.autograd.grad(dummy_loss, model.parameters(), create_graph=True)

        grad_loss = sum(torch.norm(dg - rg) for dg, rg in zip(dummy_grad, real_grad))
        grad_loss.backward()
        optimizer.step()
        scheduler.step()
        # Shortened for demonstration

    return dummy_data.detach()

# ----------------------- Dataset -----------------------
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

# ----------------------- Model -----------------------
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

def train_model(model, dataloader, criterion, optimizer, device, epochs=5):
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

def test_single_images(model, test_images, test_labels, device):
    model.eval()
    with torch.no_grad():
        for idx, (image, label) in enumerate(zip(test_images, test_labels)):
            image = image.unsqueeze(0).to(device)
            output = model(image).squeeze()
            prediction = "Real" if output > 0.5 else "Reconstructed"
            confidence = output.item()
            print(f"Image {idx+1}: Prediction={prediction}, Confidence={confidence:.4f}, Ground Truth={'Real' if label == 1 else 'Reconstructed'}")

# ----------------------- Client Classes -----------------------
class Client:
    def __init__(self, client_id, device):
        self.client_id = client_id
        self.device = device
        self.model = RealFakeClassifier().to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.BCELoss()
        self.data = []
        self.labels = []
        # For detection heuristic
        self.queries = 0

    def add_data(self, image, label):
        self.data.append(image)
        self.labels.append(label)

    def train(self, epochs=5):
        if len(self.data) == 0:
            return
        images = torch.cat(self.data)
        labels = torch.tensor(self.labels)
        train_dataset = SingleImageDataset(images, labels)
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        print(f"Training client {self.client_id}...")
        train_model(self.model, train_loader, self.criterion, self.optimizer, self.device, epochs=epochs)

    def get_model_update(self):
        return self.model.state_dict()

    def test(self):
        test_images = torch.cat(self.data)
        test_labels = torch.tensor(self.labels)
        print(f"Testing client {self.client_id}...")
        test_single_images(self.model, test_images, test_labels, self.device)

    def is_adversarial(self):
        # Heuristic for DLG-type adversary: If more than half are fake
        reconstructed_count = sum(label == 0 for label in self.labels)
        if reconstructed_count > (len(self.labels) / 2):
            return True
        # If too many queries without meaningful improvements could also indicate suspicious behavior
        if self.queries > 50:
            return True
        return False

class MIAClient(Client):
    # Membership Inference Attacker
    def __init__(self, client_id, device, target_model, population_dataset):
        super().__init__(client_id, device)
        self.target_model = target_model
        self.population_dataset = population_dataset
        self.attack_model = None

    def build_shadow_models(self, k=1):
        # Build a shadow model for demonstration purposes
        n = len(self.population_dataset)
        indices = list(range(n))
        random.shuffle(indices)
        half = n // 2
        in_indices = indices[:half]   # Shadow training set
        out_indices = indices[half:]  # Shadow test set

        in_subset = Subset(self.population_dataset, in_indices)
        out_subset = Subset(self.population_dataset, out_indices)

        shadow_model = RealFakeClassifier().to(self.device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(shadow_model.parameters(), lr=0.001)

        # Fake training: label all as real for simplicity
        train_images = []
        train_labels = []
        for (img, _) in in_subset:
            train_images.append(img)
            train_labels.append(1)
        train_dataset = SingleImageDataset(torch.stack(train_images), torch.tensor(train_labels))
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        train_model(shadow_model, train_loader, criterion, optimizer, self.device, epochs=2)

        def get_attack_data(model, subset, in_flag):
            model.eval()
            X, Y = [], []
            with torch.no_grad():
                for img, _ in subset:
                    img = img.unsqueeze(0).to(self.device)
                    out = model(img).item()
                    X.append(out)
                    Y.append(1.0 if in_flag else 0.0)
            return X, Y

        inX, inY = get_attack_data(shadow_model, in_subset, True)
        outX, outY = get_attack_data(shadow_model, out_subset, False)

        return inX, inY, outX, outY

    def train_attack_model(self):
        inX, inY, outX, outY = self.build_shadow_models()

        X = inX + outX
        Y = inY + outY

        X = torch.tensor(X).unsqueeze(1)
        Y = torch.tensor(Y)

        attack_dataset = torch.utils.data.TensorDataset(X, Y)
        attack_loader = DataLoader(attack_dataset, batch_size=4, shuffle=True)

        self.attack_model = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16,1),
            nn.Sigmoid()
        ).to(self.device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.attack_model.parameters(), lr=0.001)

        print("Training attack model (MIA)...")
        for epoch in range(5):
            total_loss = 0.0
            for bx, by in attack_loader:
                bx, by = bx.to(self.device).float(), by.to(self.device).float()
                optimizer.zero_grad()
                pred = self.attack_model(bx).squeeze()
                loss = criterion(pred, by)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Attack Model Epoch [{epoch+1}/5], Loss: {total_loss/len(attack_loader):.4f}")

    def perform_mia(self, samples):
        if self.attack_model is None:
            self.train_attack_model()
        self.queries += len(samples)  # Counting suspicious queries
        results = []
        self.target_model.eval()
        with torch.no_grad():
            for img, lbl in samples:
                img = img.unsqueeze(0).to(self.device)
                out = self.target_model(img).item()
                attack_input = torch.tensor([[out]]).float().to(self.device)
                pred = self.attack_model(attack_input).item()
                results.append(("in" if pred>0.5 else "out", out))
        return results

    def is_adversarial(self):
        # For MIA detection heuristic:
        # If the client made many MIA queries, it might be suspicious.
        # Combine with parent heuristic for extra conditions.
        parent_check = super().is_adversarial()
        if parent_check:
            return True
        # If queries exceed some threshold, flag as MIA adversary
        # This threshold is arbitrary.
        if self.queries > 20:
            return True
        return False

# ----------------------- Server Class -----------------------
class Server:
    def __init__(self, num_clients, device, adversary_ids, mia_ids, server_model_path):
        self.num_clients = num_clients
        self.device = device
        self.clients = []
        self.global_model = RealFakeClassifier().to(device)
        self.adversary_ids = adversary_ids
        self.mia_ids = mia_ids
        self.server_model_path = server_model_path

        if not os.path.exists(self.server_model_path):
            raise FileNotFoundError("Pre-trained server model not found.")
        self.global_model.load_state_dict(torch.load(self.server_model_path))
        print(f"Loaded pre-trained server model from {self.server_model_path}")

    def add_client(self, client):
        self.clients.append(client)

    def aggregate_updates(self, updates):
        global_state_dict = self.global_model.state_dict()
        for key in global_state_dict:
            global_state_dict[key] = torch.stack([update[key] for update in updates]).mean(dim=0)
        self.global_model.load_state_dict(global_state_dict)

        # Save updated model
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
                print(f"Client {client.client_id} detected as adversarial.")

        print(f"Adversarial clients detected: {adversarial_clients_detected}")
        # Evaluate detection accuracy if ground truth known
        ground_truth = self.adversary_ids + self.mia_ids
        tp = sum(1 for cid in adversarial_clients_detected if cid in ground_truth)
        fp = sum(1 for cid in adversarial_clients_detected if cid not in ground_truth)
        fn = sum(1 for cid in ground_truth if cid not in adversarial_clients_detected)
        tn = self.num_clients - (tp + fp + fn)
        accuracy = (tp + tn) / self.num_clients
        accuracy_str = f"Adversary Detection Accuracy: {accuracy * 100:.2f}%"
        print(f"Adversary Detection Accuracy: {accuracy*100:.2f}%")

        # Save results to a text file
        with open("results.txt", "w") as f:
            f.write(f"Ground Truth Adversaries: {self.adversary_ids}\n")
            f.write(f"Detected Adversaries: {adversarial_clients_detected}\n")
            f.write(f"TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}\n")   
            f.write(accuracy_str + "\n")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    adversary_ids = [2]      # DLG adversary
    mia_ids = [3]            # MIA adversary

    transform = transforms.Compose([transforms.Resize((32, 32)), transforms.Grayscale(3), transforms.ToTensor()])
    dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Initialize server
    server = Server(num_clients, device, adversary_ids, mia_ids, server_model_path)

    # Create clients
    clients = []
    for i in range(num_clients):
        if i in mia_ids:
            # MIA client: needs population dataset (use MNIST test set)
            test_dataset = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)
            client = MIAClient(i, device, server.global_model, test_dataset)
        else:
            client = Client(i, device)
        clients.append(client)

    # Add clients to server
    for c in clients:
        server.add_client(c)

    # Distribute data
    iter_data = iter(dataloader)
    for i in range(num_clients * num_data_per_client):
        client_id = i // num_data_per_client
        (real_image, label) = next(iter_data)
        real_image, label = real_image.to(device), label.to(device)

        if client_id in adversary_ids:
            # Adversary using DLG
            model_resnet = torchvision.models.resnet18(pretrained=False)
            model_resnet.fc = nn.Linear(model_resnet.fc.in_features, 10)
            model_resnet.to(device).eval()
            fake_image = deep_leakage_reconstruct(real_image, model_resnet, label, device)
            clients[client_id].add_data(fake_image.cpu(), 0)  # Fake label
        else:
            clients[client_id].add_data(real_image.cpu(), 1)  # Real label

    # MIA attackers perform queries before training (just for demonstration)
    for cid in mia_ids:
        samples = []
        for _ in range(21):
            idx = random.randint(0, len(dataset)-1)
            samples.append(dataset[idx])  # (img, lbl)
        results = clients[cid].perform_mia(samples)
        print(f"MIA Client {cid} MIA results:", results)

    # Train clients locally
    for client in clients:
        client.train(epochs=2)  # fewer epochs for demonstration

    # Clients share their updates with the server
    updates = server.receive_updates()

    # Server aggregates the updates
    server.aggregate_updates(updates)

    # Detect adversarial clients
    server.detect_adversaries()

if __name__ == "__main__":
    main()