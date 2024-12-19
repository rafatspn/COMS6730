import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms
import torchvision
import random
from torchvision.utils import save_image

# Configuration
num_clients = 10
num_data_per_client = 3
mode = 'test'
server_model_path = "server_model.pth"
reconstructed_folder = "reconstructed_images"
os.makedirs(reconstructed_folder, exist_ok=True)

# -------------------- DLG Reconstruction Function --------------------
def deep_leakage_reconstruct(real_gradients, real_label, model, device, steps=200, lr=0.1):
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

# ----------------------- Models -----------------------
def get_resnet_model(num_classes=10, device='cpu'):
    model_resnet = torchvision.models.resnet18(pretrained=False)
    model_resnet.fc = nn.Linear(model_resnet.fc.in_features, num_classes)
    model_resnet = model_resnet.to(device)
    return model_resnet

# Use the original RealFakeClassifier structure (CNN-based)
class RealFakeClassifier(nn.Module):
    def __init__(self):
        super(RealFakeClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(1000, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def train_model(model, dataloader, criterion, optimizer, device, epochs=5):
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device).long()
            optimizer.zero_grad()
            outputs = model(images)
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
        self.model = get_resnet_model(num_classes=10, device=device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        self.data = []
        self.labels = []
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

    def get_gradients(self):
        gradients = []
        for image, label in zip(self.data, self.labels):
            image = image.unsqueeze(0).to(self.device)
            label = torch.tensor([label]).to(self.device)
            self.model.zero_grad()
            output = self.model(image)
            loss = self.criterion(output, label)
            loss.backward()
            gradients.append([param.grad.clone() for param in self.model.parameters()])
        return gradients

    def test(self, classifier, device):
        gradients = self.get_gradients()
        test_gradients = torch.cat([torch.cat([g.view(-1) for g in grad]) for grad in gradients]).unsqueeze(0)
        print(f"Testing client {self.client_id} using RealFakeClassifier...")
        classifier.eval()
        with torch.no_grad():
            outputs = classifier(test_gradients.to(device))
            preds = (outputs.squeeze() > 0.5).float()
        # Return fraction of fake predictions
        fake_ratio = (preds == 0).sum().item() / len(preds)
        return fake_ratio

class MIAClient(Client):
    def __init__(self, client_id, device, target_model, population_dataset):
        super().__init__(client_id, device)
        self.target_model = target_model
        self.population_dataset = population_dataset
        self.attack_model = None

    def build_shadow_models(self, k=3):
        # Initialize lists to accumulate data from multiple shadow models
        inX_total, inY_total = [], []
        outX_total, outY_total = [], []

        n = len(self.population_dataset)
        for _ in range(k):
            indices = list(range(n))
            random.shuffle(indices)
            half = n // 2
            in_indices = indices[:half]
            out_indices = indices[half:]

            in_subset = Subset(self.population_dataset, in_indices)
            out_subset = Subset(self.population_dataset, out_indices)

            shadow_model = get_resnet_model(num_classes=10, device=self.device)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(shadow_model.parameters(), lr=0.001)

            train_images = []
            train_labels = []
            for (img, lbl) in in_subset:
                train_images.append(img)
                train_labels.append(lbl)

            train_dataset = SingleImageDataset(torch.stack(train_images), torch.tensor(train_labels))
            train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
            train_model(shadow_model, train_loader, criterion, optimizer, self.device, epochs=2)

            def get_attack_data(model, subset, in_flag):
                model.eval()
                X, Y = [], []
                with torch.no_grad():
                    for img, _lbl in subset:
                        img = img.unsqueeze(0).to(self.device)
                        out = model(img)
                        prob = torch.softmax(out, dim=1)
                        max_conf, _ = torch.max(prob, dim=1)
                        X.append(max_conf.item())
                        Y.append(1.0 if in_flag else 0.0)
                return X, Y

            inX, inY = get_attack_data(shadow_model, in_subset, True)
            outX, outY = get_attack_data(shadow_model, out_subset, False)

            # Accumulate data
            inX_total.extend(inX)
            inY_total.extend(inY)
            outX_total.extend(outX)
            outY_total.extend(outY)

        # Return all accumulated data from k shadow models
        return inX_total, inY_total, outX_total, outY_total

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
        self.queries += len(samples)
        results = []
        self.target_model.eval()
        with torch.no_grad():
            for img, lbl in samples:
                img = img.unsqueeze(0).to(self.device)
                out = self.target_model(img)
                prob = torch.softmax(out, dim=1)
                max_conf, _ = torch.max(prob, dim=1)
                attack_input = max_conf.unsqueeze(1)
                pred = self.attack_model(attack_input).item()
                results.append(("in" if pred>0.5 else "out", max_conf.item()))
        return results

# ----------------------- Server Class -----------------------
class Server:
    def __init__(self, num_clients, device, adversary_ids, mia_ids, server_model_path):
        self.num_clients = num_clients
        self.device = device
        self.clients = []
        self.global_model = get_resnet_model(num_classes=10, device=device)
        self.adversary_ids = adversary_ids
        self.mia_ids = mia_ids
        # If no pre-trained server model found, that's fine; we start from scratch
        if os.path.exists(server_model_path):
            self.global_model.load_state_dict(torch.load(server_model_path))
            print(f"Loaded pre-trained server model from {server_model_path}")
        else:
            print("No pre-trained server model found. Starting from scratch.")
        self.server_model_path = server_model_path

        # Initialize RealFakeClassifier (untrained or load if desired)
        self.real_fake_classifier = RealFakeClassifier().to(device)
        # If you have a pre-trained RealFakeClassifier, you can load it here:
        if os.path.exists("real_fake_classifier.pth"):
            self.real_fake_classifier.load_state_dict(torch.load("real_fake_classifier.pth"))
            print("Loaded pre-trained RealFakeClassifier.")
        else:
            print("No pre-trained RealFakeClassifier found, using untrained classifier.")

    def add_client(self, client):
        self.clients.append(client)

    def receive_updates(self):
        updates = []
        for client in self.clients:
            updates.append(client.get_model_update())
        return updates

    def detect_adversaries(self):
        # Use the RealFakeClassifier to determine which clients are adversarial.
        # We'll classify each client's gradients. If more than half are predicted fake, we consider it adversarial.
        adversarial_clients_detected = []
        for client in self.clients:
            fake_ratio = client.test(self.real_fake_classifier, self.device)
            # If more than half of a client's gradients are predicted as fake:
            if fake_ratio > 0.5:
                adversarial_clients_detected.append(client.client_id)
                print(f"Client {client.client_id} detected as adversarial.")

        # Additional detection for MIA: Check query count
        for client in self.clients:
            if client.client_id not in adversarial_clients_detected:
                if isinstance(client, MIAClient) and client.queries > 20:
                    adversarial_clients_detected.append(client.client_id)
                    print(f"Client {client.client_id} detected as adversarial (MIA).")

        return adversarial_clients_detected

    def aggregate_updates(self, updates, adversarial_clients):
        # Exclude adversarial clients from aggregation
        filtered_updates = [updates[i] for i in range(len(updates)) if self.clients[i].client_id not in adversarial_clients]

        if len(filtered_updates) == 0:
            print("No real updates found. Skipping aggregation.")
            return

        global_state_dict = self.global_model.state_dict()
        for key in global_state_dict:
            # global_state_dict[key] = torch.stack([u[key] for u in filtered_updates]).mean(dim=0)
            global_state_dict[key] = torch.stack([u[key].float() for u in filtered_updates]).mean(dim=0)

        self.global_model.load_state_dict(global_state_dict)

        torch.save(self.global_model.state_dict(), self.server_model_path)
        print(f"Updated server model saved to {self.server_model_path}")

    def save_results(self, adversarial_clients_detected):
        ground_truth = self.adversary_ids + self.mia_ids
        tp = sum(1 for cid in adversarial_clients_detected if cid in ground_truth)
        fp = sum(1 for cid in adversarial_clients_detected if cid not in ground_truth)
        fn = sum(1 for cid in ground_truth if cid not in adversarial_clients_detected)
        tn = self.num_clients - (tp + fp + fn)
        accuracy = (tp + tn) / self.num_clients
        accuracy_str = f"Adversary Detection Accuracy: {accuracy * 100:.2f}%"
        print(accuracy_str)

        with open("results.txt", "w") as f:
            f.write(f"Ground Truth Adversaries: {ground_truth}\n")
            f.write(f"Detected Adversaries: {adversarial_clients_detected}\n")
            f.write(f"TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}\n")
            f.write(accuracy_str + "\n")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    adversary_ids = [2,7]      # DLG adversary
    mia_ids = [3]            # MIA adversary

    transform = transforms.Compose([transforms.Resize((32, 32)), transforms.Grayscale(3), transforms.ToTensor()])
    dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    server = Server(num_clients, device, adversary_ids, mia_ids, server_model_path)

    clients = []
    for i in range(num_clients):
        if i in mia_ids:
            test_dataset = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)
            client = MIAClient(i, device, server.global_model, test_dataset)
        else:
            client = Client(i, device)
        clients.append(client)

    for c in clients:
        server.add_client(c)

    iter_data = iter(dataloader)
    for i in range(num_clients * num_data_per_client):
        client_id = i // num_data_per_client
        (real_image, label) = next(iter_data)
        real_image, label = real_image.to(device), label.to(device)

        if client_id in adversary_ids:
            # Compute real gradients from real data
            model_resnet = get_resnet_model(num_classes=10, device=device).eval()
            real_image.requires_grad = True
            criterion = nn.CrossEntropyLoss()
            real_output = model_resnet(real_image)
            real_loss = criterion(real_output, label)
            real_gradients = torch.autograd.grad(real_loss, model_resnet.parameters(), create_graph=False)

            # Reconstruct using gradient matching
            fake_image, fake_label = deep_leakage_reconstruct(real_gradients, label, model_resnet, device)

            # Save reconstructed image
            save_image(fake_image, os.path.join(reconstructed_folder, f"client_{client_id}_image_{i}.png"))
            print(f"Client {client_id} reconstructed label: {fake_label.item()}")

            # Label fake images as '0' (fake) for ground truth
            clients[client_id].add_data(fake_image.cpu(), 0)

    # MIA attackers perform queries
    for cid in mia_ids:
        samples = []
        for _ in range(21):
            idx = random.randint(0, len(dataset)-1)
            samples.append(dataset[idx])
        results = clients[cid].perform_mia(samples)
        print(f"MIA Client {cid} MIA results:", results)

    # Train clients
    for client in clients:
        client.train(epochs=5)  # fewer epochs for demonstration

    # Receive updates
    updates = server.receive_updates()

    # Detect adversaries and aggregate updates
    adversarial_clients_detected = server.detect_adversaries()
    server.aggregate_updates(updates, adversarial_clients_detected)

    # Save detection results
    server.save_results(adversarial_clients_detected)

if __name__ == "__main__":
    main()