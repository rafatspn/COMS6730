import os
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def save_reconstructed_image(image_tensor, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    save_image(image_tensor, filename)

# -------------------- DLG Reconstruction Function --------------------
def dlg_reconstruct(real_image, model, real_label, device, steps=3000, lr=0.01):
    dummy_data = torch.rand_like(real_image, requires_grad=True, device=device)
    optimizer = optim.Adam([dummy_data], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    criterion = nn.CrossEntropyLoss()

    best_loss = math.inf
    best_img = None

    for step in range(steps):
        optimizer.zero_grad()
        real_output = model(real_image)
        real_loss = criterion(real_output, real_label)
        real_grad = torch.autograd.grad(real_loss, model.parameters(), create_graph=True)

        dummy_output = model(dummy_data)
        dummy_loss = criterion(dummy_output, real_label)
        dummy_grad = torch.autograd.grad(dummy_loss, model.parameters(), create_graph=True)

        grad_loss = sum((dg - rg).pow(2).sum() for dg, rg in zip(dummy_grad, real_grad))
        grad_loss.backward()
        optimizer.step()
        scheduler.step()

        if grad_loss.item() < best_loss:
            best_loss = grad_loss.item()
            best_img = dummy_data.detach().clone()

    return best_img

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

# ----------------------- Model -----------------------
class SimpleClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleClassifier, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.fc = nn.Sequential(
            nn.Linear(32*8*8, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def train_model(model, dataloader, criterion, optimizer, device, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss/len(dataloader):.4f}")

def test_model(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total if total > 0 else 0.0
    return accuracy

# ---------------- MIA Attack Model Training ----------------
def train_shadow_models_and_attack(population_dataset, num_shadow_models=3, device=torch.device('cpu')):
    n = len(population_dataset)
    indices = list(range(n))
    random.shuffle(indices)

    chunk_size = n // (2 * num_shadow_models)
    criterion = nn.CrossEntropyLoss()

    attack_X = []
    attack_Y = []

    for i in range(num_shadow_models):
        in_start = i*2*chunk_size
        in_end = in_start + chunk_size
        out_start = in_end
        out_end = out_start + chunk_size

        in_indices = indices[in_start:in_end]
        out_indices = indices[out_start:out_end]

        in_subset = Subset(population_dataset, in_indices)
        out_subset = Subset(population_dataset, out_indices)

        shadow_model = SimpleClassifier().to(device)
        optimizer = optim.Adam(shadow_model.parameters(), lr=0.001)
        train_loader = DataLoader(in_subset, batch_size=32, shuffle=True)
        print(f"Training Shadow Model {i+1}/{num_shadow_models}...")
        train_model(shadow_model, train_loader, criterion, optimizer, device, epochs=5)

        def get_confidences(model, subset, in_flag):
            model.eval()
            X, Y = [], []
            with torch.no_grad():
                for img, lbl in subset:
                    img = img.unsqueeze(0).to(device)
                    out = model(img)
                    probs = nn.Softmax(dim=1)(out)
                    conf = probs.max(dim=1)[0].item()
                    X.append(conf)
                    Y.append(1.0 if in_flag else 0.0)
            return X, Y

        inX, inY = get_confidences(shadow_model, in_subset, True)
        outX, outY = get_confidences(shadow_model, out_subset, False)

        attack_X += inX + outX
        attack_Y += inY + outY

    attack_X = torch.tensor(attack_X).unsqueeze(1)
    attack_Y = torch.tensor(attack_Y)

    attack_dataset = torch.utils.data.TensorDataset(attack_X, attack_Y)
    attack_loader = DataLoader(attack_dataset, batch_size=16, shuffle=True)

    attack_model = nn.Sequential(
        nn.Linear(1, 32),
        nn.ReLU(),
        nn.Linear(32,1),
        nn.Sigmoid()
    ).to(device)

    attack_criterion = nn.BCELoss()
    attack_optimizer = optim.Adam(attack_model.parameters(), lr=0.001)

    print("Training MIA Attack Model...")
    for epoch in range(10):
        total_loss = 0.0
        attack_model.train()
        for bx, by in attack_loader:
            bx, by = bx.to(device).float(), by.to(device).float()
            attack_optimizer.zero_grad()
            pred = attack_model(bx).squeeze()
            loss = attack_criterion(pred, by)
            loss.backward()
            attack_optimizer.step()
            total_loss += loss.item()
        print(f"Attack Model Epoch [{epoch+1}/10], Loss: {total_loss/len(attack_loader):.4f}")

    # Evaluate attack model accuracy on the same dataset
    # For demonstration, we reuse the same dataset. In practice, use a separate test set.
    attack_model.eval()
    with torch.no_grad():
        bx_all, by_all = attack_X.to(device).float(), attack_Y.to(device).float()
        preds = attack_model(bx_all).squeeze()
        pred_labels = (preds > 0.5).float()
        correct = (pred_labels == by_all).sum().item()
        total = by_all.size(0)
        attack_accuracy = correct / total if total > 0 else 0.0
    print(f"Attack Model Accuracy: {attack_accuracy*100:.2f}%")

    return attack_model

def main():
    # Prepare data
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(3),
        transforms.ToTensor()
    ])
    train_dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    # Split training data
    target_indices = list(range(10000))
    population_indices = list(range(10000, 20000))

    target_subset = Subset(train_dataset, target_indices)
    population_subset = Subset(train_dataset, population_indices)

    target_loader = DataLoader(target_subset, batch_size=32, shuffle=True)

    # Train target model
    target_model = SimpleClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(target_model.parameters(), lr=0.001)

    print("Training Target Model...")
    train_model(target_model, target_loader, criterion, optimizer, device, epochs=10)

    # Evaluate target model accuracy on test set
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    target_acc = test_model(target_model, test_loader, device)
    print(f"Target Model Test Accuracy: {target_acc*100:.2f}%")

    # Train MIA attack model
    attack_model = train_shadow_models_and_attack(population_subset, num_shadow_models=3, device=device)

    # Perform DLG-based reconstruction
    dl = DataLoader(target_subset, batch_size=1, shuffle=True)
    num_samples_to_attack = 20
    count = 0
    reconstruction_results = []

    target_model.eval()
    for (img, lbl) in dl:
        if count >= num_samples_to_attack:
            break
        img, lbl = img.to(device), lbl.to(device)

        # Run DLG
        reconstructed = deep_leakage_reconstruct(img, target_model, lbl, device, steps=3000, lr=0.01)

        # Save the best reconstructed image
        save_filename = f"reconstructed_images/sample_{count+1}.png"
        save_reconstructed_image(reconstructed[0].cpu(), save_filename)
        print(f"Saved reconstructed image to {save_filename}")

        with torch.no_grad():
            out = target_model(reconstructed)
            probs = nn.Softmax(dim=1)(out)
            conf = probs.max(dim=1)[0].item()

            attack_model.eval()
            c_tensor = torch.tensor([[conf]]).float().to(device)
            pred = attack_model(c_tensor).item()
            mia_pred_in = (pred > 0.5)

        reconstruction_results.append(mia_pred_in)
        count += 1
        print(f"Reconstructed sample {count}/{num_samples_to_attack}: MIA says {'in' if mia_pred_in else 'out'}")

    # Compute attack success rate
    success_rate = sum(reconstruction_results) / len(reconstruction_results) if reconstruction_results else 0.0
    print(f"Attack Success Rate: {success_rate*100:.2f}%")

if __name__ == "__main__":
    main()
