import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
import torchvision
import os

server_model_path = "real_fake_classifier.pth"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ----------------------- DLG Model: Generate Reconstructed Images -----------------------
def deep_leakage_reconstruct(real_image, model, real_label, device, steps=1000, lr=0.1):
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

        # For demonstration, we run full steps.  
        # In practice, you might reduce the steps for speed.
        
    return dummy_data.detach()

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

def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).squeeze()
            predictions = (outputs > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total if total > 0 else 0
    return accuracy

def main():
    if os.path.exists(server_model_path):
        print(f"Model already exists at {server_model_path}. Delete it if you want to re-train.")
        return

    # Load MNIST dataset for pre-training
    transform = transforms.Compose([transforms.Resize((32,32)), transforms.Grayscale(3), transforms.ToTensor()])
    dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Initialize the model used for reconstruction (ResNet18)
    model_resnet = torchvision.models.resnet18(pretrained=False)
    model_resnet.fc = nn.Linear(model_resnet.fc.in_features, 10)
    model_resnet.to(device).eval()

    # Collect real images for training
    # For example, take 5 real images for training
    real_images_list = []
    real_labels_list = []
    count = 0
    for img, lbl in dataloader:
        real_images_list.append(img.to(device))
        real_labels_list.append(lbl.to(device))
        count += 1
        if count >= 5:
            break

    # For each real image in training set, create a fake (reconstructed) image
    fake_images_list = []
    fake_labels_list = []
    for real_img, real_lbl in zip(real_images_list, real_labels_list):
        fake_img = deep_leakage_reconstruct(real_img, model_resnet, real_lbl, device)
        fake_images_list.append(fake_img.cpu())
        fake_labels_list.append(torch.tensor([0]))  # fake label = 0

    # Convert real images and labels to tensors
    real_images = torch.cat([x.cpu() for x in real_images_list], dim=0)
    real_labels = torch.ones(len(real_images_list))  # label 1 for real

    fake_images = torch.cat(fake_images_list, dim=0)
    fake_labels = torch.cat(fake_labels_list).float()

    # Combine real and fake for training
    train_images = torch.cat([real_images, fake_images], dim=0)
    train_labels = torch.cat([real_labels, fake_labels], dim=0)

    train_dataset = TensorDataset(train_images, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    # Create test dataset similarly:
    # Take another 5 real images (different from the training set) for testing
    test_real_images_list = []
    test_real_labels_list = []
    count_test = 0
    # Use a new dataloader instance or reset
    test_dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    for img, lbl in test_dataloader:
        # Try to pick different ones: this is a simple approach
        # In a real scenario, you could split dataset properly.
        # Here we just skip the first 5 used for training.
        if count_test < 5:  
            # skip first 5 since they're training (not guaranteed unique, but likely)
            count_test += 1
            continue
        # Now take next 5 for testing
        test_real_images_list.append(img.to(device))
        test_real_labels_list.append(lbl.to(device))
        if len(test_real_images_list) >= 5:
            break

    # Create fake images for testing
    test_fake_images_list = []
    test_fake_labels_list = []
    for real_img, real_lbl in zip(test_real_images_list, test_real_labels_list):
        fake_img = deep_leakage_reconstruct(real_img, model_resnet, real_lbl, device)
        test_fake_images_list.append(fake_img.cpu())
        test_fake_labels_list.append(torch.tensor([0])) # fake label = 0



    test_real_images = torch.cat([x.cpu() for x in test_real_images_list], dim=0)
    test_real_labels = torch.ones(len(test_real_images_list))  # label 1 for real

    test_fake_images = torch.cat(test_fake_images_list, dim=0)
    test_fake_labels = torch.cat(test_fake_labels_list).float()

    # Combine real and fake for testing
    test_images = torch.cat([test_real_images, test_fake_images], dim=0)
    test_labels = torch.cat([test_real_labels, test_fake_labels], dim=0)

    test_dataset = TensorDataset(test_images, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    # Initialize the classifier model
    model = RealFakeClassifier().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Pre-training the server model with MNIST real & reconstructed data...")
    train(model, train_loader, criterion, optimizer, device, epochs=5)

    # Evaluate on test dataset
    test_accuracy = evaluate(model, test_loader, device)
    print(f"Pre-training Test Accuracy: {test_accuracy * 100:.2f}%")

    torch.save(model.state_dict(), server_model_path)
    print(f"Pre-trained server model saved to {server_model_path}")

if __name__ == "__main__":
    main()
