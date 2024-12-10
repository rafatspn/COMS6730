import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision
from torchvision.utils import save_image

mode = 'test'

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
            print(f"Image {idx+1}: Prediction={prediction}, Confidence={confidence:.4f}, Ground Truth={'Real' if label == 1 else 'Reconstructed'}")


# ----------------------- Main Function -----------------------
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load pre-trained ResNet18 for gradient leakage
    model_resnet = torchvision.models.resnet18(pretrained=False)
    model_resnet.fc = nn.Linear(model_resnet.fc.in_features, 10)
    model_resnet.to(device).eval()

    # Load MNIST dataset
    transform = transforms.Compose([transforms.Resize((32, 32)), transforms.Grayscale(3), transforms.ToTensor()])
    dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    if mode == "train":
        # Generate real and reconstructed images
        images = []
        real_images, reconstructed_images, labels = [], [], []
        for i, (real_image, label) in enumerate(dataloader):
            if i >= 20: break
            real_image, label = real_image.to(device), label.to(device)
            fake_image = deep_leakage_reconstruct(real_image, model_resnet, label, device)
            images.append(real_image.cpu())
            labels.append(1)  # Real
            images.append(fake_image.cpu())
            real_images.append(real_image.cpu())
            reconstructed_images.append(fake_image.cpu())
            labels.append(0)  # Fake

        # images = real_images + reconstructed_images
        images = torch.cat(images)
        labels = torch.tensor(labels)

        real_images = torch.cat(real_images)
        reconstructed_images = torch.cat(reconstructed_images)
        save_image(real_images, "real_samples.png")
        save_image(reconstructed_images, "fake_samples.png") 

        # Dataset and Dataloader
        train_dataset = SingleImageDataset(images, labels)
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

        # Train Classifier
        model = RealFakeClassifier().to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        print("Training RealFakeClassifier...")
        train(model, train_loader, criterion, optimizer, device, epochs=5)

        torch.save(model.state_dict(), "real_fake_classifier.pth")
        print("Model saved as real_fake_classifier.pth")

    elif mode == "test":
        # Load trained model
        model = RealFakeClassifier().to(device)
        model.load_state_dict(torch.load("real_fake_classifier.pth", map_location=device))
        print("Trained model loaded successfully!")

        # Generate test images
        test_images, test_labels = [], []
        for i, (real_image, label) in enumerate(dataloader):
            if i >= 20: break
            real_image, label = real_image.to(device), label.to(device)
            fake_image = deep_leakage_reconstruct(real_image, model_resnet, label, device)

            test_images.append(real_image)
            test_labels.append(1)  # Real
            test_images.append(fake_image)
            test_labels.append(0)  # Fake

        test_images = torch.cat(test_images)
        test_labels = torch.tensor(test_labels)

        # Test individual images
        test_single_images(model, test_images, test_labels, device)


if __name__ == "__main__":
    main()
