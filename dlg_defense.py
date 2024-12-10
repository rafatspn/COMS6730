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
    # Initialize dummy data
    dummy_data = torch.rand_like(real_image, requires_grad=True, device=device)
    optimizer = optim.Adam([dummy_data], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    criterion = nn.CrossEntropyLoss()

    for step in range(steps):
        optimizer.zero_grad()

        # Compute gradients for real image
        real_output = model(real_image)
        real_loss = criterion(real_output, real_label)
        real_grad = torch.autograd.grad(real_loss, model.parameters(), create_graph=True)

        # Compute gradients for dummy image
        dummy_output = model(dummy_data)
        dummy_loss = criterion(dummy_output, real_label)
        dummy_grad = torch.autograd.grad(dummy_loss, model.parameters(), create_graph=True)

        # Gradient matching loss
        grad_loss = sum(torch.norm(dg - rg) for dg, rg in zip(dummy_grad, real_grad))
        grad_loss.backward()
        optimizer.step()
        scheduler.step()  # Update learning rate

        if step % 200 == 0:
            print(f"Step {step}/{steps}, Loss: {grad_loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

    return dummy_data.detach()


# ----------------------- Pairwise Dataset -----------------------
class PairwiseDataset(Dataset):
    def __init__(self, real_images, reconstructed_images):
        self.real_images = real_images
        self.reconstructed_images = reconstructed_images
        self.labels = torch.ones(len(real_images))  # First image is real (label=1)

    def __len__(self):
        return len(self.real_images)

    def __getitem__(self, idx):
        real_img = self.real_images[idx]
        fake_img = self.reconstructed_images[idx]
        return real_img, fake_img, self.labels[idx]

# ----------------------- Siamese Network -----------------------
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, img1, img2):
        feat1 = self.cnn(img1).view(img1.size(0), -1)
        feat2 = self.cnn(img2).view(img2.size(0), -1)
        diff = torch.abs(feat1 - feat2)
        output = self.fc(diff)
        return output

# ----------------------- Train Function -----------------------
def train(model, dataloader, criterion, optimizer, device, epochs=5):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for img1, img2, label in dataloader:
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)

            optimizer.zero_grad()
            output = model(img1, img2).squeeze()
            loss = criterion(output, label.float())
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}")

# ----------------------- Test Function -----------------------
def test_siamese_model(model, dataloader, device):
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():
        for real_img, fake_img, label in dataloader:
            real_img, fake_img, label = real_img.to(device), fake_img.to(device), label.to(device)
            print(label)
            # Predict which image is real
            output = model(real_img, fake_img).squeeze()
            print(output)
            preds = (output > 0.5).float()  # Threshold to classify real vs fake
            correct += (preds == label).sum().item()
            total += label.size(0)

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy

# ----------------------- Main Function -----------------------
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load pre-trained model (ResNet18 for gradient leakage)
    model = torchvision.models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 10)  # Adjust for MNIST
    model.to(device)
    model.eval()

    # Load MNIST dataset
    transform = transforms.Compose([transforms.Resize((32, 32)), transforms.Grayscale(3), transforms.ToTensor()])
    dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    if mode=="train":
        # Generate real and reconstructed images
        real_images, reconstructed_images = [], []
        for i, (real_image, label) in enumerate(dataloader):
            if i >= 16: break  # Generate 16 images for simplicity
            real_image, label = real_image.to(device), label.to(device)
            fake_image = deep_leakage_reconstruct(real_image, model, label, device)
            real_images.append(real_image.cpu())
            reconstructed_images.append(fake_image.cpu())

        real_images = torch.cat(real_images)
        reconstructed_images = torch.cat(reconstructed_images)

        # Save sample images
        save_image(real_images, "real_samples.png")
        save_image(reconstructed_images, "fake_samples.png") 

        # Pairwise dataset and dataloader
        pairwise_dataset = PairwiseDataset(real_images, reconstructed_images)
        pairwise_dataloader = DataLoader(pairwise_dataset, batch_size=4, shuffle=True)

        # Train Siamese Network
        siamese_model = SiameseNetwork().to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(siamese_model.parameters(), lr=0.001)

        print("Training Siamese Network to distinguish real and reconstructed images...")
        train(siamese_model, pairwise_dataloader, criterion, optimizer, device, epochs=5)

        print("Model training complete. You can now evaluate it or use it for prediction.")

        # Save the trained Siamese model
        model_save_path = "siamese_model.pth"
        torch.save(siamese_model.state_dict(), model_save_path)
        print(f"Model saved at {model_save_path}")
    # elif mode=="test":
    #     siamese_model = SiameseNetwork().to(device)
    #     siamese_model.load_state_dict(torch.load("siamese_model.pth", map_location=device))
    #     print("Trained model loaded successfully!")

    #     real_test_images, reconstructed_test_images = [], []
    #     for i, (real_image, label) in enumerate(dataloader):
    #         if i >= 1: break  # Generate 16 images for simplicity
    #         real_image, label = real_image.to(device), label.to(device)
    #         fake_image = deep_leakage_reconstruct(real_image, model, label, device)
    #         real_test_images.append(real_image.cpu())
    #         reconstructed_test_images.append(fake_image.cpu())

    #     real_test_images = torch.cat(real_test_images)
    #     reconstructed_test_images = torch.cat(reconstructed_test_images)

    #     pairwise_dataset = PairwiseDataset(real_test_images, reconstructed_test_images)
        
    #     pairwise_dataloader = DataLoader(pairwise_dataset, batch_size=4, shuffle=False)
    #     test_accuracy = test_siamese_model(siamese_model, pairwise_dataloader, device)

    #     print(f"Test Accuracy: {test_accuracy:.2f}%")
    elif mode == "test":
        # Load the trained model
        siamese_model = SiameseNetwork().to(device)
        siamese_model.load_state_dict(torch.load("siamese_model.pth", map_location=device))
        print("Trained model loaded successfully!")

        # Test images: real and reconstructed images mixed
        test_images, test_labels = [], []  # Labels: 1=real, 0=reconstructed
        for i, (real_image, label) in enumerate(dataloader):
            if i >= 1: break  # Take 5 images for real
            real_image, label = real_image.to(device), label.to(device)
            fake_image = deep_leakage_reconstruct(real_image, model, label, device)

            test_images.append(real_image)  # Real image
            test_labels.append(1)  # Real label

            test_images.append(fake_image)  # Reconstructed image
            test_labels.append(0)  # Reconstructed label

        # Perform predictions one by one
        print("\nTesting individual images...")
        with torch.no_grad():
            for idx, (image, label) in enumerate(zip(test_images, test_labels)):
                image = image.unsqueeze(0).to(device)  # Add batch dimension [1, 3, H, W]
                output = siamese_model(image).squeeze()
                prediction = "Real" if output > 0.5 else "Reconstructed"
                confidence = output.item()
                print(f"Image {idx+1}: Prediction={prediction}, Confidence={confidence:.4f}, Ground Truth={'Real' if label == 1 else 'Reconstructed'}")

        

if __name__ == "__main__":
    main()
