import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
import torchvision
import os
import torchvision.utils as vutils

server_model_path = "real_fake_classifier.pth"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def save_images(images, filename, title="Images"):
    """
    Save a grid of images to a file.
    """
    vutils.save_image(images, filename, nrow=5, normalize=True)
    print(f"{title} saved to {filename}")


# ----------------------- DLG Model: Generate Reconstructed Images From Gradients -----------------------
def deep_leakage_reconstruct(real_image, model, real_label, device, steps=300, lr=0.1):
    """
    Reconstruct images by matching gradients as described in 'Deep Leakage from Gradients',
    with enhanced stability and output quality.
    """
    # Improved initialization for dummy data
    dummy_data = torch.randn_like(real_image, requires_grad=True, device=device)

    dummy_label = real_label.clone().detach()  # Fixed dummy label

    # Optimizer and scheduler
    optimizer = torch.optim.LBFGS([dummy_data], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    criterion = torch.nn.CrossEntropyLoss()

    # Normalize gradients function
    def normalize_gradients(grad_list):
        return [g / (g.norm() + 1e-8) for g in grad_list]

    # Capture gradients of real data
    model.zero_grad()
    real_output = model(real_image)
    real_loss = criterion(real_output, real_label)
    real_gradients = torch.autograd.grad(real_loss, model.parameters(), create_graph=False)
    real_gradients = normalize_gradients([g.detach() for g in real_gradients])

    def closure():
        optimizer.zero_grad()
        dummy_output = model(dummy_data)
        dummy_loss = criterion(dummy_output, dummy_label)

        # Calculate gradients of dummy data
        dummy_gradients = torch.autograd.grad(dummy_loss, model.parameters(), create_graph=True)
        dummy_gradients = normalize_gradients(dummy_gradients)

        # Match gradients (L2 loss)
        grad_diff = sum(torch.nn.functional.mse_loss(g_dummy, g_real)
                        for g_dummy, g_real in zip(dummy_gradients, real_gradients))

        grad_diff.backward()
        return grad_diff

    # Optimize dummy data
    for step in range(steps):
        optimizer.step(closure)
        scheduler.step()
        with torch.no_grad():
            dummy_data.clamp_(0, 1)  # Ensure pixel values are in [0, 1]
        
        if step % 50 == 0:
            loss = closure()
            print(f"Step [{step}/{steps}], Gradient Difference: {loss.item():.6f}")

    return dummy_data.detach()






# ----------------------- Gradient-Based RealFakeClassifier -----------------------
class RealFakeClassifier(nn.Module):
    """
    Classify images based on their gradients rather than raw pixel values.
    """
    def __init__(self):
        super(RealFakeClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(3 * 32 * 32, 128),  # Flattened gradient input
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten gradients
        x = self.fc(x)
        return x


def compute_image_gradients(model, images, labels, criterion, device):
    """
    Compute gradients of input images with respect to the model loss.
    """
    images.requires_grad = True
    outputs = model(images)
    loss = criterion(outputs, labels)
    gradients = torch.autograd.grad(loss, images, grad_outputs=torch.ones_like(loss), create_graph=False)[0]
    return gradients.detach()


def train(model, dataloader, criterion, optimizer, resnet, device, epochs=5):
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device).float()

            # Compute gradients for images
            gradients = compute_image_gradients(resnet, images, labels, criterion, device)

            optimizer.zero_grad()
            outputs = model(gradients).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss / len(dataloader):.4f}")


def evaluate(model, dataloader, resnet, criterion, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            gradients = compute_image_gradients(resnet, images, labels, criterion, device)
            outputs = model(gradients).squeeze()
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

    # Collect real and fake images
    real_images_list = []
    real_labels_list = []
    for i, (img, lbl) in enumerate(dataloader):
        real_images_list.append(img.to(device))
        real_labels_list.append(lbl.to(device))
        if i >= 0:
            break

     # For each real image, reconstruct a fake image
    fake_images_list = []
    fake_labels_list = []
    for idx, (real_img, real_lbl) in enumerate(zip(real_images_list, real_labels_list)):
        fake_img = deep_leakage_reconstruct(real_img, model_resnet, real_lbl, device)
        fake_images_list.append(fake_img.cpu())
        fake_labels_list.append(torch.tensor([0]))  # Fake label = 0

        # Save real and fake images for comparison
        save_images(real_img.cpu(), f"real_image_{idx}.png", title=f"Real Image {idx}")
        save_images(fake_img.cpu(), f"reconstructed_image_{idx}.png", title=f"Reconstructed Image {idx}")


    real_images = torch.cat(real_images_list, dim=0)
    real_labels = torch.ones(len(real_images_list))
    fake_images = torch.cat(fake_images_list, dim=0)
    fake_labels = torch.cat(fake_labels_list).float()

    # Save the top 10 real and reconstructed images for comparison
    save_images(real_images[:10], "top10_real_images.png", title="Top 10 Real Images")
    save_images(fake_images[:10], "top10_reconstructed_images.png", title="Top 10 Reconstructed Images")

    real_images = real_images.to(device)
    fake_images = fake_images.to(device)



    train_images = torch.cat([real_images, fake_images], dim=0)
    train_labels = torch.cat([real_labels, fake_labels], dim=0)
    train_dataset = TensorDataset(train_images, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    # Initialize the classifier
    model = RealFakeClassifier().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Training classifier with gradient-based inputs...")
    train(model, train_loader, criterion, optimizer, model_resnet, device, epochs=5)

    test_accuracy = evaluate(model, train_loader, model_resnet, criterion, device)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

    torch.save(model.state_dict(), server_model_path)
    print(f"Classifier model saved to {server_model_path}")


if __name__ == "__main__":
    main()
