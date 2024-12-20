import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import MNIST, FashionMNIST
import copy
import random
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.utils as vutils
from sklearn.decomposition import PCA

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def choose_random_label(loader):
    """
    Choose a random label from a given data loader.

    Args:
        loader (DataLoader): PyTorch DataLoader object.

    Returns:
        int: A randomly selected label.
    """
    labels = []
    for _, batch_labels in loader:
        labels.extend(batch_labels.tolist())
    return random.choice(labels)


# ------------------ Client Class ---------------------
class Client:
    def __init__(self, client_id, train_loader, initial_weights, device):
        self.client_id = client_id
        self.train_loader = train_loader
        self.device = device
        self.model = models.resnet18(pretrained=False, num_classes=10)  # Use default 3-channel input
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

# ------------------ Modified Malicious Client ---------------------
class MaliciousClient(Client):
    def __init__(self, client_id, mnist_loader, fashion_loader, initial_weights, device):
        super().__init__(client_id, mnist_loader, initial_weights, device)
        self.fashion_loader = fashion_loader
        self.mnist_loader = mnist_loader
        self.generator = self.initialize_generator()

    def initialize_generator(self):
        """Initialize a simple generator network."""
        class Generator(nn.Module):
            def __init__(self, latent_dim, img_shape):
                super(Generator, self).__init__()
                self.model = nn.Sequential(
                    nn.Linear(latent_dim, 128),
                    nn.ReLU(),
                    nn.Linear(128, 256),
                    nn.ReLU(),
                    nn.Linear(256, img_shape[0] * img_shape[1] * img_shape[2]),
                    nn.Tanh(),
                )
                self.img_shape = img_shape

            def forward(self, z):
                img = self.model(z)
                return img.view(img.size(0), *self.img_shape)

        img_shape = (1, 64, 64)
        return Generator(latent_dim=100, img_shape=img_shape).to(self.device)

    def deep_leakage_reconstruct(self, real_image, real_label, model, steps=2000, lr=0.1):
        """Reconstruct image using Deep Leakage from Gradients."""
        def total_variation_loss(img):
            tv_h = torch.mean(torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]))
            tv_w = torch.mean(torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]))
            return tv_h + tv_w

        dummy_data = torch.rand_like(real_image, requires_grad=True, device=self.device)
        optimizer = optim.Adam([dummy_data], lr=lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
        loss_fn = nn.CrossEntropyLoss()

        model.eval()
        for step in range(steps):
            optimizer.zero_grad()

            real_output = model(real_image)
            real_loss = loss_fn(real_output, real_label)
            real_grad = torch.autograd.grad(real_loss, model.parameters(), create_graph=True)

            dummy_output = model(dummy_data)
            dummy_loss = loss_fn(dummy_output, real_label)
            dummy_grad = torch.autograd.grad(dummy_loss, model.parameters(), create_graph=True)

            grad_loss = sum(torch.norm(dg - rg) for dg, rg in zip(dummy_grad, real_grad))
            tv_loss = total_variation_loss(dummy_data)
            total_loss = grad_loss + 1e-4 * tv_loss

            total_loss.backward()
            optimizer.step()
            scheduler.step()

        return dummy_data

    def save_reconstructed_image(self, image, i):
        """Save the reconstructed image."""
        image = image.squeeze().cpu()
        plt.imshow(image, cmap="gray")
        plt.axis("off")
        plt.savefig(f"reconstructed_image_{i}.png")
        plt.close()
        print(f"Reconstructed image saved as 'reconstructed_image_{i}.png'.")

    def local_train(self, global_weights, epochs=1, lr=0.01):
        """Perform local training and reconstruct images."""
        self.model.load_state_dict(global_weights)
        self.model.eval()

        # Confidence evaluation to select dataset
        confidence_mnist = self._evaluate_confidence(self.mnist_loader)
        confidence_fashion = self._evaluate_confidence(self.fashion_loader)

        print(f"[Malicious Client] Confidence on MNIST: {confidence_mnist:.4f}")
        print(f"[Malicious Client] Confidence on FashionMNIST: {confidence_fashion:.4f}")

        if confidence_mnist > confidence_fashion:
            print("[Malicious Client] Using MNIST for reconstruction.")
            chosen_loader = self.mnist_loader
        else:
            print("[Malicious Client] Using FashionMNIST for reconstruction.")
            chosen_loader = self.fashion_loader

        # Perform DLG attack for each image in the batch
        real_images, real_labels = next(iter(chosen_loader))
        real_images, real_labels = real_images.to(self.device), real_labels.to(self.device)

        print("[Malicious Client] Performing Deep Leakage Attack for each image...")

        for i in range(real_images.size(0)):  # Loop through each image in the batch
            single_image = real_images[i].unsqueeze(0)  # Select one image and add batch dimension
            single_label = real_labels[i].unsqueeze(0)  # Select corresponding label and add batch dimension

            # Perform DLG attack to reconstruct image
            reconstructed_image = self.deep_leakage_reconstruct(single_image, single_label, self.model)

            # Save original and reconstructed images
            torchvision.utils.save_image(single_image, f"real_image_{i}.png")
            torchvision.utils.save_image(reconstructed_image, f"reconstructed_image_{i}.png")
            print(f"Saved real_image_{i}.png and reconstructed_image_{i}.png.")

        # Continue normal local training and return gradients
        gradients = super().local_train(global_weights, epochs, lr)
        return gradients
    
    def _evaluate_confidence(self, loader):
        """Evaluate average confidence score for a dataset."""
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

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(3),
        transforms.ToTensor()
    ])


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
    global_model.fc = nn.Linear(global_model.fc.in_features, 10)
    # global_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
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
