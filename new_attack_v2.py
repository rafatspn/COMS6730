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

def get_resnet_model(num_classes=10, device='cpu'):
    model_resnet = torchvision.models.resnet18(pretrained=False)
    model_resnet.fc = nn.Linear(model_resnet.fc.in_features, num_classes)
    model_resnet = model_resnet.to(device)
    return model_resnet

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
        self.population = ""
        self.model = models.resnet18(pretrained=False, num_classes=10)  # Use default 3-channel input
        self.model.load_state_dict(initial_weights)
        self.attack_model = None
    
    def build_shadow_models(self, k=3):
        # Initialize lists to accumulate data from multiple shadow models
        inX_total, inY_total = [], []
        outX_total, outY_total = [], []

        if self.population == "mnist":
            self.population_dataset =  MNIST(root="./data", train=True, download=True, transform=transform)   
        else:
            self.population_dataset = FashionMNIST(root="./data", train=True, download=True, transform=transform)

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

            train_loader = self.mnist_loader
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

    # def perform_mia(self, samples):
    #     if self.attack_model is None:
    #         self.train_attack_model()
    #     self.queries += len(samples)
    #     results = []
    #     self.target_model.eval()
    #     with torch.no_grad():
    #         for img, lbl in samples:
    #             img = img.unsqueeze(0).to(self.device)
    #             out = self.target_model(img)
    #             prob = torch.softmax(out, dim=1)
    #             max_conf, _ = torch.max(prob, dim=1)
    #             attack_input = max_conf.unsqueeze(1)
    #             pred = self.attack_model(attack_input).item()
    #             results.append(("in" if pred>0.5 else "out", max_conf.item()))
    #     return results
   
    @staticmethod
    def pca_denoise(image_tensor, n_components_ratio=0.5):
        """
        Apply PCA-based denoising to an image tensor.

        Args:
            image_tensor (torch.Tensor): Image tensor of shape (C, H, W) or (1, H, W) or (H, W).
            n_components_ratio (float): Ratio of components to retain (0 < ratio <= 1).

        Returns:
            torch.Tensor: Denoised image tensor.
        """
        # Detach and convert tensor to numpy array
        image_numpy = image_tensor.detach().cpu().numpy()

        # Handle cases where the image may still have extra dimensions
        if len(image_numpy.shape) == 3:  # Shape is (C, H, W)
            image_numpy = image_numpy[0]  # Use the first channel
        elif len(image_numpy.shape) > 3:
            raise ValueError(f"Unexpected tensor shape: {image_numpy.shape}")

        # Ensure shape is now (H, W)
        H, W = image_numpy.shape

        # Flatten the image into rows for PCA
        flattened = image_numpy.reshape(H, -1)

        # Dynamically set the number of components
        max_components = min(flattened.shape)
        n_components = max(1, int(n_components_ratio * max_components))

        # Apply PCA
        pca = PCA(n_components=n_components, svd_solver='full')
        transformed = pca.fit_transform(flattened)  # Dimensionality reduction
        restored = pca.inverse_transform(transformed)  # Reconstruct the image

        # Reshape back to (1, H, W)
        denoised_image = torch.tensor(restored, dtype=torch.float32).reshape(1, H, W)
        denoised_image = torch.clamp(denoised_image, 0, 1)  # Ensure values are between 0 and 1

        return denoised_image
    

    


    def deep_leakage_reconstruct(self, real_image, real_label, model, steps=2000, lr=0.1):
        model.eval()

        # Define a random initial image with the same shape as the original image
        initial_image = torch.randn_like(real_image, requires_grad=True, device=DEVICE)

        # Define the loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam([initial_image], lr=lr)

        # Perform gradient ascent
        for i in range(steps):
            optimizer.zero_grad()
            output = model(initial_image)
            loss = -criterion(output, torch.tensor([real_label]).to(DEVICE))  # Maximize the probability of the desired label
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print(f'Iteration {i}, Loss: {loss.item()}')

        # Ensure the generated image matches the desired label
        # with torch.no_grad():
        #     final_output = model(initial_image)
        #     predicted_label = torch.argmax(final_output, dim=1)

        # Return the generated image
        generated_image = initial_image.squeeze().cpu()
        return generated_image
    


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
            self.population = "mnist"
        else:
            print("[Malicious Client] Using FashionMNIST for reconstruction.")
            chosen_loader = self.fashion_loader

        if self.attack_model is None:
            self.train_attack_model()

        # Perform DLG attack for each image in the batch
        # real_images, real_labels = next(iter(chosen_loader))
        # real_images, real_labels = real_images.to(self.device), real_labels.to(self.device)

        # print("[Malicious Client] Performing Deep Leakage Attack for each image...")

        # for i in range(real_images.size(0)):  # Loop through each image in the batch
        #     single_image = real_images[i].unsqueeze(0)  # Select one image and add batch dimension
        #     single_label = real_labels[i].unsqueeze(0)  # Select corresponding label and add batch dimension

        #     # Perform DLG attack to reconstruct image
        #     reconstructed_image = self.deep_leakage_reconstruct(single_image, single_label, self.model)

        #     # denoised_image = MaliciousClient.pca_denoise(reconstructed_image.squeeze(), n_components_ratio=0.5)

        #     # Save original and reconstructed images
        #     torchvision.utils.save_image(single_image, f"real_image_{i}.png")
        #     torchvision.utils.save_image(reconstructed_image, f"reconstructed_image_{i}.png")
        #     print(f"Saved real_image_{i}.png and reconstructed_image_{i}.png.")

        # # Continue normal local training and return gradients
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

    # Instantiate model
    # global_model = vgg16(pretrained=False)
    # global_model.classifier[6] = nn.Linear(4096, 10)
    # global_model.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)

    # global_model = SimpleCNN().to(DEVICE)
    # global_model.eval()


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
