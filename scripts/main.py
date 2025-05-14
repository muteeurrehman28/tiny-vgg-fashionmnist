import os
import sys
import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Add project root to sys.path to ensure imports work
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from models.tiny_vgg import TinyVGG
from train import train_model
from evaluate import evaluate_model
from visualize import visualize_feature_maps
from utils.helpers import plot_loss_accuracy

def main():
    # Create output directories using absolute paths
    os.makedirs(os.path.join(project_root, 'oautputs', 'plots'), exist_ok=True)
    os.makedirs(os.path.join(project_root, 'outputs', 'models'), exist_ok=True)

    # Load FashionMNIST dataset
    transform = transforms.Compose([transforms.ToTensor()])
    train_data = datasets.FashionMNIST(root=os.path.join(project_root, 'data'), train=True, download=True, transform=transform)
    test_data = datasets.FashionMNIST(root=os.path.join(project_root, 'data'), train=False, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
    class_names = train_data.classes

    # Initialize model, loss, and optimizer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TinyVGG().to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    epochs = 15

    # Train model
    train_losses, test_losses, train_accs, test_accs = train_model(model, train_loader, test_loader, loss_fn, optimizer, epochs, device)
    plot_loss_accuracy(train_losses, test_losses, train_accs, test_accs)

    # Save model
    torch.save(model.state_dict(), os.path.join(project_root, 'outputs', 'models', 'tiny_vgg.pth'))

    # Evaluate model
    evaluate_model(model, test_loader, loss_fn, class_names, device)

    # Visualize feature maps for the first test image
    image, _ = test_data[0]
    visualize_feature_maps(model, image, device)

if __name__ == "__main__":
    main()