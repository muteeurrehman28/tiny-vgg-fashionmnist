import matplotlib.pyplot as plt
import os

# Define project root for consistent output paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def plot_loss_accuracy(train_losses, test_losses, train_accs, test_accs):
    os.makedirs(os.path.join(project_root, 'outputs', 'plots'), exist_ok=True)
    
    # Plot loss
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.legend()
    plt.title("Loss vs Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(project_root, 'outputs', 'plots', 'loss_curve.png'))
    plt.show()
    
    # Plot accuracy
    plt.figure(figsize=(8, 6))
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(test_accs, label='Test Accuracy')
    plt.legend()
    plt.title("Accuracy vs Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.savefig(os.path.join(project_root, 'outputs', 'plots', 'accuracy_curve.png'))
    plt.show()