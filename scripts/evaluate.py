import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os

# Define project root for consistent output paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def evaluate_model(model, test_loader, loss_fn, class_names, device):
    model.eval()
    test_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            test_loss += loss.item()
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_loss /= len(test_loader)
    test_acc = correct / total
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
    
    # Plot confusion matrix
    os.makedirs(os.path.join(project_root, 'outputs', 'plots'), exist_ok=True)
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(project_root, 'outputs', 'plots', 'confusion_matrix.png'))
    plt.show()