import torch
import matplotlib.pyplot as plt
import os

# Define project root for consistent output paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def visualize_feature_maps(model, image, device):
    model.eval()
    image_tensor = image.unsqueeze(0).to(device)
    
    with torch.no_grad():
        features = model.conv_block_1(image_tensor)
    
    features = features.squeeze(0).cpu()
    os.makedirs(os.path.join(project_root, 'outputs', 'plots'), exist_ok=True)
    fig, axs = plt.subplots(2, 5, figsize=(12, 5))
    for i in range(10):
        ax = axs[i // 5][i % 5]
        ax.imshow(features[i], cmap='gray')
        ax.axis('off')
    plt.suptitle("Feature Maps from conv_block_1")
    plt.savefig(os.path.join(project_root, 'outputs', 'plots', 'feature_maps.png'))
    plt.show()
    
    print("Feature maps visualized and saved to 'outputs/plots/feature_maps.png'.")
    print("Observation: The feature maps show various patterns learned by the first convolutional block, "
          "such as edges, textures, or shapes specific to the input image. Each map highlights different "
          "aspects of the image, indicating how the model extracts low-level features for classification.")