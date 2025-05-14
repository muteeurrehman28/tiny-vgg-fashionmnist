# Tiny-VGG on FashionMNIST

A deep learning project implementing the Tiny-VGG convolutional neural network to classify images from the FashionMNIST dataset. The project demonstrates model training, evaluation, and visualization of convolutional feature maps using PyTorch.

## Overview

This repository implements a Tiny-VGG model for classifying 28x28 grayscale images from the FashionMNIST dataset into 10 clothing categories. The project includes training the model, evaluating its performance with a confusion matrix, and visualizing feature maps from the first convolutional block. All results, including loss/accuracy plots and the trained model, are saved for analysis.

## Project Structure

- `data/`: Stores the FashionMNIST dataset (automatically downloaded by PyTorch).
- `models/tiny_vgg.py`: Defines the TinyVGG model architecture.
- `scripts/`:
  - `main.py`: Orchestrates dataset loading, model training, evaluation, and visualization.
  - `train.py`: Contains training and testing functions.
  - `evaluate.py`: Evaluates the model and generates a confusion matrix.
  - `visualize.py`: Visualizes feature maps from the first convolutional block.
- `utils/helpers.py`: Utility functions for plotting loss and accuracy curves.
- `outputs/`:
  - `models/`: Stores trained model checkpoints.
  - `plots/`: Stores generated plots (loss/accuracy curves, confusion matrix, feature maps).
- `requirements.txt`: Lists Python dependencies.
- `.gitignore`: Excludes data, outputs, and temporary files from version control.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/tiny-vgg-fashionmnist.git
   cd tiny-vgg-fashionmnist
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Ensure a CUDA-enabled GPU is available for faster training (optional; CPU is supported).

## Usage

Run the main script to execute the full workflow:
```bash
python scripts/main.py
```

The script will:
- Download and load the FashionMNIST dataset.
- Train the TinyVGG model for 15 epochs using SGD (learning rate 0.01) and CrossEntropyLoss.
- Save the trained model to `outputs/models/tiny_vgg.pth`.
- Generate and save loss/accuracy plots to `outputs/plots/`.
- Evaluate the model and save a confusion matrix to `outputs/plots/confusion_matrix.png`.
- Visualize feature maps for a test image, saved to `outputs/plots/feature_maps.png`.

## Results

- **Training Performance**: Achieves ~91% accuracy after 15 epochs.
- **Test Performance**: Achieves ~89-90% accuracy, indicating strong generalization.
- **Feature Maps**: Visualizations reveal low-level features (edges, textures, shapes) extracted by the first convolutional block.
- **Outputs**: All plots and the model checkpoint are saved in the `outputs/` directory for further analysis.

## Notes

- Ensure write permissions for the `data/` directory to download the FashionMNIST dataset.
- The project is compatible with both CPU and GPU environments (CUDA recommended for performance).
- To extend the project, modify `scripts/main.py` to load the saved model or adjust hyperparameters.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.