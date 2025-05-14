# 👟 Tiny-VGG on FashionMNIST — CNNs Made Simple & Powerful

A powerful yet beginner-friendly deep learning project that implements a **Tiny-VGG** convolutional neural network to classify fashion items from the **FashionMNIST** dataset. Built with **PyTorch**, this project showcases model training, performance evaluation, and feature map visualization — all in one place.

---

## 🔍 Project Highlights

* ✅ Classifies **28x28 grayscale images** into **10 clothing categories**
* 🔁 Trains a custom **Tiny-VGG architecture**
* 📊 Visualizes **loss/accuracy trends** and **confusion matrix**
* 🔍 Inspects **feature maps** from convolutional layers
* 📀 Saves all outputs (plots, models) for easy analysis
* 🚀 Compatible with **CPU & GPU** (CUDA)

---

## 📁 Directory Structure

```plaintext
tiny-vgg-fashionmnist/
|
├── data/                     # Auto-downloaded FashionMNIST dataset
├── models/
|   └── tiny_vgg.py           # TinyVGG model architecture
├── scripts/
|   ├── main.py               # Full workflow (training, eval, visualization)
|   ├── train.py              # Training and validation logic
|   ├── evaluate.py           # Test evaluation and confusion matrix
|   └── visualize.py          # Feature map visualization
├── utils/
|   └── helpers.py            # Plotting utilities
├── outputs/
|   ├── models/               # Saved trained models
|   └── plots/                # Accuracy/loss curves, confusion matrix, features
├── requirements.txt          # Python dependencies
└── .gitignore                # Ignored files and directories
```

---

## 🚀 Installation Guide

**1. Clone the Repository**

```bash
git clone https://github.com/your-username/tiny-vgg-fashionmnist.git
cd tiny-vgg-fashionmnist
```

**2. (Optional) Create a Virtual Environment**

```bash
python -m venv venv
source venv/bin/activate  # For Windows: venv\Scripts\activate
```

**3. Install Required Dependencies**

```bash
pip install -r requirements.txt
```

> 💡 *Tip:* Enable a CUDA-capable GPU for significantly faster training.

---

## ⚙️ How to Use

Simply run the main pipeline:

```bash
python scripts/main.py
```

### This will:

* 📅 Download and preprocess the **FashionMNIST** dataset
* 🧠 Train the **TinyVGG** model for 15 epochs with **SGD (lr=0.01)** and **CrossEntropyLoss**
* 📀 Save the trained model to `outputs/models/tiny_vgg.pth`
* 📈 Generate and save **loss/accuracy** plots
* 📉 Evaluate model performance with a **confusion matrix**
* 🧠 Visualize **feature maps** from the first conv block

---

## 📊 Results

| Metric                | Value    |
| --------------------- | -------- |
| **Training Accuracy** | \~91%    |
| **Test Accuracy**     | \~89-90% |

### 🔬 Feature Maps:

Visualized feature maps show how TinyVGG learns to detect **edges**, **textures**, and **basic patterns**, helping the model understand clothing items more effectively.

All results (plots, models, images) are saved in the `outputs/` folder for review and reporting.

---

## 💡 Customization & Tips

* Modify hyperparameters in `scripts/main.py` (e.g., learning rate, batch size, optimizer).
* Load a pre-trained model for inference or fine-tuning.
* Add new visualization techniques in `visualize.py` to explore deeper layers.

---

## 🛠️ Requirements

All dependencies are listed in `requirements.txt`. Install using:

```bash
pip install -r requirements.txt
```

---

## 📅 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.
