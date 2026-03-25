# 🚀 Vision Transformer (ViT) from Scratch on CIFAR-10

This project implements a **Vision Transformer (ViT)** from scratch using PyTorch and trains it on the CIFAR-10 dataset.

Unlike typical implementations that rely on prebuilt libraries, this project focuses on **understanding and building the architecture from the ground up**.

---

## 🧠 Overview

Vision Transformers (ViT) apply the Transformer architecture (originally designed for NLP) to image classification tasks by:

- Splitting images into **fixed-size patches**
- Converting patches into **token embeddings**
- Processing them using **self-attention mechanisms**

---

## ⚙️ Architecture

### Key Components:
- 📦 **Patch Embedding**
  - Converts image into sequence of flattened patches

- ➕ **Positional Encoding**
  - Adds spatial information to patch embeddings

- 🧠 **Transformer Encoder**
  - Multi-head self-attention  
  - Feed-forward layers  
  - Layer normalization  

- 🏷️ **Classification Head**
  - Uses CLS token for final prediction

---

## 🏋️ Training Details

- Dataset: **CIFAR-10**
- Optimizer: Adam / AdamW  
- Loss Function: Cross-Entropy Loss  
- Epochs: 200  
- Training Strategy:
  - Data augmentation  
  - Regularization  

---

## 📊 Results

### ✅ Overall Accuracy
**89.11% on CIFAR-10**

---

### 📈 Training Curves
![Training Curves](./vit_training_curves.png)

---

### 📊 Confusion Matrix
![Confusion Matrix](./vit_confusion_matrix.png)

---

### 📌 Per-Class Accuracy

| Class       | Accuracy |
|------------|----------|
| Airplane   | 91.7%    |
| Automobile | 95.0%    |
| Bird       | 86.2%    |
| Cat        | 75.8%    |
| Deer       | 87.2%    |
| Dog        | 82.5%    |
| Frog       | 93.2%    |
| Horse      | 92.7%    |
| Ship       | 93.8%    |
| Truck      | 93.0%    |

---

## 🔍 Key Observations

- Strong performance on **vehicles and structured objects**
- Lower accuracy on **animal classes (cat vs dog confusion)**  
- Indicates:
  - Similar visual features between certain classes  
  - Known challenge in CIFAR-10 classification  

---

## 💡 What I Learned

- How to implement **Transformers from scratch**
- How attention works in **vision tasks**
- Differences between **CNNs vs Transformers**
- Importance of:
  - Data augmentation  
  - Hyperparameter tuning  
  - Model analysis (not just accuracy)

---

## 🚀 Future Improvements

- Compare performance with CNN baselines (ResNet, etc.)
- Experiment with:
  - Patch sizes  
  - Number of heads  
  - Deeper architectures  
- Add attention visualization  
- Train on larger datasets (CIFAR-100, ImageNet subset)

---

## 📂 Project Structure
