# Pavement Crack Detection using Semantic Segmentation (U-Net++)

## Project Overview
This project involves semantic segmentation of pavement cracks using a U-Net++ neural network architecture. 
The goal is to accurately detect and segment cracks of varying sizes and shapes on road surfaces.

## ✨ Features

-  **Semantic segmentation** of road cracks (binary: crack vs. background)
-  **UNet++ architecture** using `segmentation-models-pytorch`
-  Trained on **CRACK500** dataset
-  Supports evaluation with ROC / PR curves, loss tracking, and penta diagrams
-  **Fully containerized** with Docker
- Works on **GPU & CPU**
-  Organized structure with automatic data prep and training pipeline

---

## Dataset
- **Crack500 Dataset**
- [Paper and dataset link](https://arxiv.org/abs/1901.06340)

Dataset consists of:
- **Training:** 1896 images
- **Validation:** 348 images
- **Test:** 1124 images

## Model Architecture
- **U-Net++**
- Encoder backbone: **ResNet34** (pre-trained on ImageNet)

U-Net++ includes nested skip connections and dense blocks which improve the segmentation quality, particularly for thin and detailed structures like cracks.

## Loss Functions
Evaluated loss functions include:
- **Dice Loss** (baseline)
- **Tversky Loss**
- **Combined Loss (Weighted Cross-Entropy + Dice Loss)**

The **baseline Dice Loss** provided the best overall performance.

## Metrics
Evaluation metrics used:
- Accuracy
- Precision
- Recall
- F1-score
- IoU (Intersection over Union)

## Best Model Results (Dice Loss)

- **Best Epoch:** 46
- **IoU:** 0.6664
- **Accuracy:** 97.83%
- **Precision:** 81.38%
- **Recall:** 80.47%
- **F1-score:** 80.92%

The Dice Loss model showed stable convergence and effectively managed class imbalance, making it the recommended choice for pavement crack segmentation tasks with limited and imbalanced data.

## Folder Structure
```
Task2/
├── data/
│   ├── train/
│   │   ├── images/
│   │   └── masks/
│   ├── val/
│   │   ├── images/
│   │   └── masks/
│   └── test/
│       ├── images/
│       └── masks/
├── results/
│   ├── models/          # Saved model weights
│   ├── plots/           # Metrics and evaluation plots
│   └── predictions/     # Predictions on the test set
├── Pavement_Crack_Detection_SemSegm_Mary.ipynb # Main notebook
└── README.md
```

---

## 📌 Running the Project using DockerHub

### 1️⃣ Pull the Prebuilt Docker Image

```bash
docker pull marymikkey/crack_semsegm_unetpp:latest
```

### 2️⃣ Run the Docker Container

#### Using GPU:

```bash
docker run --gpus all -v $(pwd)/Task2:/app/Task2 -it marymikkey/crack_semsegm_unetpp
```

#### Using CPU:

```bash
docker run -v $(pwd)/Task2:/app/Task2 -e USE_CPU=true -it marymikkey/crack_semsegm_unetpp
```

### 3️⃣ Train & Evaluate the Model

Inside the container, run:

```bash
python3 Pavement_Crack_Detection_SemSegm_Mary.py --loss [baseline|tversky|combined] --epochs [NUM_EPOCHS]
```

**Examples:**

Train with **Dice Loss** for 50 epochs:
```bash
python3 Pavement_Crack_Detection_SemSegm_Mary.py --loss baseline --epochs 50
```

Train with **Tversky Loss** for 60 epochs:
```bash
python3 Pavement_Crack_Detection_SemSegm_Mary.py --loss tversky --epochs 60
```

Train with **Combined Loss**:
```bash
python3 Pavement_Crack_Detection_SemSegm_Mary.py --loss combined --epochs 50
```

---

## 🧠 Python Dependencies

```txt
torch
segmentation-models-pytorch
opencv-python
matplotlib
numpy
scikit-learn
tqdm
```

---

## 🐳 Docker Image Details

- **Base Image**: `nvidia/cuda:12.1.1-devel-ubuntu20.04`  
- **Python Version**: 3.9  
- Supports **GPU & CPU**  
- Lightweight and ready-to-run with pre-installed dependencies  

---


