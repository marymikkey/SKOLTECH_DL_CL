# Project Overview

This project implements a solution to the problem of road damage detection using YOLOv11 model. The model is trained to recognize different types of road cracks and damage based on the RDD2022 dataset. 
The project is structured to solve two separate tasks:

##  Features
- **Task 1 (China_MotorBike)** - **Binary classification** (crack vs. no crack).
- **Task 2 (Japan)** - **Multi-class classification** (4 types of damage).
- **Fully containerized** - Works inside **Docker**.
- **Supports GPU & CPU** execution.
- **Automatically processes annotations** (XML → YOLO format).

---

## 📌 Running the Project using DockerHub

### 1️⃣ **Pull the Prebuilt Docker Image**
```bash
docker pull marymikkey/yolo_road_damage:latest
```

### 2️⃣ **Run the Docker Container**
####  **Using GPU**
```bash
docker run --gpus all -v $(pwd)/Task1:/app/Task1 -it marymikkey/yolo_road_damage
```
####  **Using CPU**
```bash
docker run -v $(pwd)/Task1:/app/Task1 -e USE_CPU=true -it marymikkey/yolo_road_damage
```

### 3️⃣ **Train the Model Inside the Container**
You can train the model manually inside the container:
```bash
python3 Task1_Roads_YOLO_Mary.py --train_china
python3 Task1_Roads_YOLO_Mary.py --train_japan
```
OR train both models together:
```bash
python3 Task1_Roads_YOLO_Mary.py --train_china --train_japan
```

---

# Project Structure

```
Task1/
│   Task1_Roads_YOLO_Mary.ipynb
│   yolo11s.pt
│   yolo11m.pt
│
├── China_MotorBike/
│   ├── data.yaml
│   ├── train/
│   │   ├── images/
│   │   ├── labels/
│   │
│   ├── val/
│   │   ├── images/
│   │   ├── labels/
│   │
│   ├── test/
│   │   ├── images/
│   │   ├── predictions/
│
│   ├── runs/
│   │   ├── train/
│   │   ├── val/
│
├── Japan/
│   ├── data.yaml
│   ├── train/
│   │   ├── images/
│   │   ├── labels/
│   │
│   ├── val/
│   │   ├── images/
│   │   ├── labels/
│   │
│   ├── test/
│   │   ├── images/
│   │   ├── predictions/
│
│   ├── runs/
│   │   ├── train/
│   │   ├── val/
```
---

## 📦 **Python Dependencies**
The project uses the following Python packages:

```txt
torch
ultralytics
opencv-python
matplotlib
numpy
torchvision
torchaudio
PyYAML
```

---

##  **Docker Image Details**
- **Base Image:** `nvidia/cuda:12.1.1-devel-ubuntu20.04`
- **Python Version:** `3.9`
- **Supports GPU & CPU execution**

---

##  **Results & Evaluation**
After training, the model will:
- **Save trained weights** inside `/Task1/runs/train/`
- **Generate Precision-Recall Curves**
- **Compute mAP (mean Average Precision)**

For evaluation:
```bash
python3 Task1_Roads_YOLO_Mary.py --evaluate
```
---


