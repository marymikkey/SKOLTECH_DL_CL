# Project Overview

This project implements a solution to the problem of road damage detection using YOLOv11 model. The model is trained to recognize different types of road cracks and damage based on the RDD2022 dataset. 
The project is structured to solve two separate tasks:

- Task 1 (China_MotorBike) - binary classification (crack vs. no crack).

- Task 2 (Japan) - multi-class classification (4 types of damage).

The project is designed to run in a Docker container, making it easy to deploy on different systems.

# Running the Project using DockerHub

1.Pull the Docker Image

docker pull marymikkey/yolo_road_damage:latest

2.Run the Docker Container

docker run --gpus all -v $(pwd)/Task1:/app/Task1 -it marymikkey/yolo_road_damage

If using a CPU-only system:

docker run -v $(pwd)/Task1:/app/Task1 -e USE_CPU=true -it marymikkey/yolo_road_damage

If u need to train the Model inside the container, run:

python3 Task1_Roads_YOLO_Mary.py

# Project Structure
'''
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
'''

