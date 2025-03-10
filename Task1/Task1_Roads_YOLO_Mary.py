# -*- coding: utf-8 -*-
""""Task1_Roads_YOLO_Mary"

# Road damage detection with YOLO
"""

from ultralytics import YOLO

import shutil
import os
import matplotlib.pyplot as plt
import torch
import numpy as np
import time
import cv2
import random
import xml.etree.ElementTree as ET
import argparse

running_in_colab = "COLAB_GPU" in os.environ

if running_in_colab:
    from google.colab import files
    from google.colab import drive
    drive.mount('/content/drive')

if running_in_colab:
    common_path = "/content/drive/MyDrive/Colab Notebooks/Task1/"  # Google Colab
else:
    common_path = "/app/Task1/"  # Docker

jp_train_ds_path = os.path.join(common_path, "Japan/train/images")
jp_test_ds_path = os.path.join(common_path, "Japan/test/images")
ch_train_ds_path = os.path.join(common_path, "China_MotorBike/train/images")
ch_test_ds_path = os.path.join(common_path, "China_MotorBike/test/images")

jp_train_ann_path = os.path.join(common_path, "Japan/train/annotations/xmls")
ch_train_ann_path = os.path.join(common_path, "China_MotorBike/train/annotations/xmls")

datasets_img = {
    "Japan Train": jp_train_ds_path,
    "Japan Test": jp_test_ds_path,
    "China Train": ch_train_ds_path,
    "China Test": ch_test_ds_path,
}

annots = {
    "Japan Train": jp_train_ann_path,
    "China Train": ch_train_ann_path,
}

class XMLtoYOLOConverter:
    """
    Конвертер XML-аннотаций в YOLO .txt файлы.
    """

    def __init__(self, annotations_dir, labels_dir, class_mapping):
        self.annotations_dir = annotations_dir
        self.labels_dir = labels_dir
        self.class_mapping = class_mapping
        os.makedirs(self.labels_dir, exist_ok=True)

    def convert(self):

        xml_files = [f for f in os.listdir(self.annotations_dir) if f.endswith('.xml')]

        for xml_file in xml_files:
            xml_path = os.path.join(self.annotations_dir, xml_file)
            txt_path = os.path.join(self.labels_dir, xml_file.replace('.xml', '.txt'))
            self.parse_xml(xml_path, txt_path)

    def parse_xml(self, xml_path, txt_path):
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            img_width = int(root.find("size/width").text)
            img_height = int(root.find("size/height").text)

            with open(txt_path, "w") as txt_file:
                for obj in root.findall("object"):
                    cls = obj.find("name").text.strip()

                    if cls not in self.class_mapping:
                        continue

                    cls_id = self.class_mapping[cls]
                    bbox = obj.find("bndbox")

                    xmin = int(bbox.find("xmin").text)
                    ymin = int(bbox.find("ymin").text)
                    xmax = int(bbox.find("xmax").text)
                    ymax = int(bbox.find("ymax").text)

                    x_center = ((xmin + xmax) / 2) / img_width
                    y_center = ((ymin + ymax) / 2) / img_height
                    width = (xmax - xmin) / img_width
                    height = (ymax - ymin) / img_height

                    txt_file.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

        except Exception as e:
            print(f"Ошибка чтения {xml_path}: {e}")

class DatasetProcessor:
    """
    Класс для обработки датасета: разбиение на train/val 80/20, создание YAML-файла.
    """

    def __init__(self, dataset_path, class_mapping, val_ratio=0.2):
        self.dataset_path = dataset_path
        self.class_mapping = class_mapping
        self.val_ratio = val_ratio

    def split_train_validation(self):
        train_image_folder = os.path.join(self.dataset_path, "train/images")
        train_label_folder = os.path.join(self.dataset_path, "train/labels")
        val_image_folder = os.path.join(self.dataset_path, "val/images")
        val_label_folder = os.path.join(self.dataset_path, "val/labels")

        os.makedirs(val_image_folder, exist_ok=True)
        os.makedirs(val_label_folder, exist_ok=True)

        images = [f for f in os.listdir(train_image_folder) if f.endswith(('.jpg', '.png'))]
        random.shuffle(images)
        val_size = int(len(images) * self.val_ratio)
        val_images = images[:val_size]
        train_images = images[val_size:]

        for img in val_images:
            shutil.move(os.path.join(train_image_folder, img), os.path.join(val_image_folder, img))
            label = img.replace('.jpg', '.txt').replace('.png', '.txt')
            if os.path.exists(os.path.join(train_label_folder, label)):
                shutil.move(os.path.join(train_label_folder, label), os.path.join(val_label_folder, label))

        print(f"Данные разделены: {len(train_images)} в Train, {len(val_images)} в Validation")

    def create_yaml(self):

        unique_classes = set(self.class_mapping.values())
        num_classes = len(unique_classes)

        if num_classes == 1:
            class_names = ["Crack"]
        else:
            class_names = sorted(set(self.class_mapping.keys()))

        yaml_path = os.path.join(self.dataset_path, "data.yaml")

        with open(yaml_path, "w", encoding="utf-8") as f:
            f.write(f"path: {self.dataset_path}\n")
            f.write(f"train: {os.path.join(self.dataset_path, 'train', 'images')}\n")
            f.write(f"val: {os.path.join(self.dataset_path, 'val', 'images')}\n")
            f.write(f"nc: {num_classes}\n")
            f.write(f"names: {class_names}\n")

class YOLOEvaluator:
    def __init__(self, model, results_path):
        self.model = model
        self.results_path = results_path
        os.makedirs(self.results_path, exist_ok=True)

    def evaluate(self):
        results = self.model.val()
        mean_ap50 = results.box.map50
        mean_ap50_95 = results.box.map

        with open(os.path.join(self.results_path, "mean_ap.txt"), "w") as f:
            f.write(f"mAP50: {mean_ap50:.4f}\n")
            f.write(f"mAP50-95: {mean_ap50_95:.4f}\n")

        print(f"mAP50: {mean_ap50:.4f}, mAP50-95: {mean_ap50_95:.4f}")

    def plot_precision_recall(self):
        results = self.model.val()
        if hasattr(results.box, "curves_results") and results.box.curves_results:
            pr_curve = results.box.curves_results[0]
            recall = pr_curve[0]
            precision = pr_curve[1].flatten()
            plt.figure(figsize=(6, 6))
            plt.plot(recall, precision, marker=".", label="Precision-Recall Curve")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.legend()
            plt.grid()
            plt.savefig(os.path.join(self.results_path, "pr_curve.png"))
            plt.show()
        else:
            print("Precision-Recall Curve не построен (пустые результаты).")

def run_pipeline(base_path, class_mapping, model_name, epochs=100, batch_size=32, img_size=640, device="cuda"):
    """
    Полный пайплайн: конвертация аннотаций, разбиение train/val, создание YAML, тренировка, оценка и предсказания
    """
    if device == "cuda" and not torch.cuda.is_available():
        if torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    xml_converter = XMLtoYOLOConverter(
        os.path.join(base_path, "train/annotations/xmls"),
        os.path.join(base_path, "train/labels"),
        class_mapping
    )
    xml_converter.convert()

    dataset_processor = DatasetProcessor(base_path, class_mapping)
    dataset_processor.split_train_validation()
    dataset_processor.create_yaml()

    model = YOLO(model_name)
    model.train(
        data=os.path.join(base_path, "data.yaml"),
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        device=device,
        project=os.path.join(base_path, "runs/train"),
        name=f"yolo_training_{os.path.basename(base_path)}"
    )

    evaluator = YOLOEvaluator(model, os.path.join(base_path, "test/results"))
    evaluator.evaluate()
    evaluator.plot_precision_recall()

    model.predict(
        source=os.path.join(base_path, "test/images"),
        save=True,
        save_txt=True,
        save_conf=True,
        project=os.path.join(base_path, "test/predictions"),
        iou=0.7
    )

china_path = os.path.join(common_path, "China_MotorBike")
japan_path = os.path.join(common_path, "Japan")

class_mapping_one = {"D00": 0, "D01": 0, "D10": 0, "D11": 0, "D20": 0}
class_mapping_multi = {
        "D00": 0, "D01": 0,
        "D10": 1, "D11": 1,
        "D20": 2,
        "D40": 3, "D43": 3, "D44": 3
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO Training Pipeline")
    parser.add_argument("--train_china", action="store_true", help="Train YOLO on China dataset")
    parser.add_argument("--train_japan", action="store_true", help="Train YOLO on Japan dataset")
    args = parser.parse_args()

    if not args.train_china and not args.train_japan:
        parser.error("Необходимо указать хотя бы один аргумент: --train_china или --train_japan")

    if args.train_china:
        print("Обучаем модель на China_MotorBike")
        run_pipeline(
            base_path=china_path,
            class_mapping=class_mapping_one,
            model_name="yolo11s.pt",
            epochs=100,
            batch_size=32,
            img_size=640
        )

    if args.train_japan:
        print("Обучаем модель на Japan")
        run_pipeline(
            base_path=japan_path,
            class_mapping=class_mapping_multi,
            model_name="yolo11s.pt",
            epochs=100,
            batch_size=32,
            img_size=640
        )

