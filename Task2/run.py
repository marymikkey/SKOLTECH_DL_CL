import argparse
import os
import copy
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import segmentation_models_pytorch as smp

def parse_args():
    parser = argparse.ArgumentParser(description="Обучение модели сегментации трещин")
    parser.add_argument("--epochs", type=int, default=50, help="Количество эпох обучения")
    parser.add_argument("--loss_variant", type=str, choices=["baseline", "tversky", "combined"], default="baseline", help="Вариант лосс-функции")
    return parser.parse_args()

args = parse_args()

common_path = "/app/Task2/"  # для Docker; если в Colab, можно заменить на другой путь
dataset_path = os.path.join(common_path, "CRACK500")
dest_base = os.path.join(common_path, "data")
train_images_dest = os.path.join(dest_base, "train", "images")
train_masks_dest  = os.path.join(dest_base, "train", "masks")
val_images_dest   = os.path.join(dest_base, "val", "images")
val_masks_dest    = os.path.join(dest_base, "val", "masks")
test_images_dest  = os.path.join(dest_base, "test", "images")
test_masks_dest   = os.path.join(dest_base, "test", "masks")

results_dir = os.path.join(common_path, "results")
models_dir = os.path.join(results_dir, "models")
plots_dir  = os.path.join(results_dir, "plots")
os.makedirs(models_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

class CrackDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None, target_size=(512, 512)):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_files = sorted(os.listdir(images_dir))
        self.mask_files  = sorted(os.listdir(masks_dir))
        self.transform = transform
        self.target_size = target_size 
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        mask_path = os.path.join(self.masks_dir, self.mask_files[idx])
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        image = cv2.resize(image, self.target_size)
        mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)
        
        if 255 in np.unique(mask):
            mask = mask // 255
        
        image = np.transpose(image, (2, 0, 1)).astype(np.float32) / 255.0
        
        image_tensor = torch.from_numpy(image).float()
        mask_tensor = torch.from_numpy(mask).long()
        return image_tensor, mask_tensor

class UNetPlusPlusModel(nn.Module):
    def __init__(self, encoder_name="resnet34", encoder_weights="imagenet", classes=2):
        super(UNetPlusPlusModel, self).__init__()
        self.model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=classes
        )
        
    def forward(self, x):
        return self.model(x)

class Trainer:
    def __init__(self, model, device, train_loader, val_loader, optimizer, loss_variant="baseline"):
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.loss_variant = loss_variant
        self.loss_fn = self.get_loss_function(loss_variant)
    
    def get_loss_function(self, loss_variant):
        if loss_variant == "baseline":
            return smp.losses.DiceLoss(mode='multiclass')
        elif loss_variant == "tversky":
            return smp.losses.TverskyLoss(mode='multiclass', alpha=0.7, beta=0.3)
        elif loss_variant == "combined":
            class_weights = torch.tensor([0.53, 7.89], dtype=torch.float32).to(self.device)
            weighted_ce = nn.CrossEntropyLoss(weight=class_weights)
            dice_loss = smp.losses.DiceLoss(mode='multiclass')
            class CombinedLoss(nn.Module):
                def __init__(self, ce_loss, dice_loss, alpha=0.5):
                    super(CombinedLoss, self).__init__()
                    self.ce_loss = ce_loss
                    self.dice_loss = dice_loss
                    self.alpha = alpha
                def forward(self, outputs, targets):
                    return self.alpha * self.ce_loss(outputs, targets) + (1 - self.alpha) * self.dice_loss(outputs, targets)
            return CombinedLoss(weighted_ce, dice_loss, alpha=0.5)
        else:
            raise ValueError("Неизвестный вариант лосс-функции")
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        for images, masks in tqdm(self.train_loader, desc="Training"):
            images = images.to(self.device)
            masks = masks.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.loss_fn(outputs, masks)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(self.train_loader)
    
    def validate(self):
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_trues = []
        with torch.no_grad():
            for images, masks in tqdm(self.val_loader, desc="Validation"):
                images = images.to(self.device)
                masks = masks.to(self.device)
                outputs = self.model(images)
                loss = self.loss_fn(outputs, masks)
                total_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                all_preds.append(preds.cpu().numpy())
                all_trues.append(masks.cpu().numpy())
        return total_loss / len(self.val_loader), np.concatenate(all_trues, axis=0), np.concatenate(all_preds, axis=0)

class Evaluator:
    def __init__(self, plots_dir):
        self.plots_dir = plots_dir
        os.makedirs(self.plots_dir, exist_ok=True)
    
    def plot_loss_curve(self, train_losses, val_losses, filename="loss_curve.png"):
        plt.figure(figsize=(8,6))
        epochs_range = range(1, len(train_losses)+1)
        plt.plot(epochs_range, train_losses, label="Train Loss")
        plt.plot(epochs_range, val_losses, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss Curve")
        plt.legend()
        save_path = os.path.join(self.plots_dir, filename)
        plt.savefig(save_path)
        plt.close()
        print("Loss curve saved to:", save_path)
    
    def plot_lr_curve(self, lrs, filename="lr_curve.png"):
        plt.figure(figsize=(8,6))
        epochs_range = range(1, len(lrs)+1)
        plt.plot(epochs_range, lrs, label="Learning Rate")
        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate")
        plt.title("Learning Rate Schedule")
        plt.legend()
        save_path = os.path.join(self.plots_dir, filename)
        plt.savefig(save_path)
        plt.close()
        print("Learning rate curve saved to:", save_path)
    
    def plot_roc_curve(self, true_flat, prob_flat, filename="roc_curve.png"):
        fpr, tpr, _ = roc_curve(true_flat, prob_flat)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(8,6))
        plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.2f})')
        plt.plot([0,1], [0,1], 'r--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        save_path = os.path.join(self.plots_dir, filename)
        plt.savefig(save_path)
        plt.close()
        print("ROC curve saved to:", save_path)
    
    def plot_precision_recall_curve(self, true_flat, prob_flat, filename="precision_recall_curve.png"):
        precision, recall, _ = precision_recall_curve(true_flat, prob_flat)
        plt.figure(figsize=(8,6))
        plt.plot(recall, precision, label="Precision-Recall")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend()
        save_path = os.path.join(self.plots_dir, filename)
        plt.savefig(save_path)
        plt.close()
        print("Precision-Recall curve saved to:", save_path)
    
    def plot_penta_diagram(self, metrics, filename="penta_diagram.png"):
        labels = list(metrics.keys())
        values = list(metrics.values())
        num_vars = len(labels)
        angles = np.linspace(0, 2*np.pi, num_vars, endpoint=False).tolist()
        values += values[:1]
        angles += angles[:1]
        fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
        ax.plot(angles, values, linewidth=2, linestyle='solid', label="Metrics")
        ax.fill(angles, values, alpha=0.25)
        ax.set_thetagrids(np.degrees(angles[:-1]), labels)
        ax.set_title("Penta Diagram")
        ax.grid(True)
        save_path = os.path.join(self.plots_dir, filename)
        plt.savefig(save_path)
        plt.close()
        print("Penta Diagram saved to:", save_path)

def calculate_metrics(true_masks, pred_masks):
    true_flat = true_masks.flatten()
    pred_flat = pred_masks.flatten()
    TP = np.sum((true_flat == 1) & (pred_flat == 1))
    TN = np.sum((true_flat == 0) & (pred_flat == 0))
    FP = np.sum((true_flat == 0) & (pred_flat == 1))
    FN = np.sum((true_flat == 1) & (pred_flat == 0))
    accuracy  = (TP + TN) / (TP + TN + FP + FN + 1e-6)
    precision = TP / (TP + FP + 1e-6)
    recall    = TP / (TP + FN + 1e-6)
    f1        = 2 * precision * recall / (precision + recall + 1e-6)
    iou       = TP / (TP + FP + FN + 1e-6)
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "iou": iou}

class Predictor:
    def __init__(self, model, device, predictions_dir, target_size=(512,512), threshold=0.5):
        self.model = model
        self.device = device
        self.predictions_dir = predictions_dir
        os.makedirs(self.predictions_dir, exist_ok=True)
        self.target_size = target_size
        self.threshold = threshold
        
    def predict_and_save(self, images_dir):
        test_img_files = sorted(os.listdir(images_dir))
        for img_file in test_img_files:
            img_path = os.path.join(images_dir, img_file)
            image = cv2.imread(img_path)
            if image is None:
                print("Не удалось прочитать:", img_path)
                continue
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            resized_img = cv2.resize(image_rgb, self.target_size)
            input_tensor = torch.from_numpy(np.transpose(resized_img, (2, 0, 1))).float().unsqueeze(0).to(self.device) / 255.0
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probs = torch.softmax(outputs, dim=1)[:, 1, :, :].squeeze().cpu().numpy()
                pred_mask = (probs > self.threshold).astype(np.uint8)
            contours, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                conf = np.mean(probs[y:y+h, x:x+w])
                cv2.rectangle(resized_img, (x,y), (x+w, y+h), (255,0,0), 2)
                cv2.putText(resized_img, f"{conf:.2f}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
            result_bgr = cv2.cvtColor(resized_img, cv2.COLOR_RGB2BGR)
            save_path = os.path.join(self.predictions_dir, img_file)
            cv2.imwrite(save_path, result_bgr)
            print(f"[{self.threshold}] Сохранено: {save_path}")
    
    def evaluate_mean_iou(self, images_dir, masks_dir):
        iou_sum = 0.0
        count = 0
        test_files = sorted(os.listdir(images_dir))
        for img_file in test_files:
            img_path = os.path.join(images_dir, img_file)
            mask_path = os.path.join(masks_dir, os.path.splitext(img_file)[0] + ".png")
            if not os.path.exists(mask_path):
                continue
            image = cv2.imread(img_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            resized_img = cv2.resize(image_rgb, self.target_size)
            input_tensor = torch.from_numpy(np.transpose(resized_img, (2, 0, 1))).float().unsqueeze(0).to(self.device) / 255.0
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probs = torch.softmax(outputs, dim=1)[:, 1, :, :].squeeze().cpu().numpy()
                pred_mask = (probs > self.threshold).astype(np.uint8)
            true_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            true_mask = cv2.resize(true_mask, self.target_size, interpolation=cv2.INTER_NEAREST)
            if 255 in np.unique(true_mask):
                true_mask = true_mask // 255
            true_flat = true_mask.flatten()
            pred_flat = pred_mask.flatten()
            TP = np.sum((true_flat == 1) & (pred_flat == 1))
            FP = np.sum((true_flat == 0) & (pred_flat == 1))
            FN = np.sum((true_flat == 1) & (pred_flat == 0))
            iou = TP / (TP + FP + FN + 1e-6)
            iou_sum += iou
            count += 1
        mean_iou = iou_sum / count if count > 0 else 0
        return mean_iou

def main():
    train_dataset = CrackDataset(train_images_dest, train_masks_dest)
    val_dataset = CrackDataset(val_images_dest, val_masks_dest)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("Используем устройство:", device)
    
    model = UNetPlusPlusModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    trainer = Trainer(model, device, train_loader, val_loader, optimizer, args.loss_variant)
    
    best_iou = 0
    best_state = None
    train_losses = []
    val_losses = []
    lrs = []
    for epoch in range(1, args.epochs+1):
        train_loss = trainer.train_epoch()
        val_loss, true_masks, pred_masks = trainer.validate()
        metrics = calculate_metrics(true_masks, pred_masks)
        current_lr = optimizer.param_groups[0]['lr']
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        lrs.append(current_lr)
        print(f"Epoch {epoch}/{args.epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - LR: {current_lr:.6f}")
        print("Validation Metrics:", metrics)
        if metrics["iou"] > best_iou:
            best_iou = metrics["iou"]
            best_state = copy.deepcopy(model.state_dict())
        scheduler.step()
    
    if best_state is not None:
        final_model_path = os.path.join(models_dir, f"best_model_{args.loss_variant}.pth")
        torch.save(best_state, final_model_path)
        print("Лучшее состояние модели сохранено по пути:", final_model_path)
    
    model.load_state_dict(torch.load(final_model_path))
    val_loss, true_masks, pred_masks = trainer.validate()
    final_metrics = calculate_metrics(true_masks, pred_masks)
    print("Final metrics on validation set:", final_metrics)
    
    evaluator = Evaluator(plots_dir)
    evaluator.plot_loss_curve(train_losses, val_losses, f"loss_curve_{args.loss_variant}.png")
    evaluator.plot_lr_curve(lrs, f"lr_curve_{args.loss_variant}.png")
    
    all_probs = []
    all_trues = []
    model.eval()
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1, :, :].cpu().numpy()
            all_probs.append(probs.flatten())
            all_trues.append(masks.numpy().flatten())
    all_probs = np.concatenate(all_probs)
    all_trues = np.concatenate(all_trues)
    evaluator.plot_roc_curve(all_trues, all_probs, f"roc_curve_{args.loss_variant}.png")
    evaluator.plot_precision_recall_curve(all_trues, all_probs, f"precision_recall_curve_{args.loss_variant}.png")
    evaluator.plot_penta_diagram(final_metrics, f"penta_diagram_{args.loss_variant}.png")
    
    test_predictions_dir = os.path.join(os.path.dirname(test_images_dest), "predictions", args.loss_variant)
    os.makedirs(test_predictions_dir, exist_ok=True)
    predictor = Predictor(model, device, test_predictions_dir, target_size=(512,512), threshold=0.5)
    predictor.predict_and_save(test_images_dest)
    
    if os.path.exists(test_masks_dest):
        mean_iou_test = predictor.evaluate_mean_iou(test_images_dest, test_masks_dest)
        mean_iou_path = os.path.join(test_predictions_dir, "mean_iou.txt")
        with open(mean_iou_path, "w") as f:
            f.write(f"Mean IoU on test set: {mean_iou_test:.4f}\n")
        print("Mean IoU сохранено по пути:", mean_iou_path)

if __name__ == "__main__":
    main()