import os
import glob
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

##################################
# CONFIGURATION
##################################
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_IMG_DIR = "coco_tv_images"
TRAIN_MASK_DIR = "coco_tv_masks"
VAL_IMG_DIR = "val_coco_tv_images"
VAL_MASK_DIR = "val_coco_tv_masks"

BATCH_SIZE = 16
LEARNING_RATE = 5e-5  # Трансформеры требуют меньший LR
NUM_EPOCHS = 50
IMAGE_SIZE = (256, 256)  # (ширина, высота)
NUM_WORKERS = 6
PIN_MEMORY = True
MIXED_PRECISION = True

MODEL_NAME = "segformer_mit_b4"
SAVE_MODEL = True
CHECKPOINT_DIR = "checkpoints"
OUTPUT_PREDICTIONS_DIR = "training_predictions"
SAVE_PREDICTIONS_EVERY = 5  # Сохранять примеры каждые N эпох

# Создание необходимых директорий
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(OUTPUT_PREDICTIONS_DIR, exist_ok=True)


##################################
# DATASET
##################################
class ScreenSegmentationDataset(Dataset):
    """Dataset для сегментации изображений."""

    def __init__(self, image_dir: str, mask_dir: str, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_filenames = os.listdir(image_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_filenames)

    def __getitem__(self, idx: int):
        image_filename = self.image_filenames[idx]
        image_path = os.path.join(self.image_dir, image_filename)
        mask_filename = image_filename.replace(".jpg", ".png")
        mask_path = os.path.join(self.mask_dir, mask_filename)

        # Чтение изображения и маски
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Изображение не найдено: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Маска не найдена: {mask_path}")
        mask = (mask / 255.0).astype(np.float32)  # Нормализация в 0-1 (SegFormer требует float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]

        return image, mask, image_filename


##################################
# AUGMENTATIONS
##################################
train_transform = A.Compose([
    A.Resize(IMAGE_SIZE[1], IMAGE_SIZE[0]),  # Приведение к стандартному размеру
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Affine(scale=(0.8, 1.2), translate_percent=(0.1, 0.1), rotate=(-20, 20), p=0.5),  # Исправлено
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(IMAGE_SIZE[1], IMAGE_SIZE[0]),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])


##################################
# TRAINING FUNCTION
##################################
def train_one_epoch(train_loader, model, optimizer, loss_fn, scaler) -> float:
    model.train()
    total_loss = 0.0

    for batch_idx, (images, masks, _) in enumerate(tqdm(train_loader, desc="Training", leave=False)):
        images = images.to(DEVICE)
        masks = masks.to(DEVICE).float().unsqueeze(1)

        with torch.cuda.amp.autocast(enabled=MIXED_PRECISION):
            predictions = model(images)
            loss = loss_fn(predictions, masks)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(train_loader)


##################################
# VALIDATION FUNCTION
##################################
def validate_model(val_loader, model, loss_fn, current_epoch) -> float:
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch_idx, (images, masks, filenames) in enumerate(tqdm(val_loader, desc="Validation", leave=False)):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE).float().unsqueeze(1)

            predictions = model(images)
            loss = loss_fn(predictions, masks)
            total_loss += loss.item()

    return total_loss / len(val_loader)


##################################
# LOSS PLOTTING
##################################
def plot_losses(train_losses: list, val_losses: list) -> None:
    """Строит и сохраняет график значений потерь за эпохи."""
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss", color="blue")
    plt.plot(val_losses, label="Val Loss", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.title("Training vs Validation Loss")
    plt.savefig("training_loss.png")  # Сохранение графика
    plt.pause(0.1)
    plt.close()



##################################
# MAIN TRAINING LOOP
##################################
def main():
    torch.multiprocessing.set_start_method("spawn", force=True)

    train_dataset = ScreenSegmentationDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, transform=train_transform)
    val_dataset = ScreenSegmentationDataset(VAL_IMG_DIR, VAL_MASK_DIR, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    print(f"✅ Датасет загружен: {len(train_dataset)} train, {len(val_dataset)} val")

    model = smp.Segformer(
        encoder_name="mit_b4",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        decoder_dropout=0.3
    ).to(DEVICE)

    loss_function = smp.losses.TverskyLoss(mode="binary", alpha=0.7, beta=0.3, from_logits=True)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-3)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-4, steps_per_epoch=len(train_loader), epochs=NUM_EPOCHS
    )

    scaler = torch.cuda.amp.GradScaler(enabled=MIXED_PRECISION)

    # ✅ Добавляем списки для хранения ошибок
    train_losses = []
    val_losses = []

    plt.ion()  # Включаем интерактивный режим

    for epoch in range(NUM_EPOCHS):
        print(f"\n🔄 Эпоха {epoch + 1}/{NUM_EPOCHS}")
        train_loss = train_one_epoch(train_loader, model, optimizer, loss_function, scaler)
        val_loss = validate_model(val_loader, model, loss_function, epoch)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}] - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

        scheduler.step()
        plot_losses(train_losses, val_losses)  # ✅ Построение графика

        if SAVE_MODEL and (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"{CHECKPOINT_DIR}/{MODEL_NAME}_epoch{epoch+1}.pth")

    plt.ioff()
    print("🎉 Обучение завершено!")



if __name__ == "__main__":
    main()
