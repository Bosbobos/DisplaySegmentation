import os
import torch
import torch.nn as nn
import torch.optim as optim
import albumentations as A
import numpy as np
import cv2
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import glob

# ===================== CONFIG =====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_IMG_DIR = "coco_tv_images"
TRAIN_MASK_DIR = "coco_tv_masks"
VAL_IMG_DIR = "val_coco_tv_images"
VAL_MASK_DIR = "val_coco_tv_masks"
BATCH_SIZE = 8
LR = 1e-3
NUM_EPOCHS = 50
IMAGE_SIZE = 256
NUM_WORKERS = 6
PIN_MEMORY = True
MIXED_PRECISION = True
MODEL_NAME = "unet_resnet34"
SAVE_MODEL = True
CHECKPOINT_DIR = "checkpoints"  # Папка для сохранения чекпоинтов

# Создаём папку для чекпоинтов
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ===================== DATASET =====================
class ScreenSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_filenames = os.listdir(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name.replace(".jpg", ".png"))

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        mask = np.where(mask > 128, 1, 0).astype(np.uint8)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]

        return image, mask

# ===================== AUGMENTATIONS =====================
train_transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.GaussNoise(p=0.2),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# ===================== TRAIN FUNCTION =====================
def train_fn(train_loader, model, optimizer, loss_fn, scaler):
    model.train()
    total_loss = 0
    for batch_idx, (images, masks) in enumerate(train_loader):
        images, masks = images.to(DEVICE), masks.to(DEVICE).float().unsqueeze(1)

        with torch.cuda.amp.autocast(enabled=MIXED_PRECISION):
            preds = model(images)
            loss = loss_fn(preds, masks)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        if batch_idx % 10 == 0:
            print(f"Batch [{batch_idx}/{len(train_loader)}] - Loss: {loss.item():.4f}")

    return total_loss / len(train_loader)

# ===================== VALIDATION FUNCTION =====================
def validate(val_loader, model, loss_fn):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE).float().unsqueeze(1)

            preds = model(images)
            loss = loss_fn(preds, masks)

            total_loss += loss.item()

    return total_loss / len(val_loader)

# ===================== LOAD LAST CHECKPOINT =====================
def load_checkpoint(model, optimizer):
    checkpoints = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, f"{MODEL_NAME}_epoch*.pth")))
    if checkpoints:
        latest_checkpoint = checkpoints[-1]
        checkpoint_epoch = int(latest_checkpoint.split("_epoch")[1].split(".pth")[0])
        print(f"📥 Загружаем чекпоинт: {latest_checkpoint}")
        model.load_state_dict(torch.load(latest_checkpoint, map_location=DEVICE))
        return checkpoint_epoch
    return 0  # Если чекпоинтов нет, начинаем с 0

# ===================== MAIN =====================
if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)

    # Загружаем датасеты
    train_dataset = ScreenSegmentationDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, transform=train_transform)
    val_dataset = ScreenSegmentationDataset(VAL_IMG_DIR, VAL_MASK_DIR, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    print(f"✅ Датасет загружен: {len(train_dataset)} train, {len(val_dataset)} val")

    # Загружаем модель
    model = smp.Unet("resnet34", encoder_weights="imagenet", in_channels=3, classes=1).to(DEVICE)

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    scaler = torch.cuda.amp.GradScaler(enabled=MIXED_PRECISION)

    # Проверяем, есть ли чекпоинт
    start_epoch = load_checkpoint(model, optimizer)

    # Обучение модели
    for epoch in range(start_epoch, NUM_EPOCHS):
        train_loss = train_fn(train_loader, model, optimizer, loss_fn, scaler)
        val_loss = validate(val_loader, model, loss_fn)

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

        if SAVE_MODEL and (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{MODEL_NAME}_epoch{epoch+1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"📌 Чекпоинт сохранён: {checkpoint_path}")

    print("🎉 Обучение завершено!")

    # ===================== ПРОВЕРКА КАЧЕСТВА НА ВАЛИДАЦИИ =====================
    model.eval()
    img, _ = val_dataset[0]
    img_tensor = img.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred_mask = torch.sigmoid(model(img_tensor))
        pred_mask = (pred_mask > 0.5).float()

    plt.subplot(1, 2, 1)
    plt.imshow(img.permute(1, 2, 0).cpu().numpy())
    plt.subplot(1, 2, 2)
    plt.imshow(pred_mask.squeeze().cpu().numpy(), cmap="gray")
    plt.show()
