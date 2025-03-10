import os
import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from train import val_transform


def dice_loss(preds, targets, smooth=1e-6):
    """Функция расчета Dice Loss"""
    preds = torch.sigmoid(preds)
    preds = (preds > 0.5).float()
    intersection = (preds * targets).sum()
    return 1 - (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)


class ImageFolderDataset(Dataset):
    def __init__(self, image_dir, mask_dir=None, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_filenames = [f for f in os.listdir(image_dir) if f.endswith(".jpg") or f.endswith(".png")]
        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_filename = self.image_filenames[idx]
        image_path = os.path.join(self.image_dir, image_filename)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        original_shape = image.shape[:2]  # (height, width)

        mask = None
        if self.mask_dir:
            mask_path = os.path.join(self.mask_dir, image_filename.replace(".jpg", ".png"))
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                mask = (mask / 255.0).astype(np.float32)
                mask = torch.tensor(mask).unsqueeze(0)  # Добавляем измерение канала
            else:
                mask = None

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        return image, mask if mask is not None else torch.zeros((1, 256, 256),
                                                                dtype=torch.float32), image_filename, original_shape


def generate_masks_and_evaluate(image_dir, mask_dir=None, model_path=None):
    """Генерирует маски и вычисляет Dice Loss для двух вариантов: 256x256 и оригинального размера."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = smp.Segformer(
        encoder_name="mit_b4",
        encoder_weights=None,
        in_channels=3,
        classes=1
    ).to(device)

    if model_path:
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    dataset = ImageFolderDataset(image_dir, mask_dir, transform=val_transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    dice_losses_256 = []
    dice_losses_original = []
    os.makedirs("generated_masks", exist_ok=True)

    with torch.no_grad():
        for images, masks, filenames, original_shapes in tqdm(dataloader, desc="Processing images"):
            images = images.to(device)

            preds = model(images)
            preds = torch.sigmoid(preds)
            preds = (preds > 0.5).float().cpu().numpy()[0, 0].astype(np.float32)

            # Восстанавливаем маску в оригинальный размер изображения
            original_h, original_w = map(int, original_shapes)
            resized_pred_mask = cv2.resize(preds, (original_w, original_h), interpolation=cv2.INTER_NEAREST)

            mask_path = os.path.join("generated_masks", filenames[0])
            cv2.imwrite(mask_path, (preds * 255).astype(np.uint8))

            if mask_dir and masks[0] is not None:
                # Dice loss для 256x256
                resized_mask_256 = cv2.resize(masks[0].numpy()[0], (256, 256), interpolation=cv2.INTER_NEAREST)
                dice_256 = dice_loss(torch.tensor(preds), torch.tensor(resized_mask_256))
                dice_losses_256.append(dice_256.item())

                # Dice loss для оригинального размера
                resized_mask_original = cv2.resize(masks[0].numpy()[0], (original_w, original_h),
                                                   interpolation=cv2.INTER_NEAREST)
                dice_original = dice_loss(torch.tensor(resized_pred_mask), torch.tensor(resized_mask_original))
                dice_losses_original.append(dice_original.item())

    if dice_losses_256 and dice_losses_original:
        avg_dice_loss_256 = np.mean(dice_losses_256)
        avg_dice_loss_original = np.mean(dice_losses_original)
        print(f"Средний Dice Loss (256x256): {avg_dice_loss_256:.4f}")
        print(f"Средний Dice Loss (Оригинальный размер): {avg_dice_loss_original:.4f}")
    else:
        print("Dice Loss не вычислен, так как нет масок для сравнения.")

# Пример использования
generate_masks_and_evaluate("MostImportant", None, "checkpoints/segformer_mit_b4_epoch20.pth")
