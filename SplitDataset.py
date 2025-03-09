import os
import shutil
import random

# Пути к исходным изображениям и маскам
IMG_DIR = "val_coco_tv_images"
MASK_DIR = "val_coco_tv_masks"

# Пути к новым папкам для валидации
VAL_IMG_DIR = "test_coco_tv_images"
VAL_MASK_DIR = "test_coco_tv_masks"

# Доля валидационных данных
VAL_SPLIT = 0.5


# ✅ Функция перемещения файлов валидации
def move_validation_data(img_dir, mask_dir, val_img_dir, val_mask_dir, val_split=0.2):
    # Создаём папки, если их нет
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(val_mask_dir, exist_ok=True)

    # Получаем список всех изображений
    images = sorted(os.listdir(img_dir))
    masks = sorted(os.listdir(mask_dir))

    # Убеждаемся, что изображения и маски соответствуют
    assert len(images) == len(masks), "Ошибка: Количество изображений и масок не совпадает!"

    # Перемешиваем список для случайного разбиения
    random.shuffle(images)

    # Количество файлов для валидации
    val_size = int(len(images) * val_split)

    # Выбираем 20% данных для валидации
    val_images = images[:val_size]

    # ✅ Перемещаем файлы
    for img_name in val_images:
        shutil.move(os.path.join(img_dir, img_name), os.path.join(val_img_dir, img_name))
        shutil.move(os.path.join(mask_dir, img_name.replace(".jpg", ".png")),
                    os.path.join(val_mask_dir, img_name.replace(".jpg", ".png")))

    print(f"✅ Файлы перемещены! {val_size} изображений валидации.")


# ✅ Запускаем перемещение
move_validation_data(IMG_DIR, MASK_DIR, VAL_IMG_DIR, VAL_MASK_DIR, VAL_SPLIT)
