import os
import shutil
import random

# Пути к исходным изображениям и маскам
IMG_DIR = "valid_images"
MASK_DIR = "valid_masks"

# Пути к папкам для разделения выборки
TRAIN_IMG_DIR = "train_images"
TRAIN_MASK_DIR = "train_masks"
VAL_IMG_DIR = "val_images"
VAL_MASK_DIR = "val_masks"
TEST_IMG_DIR = "test_images"
TEST_MASK_DIR = "test_masks"

# Доли данных
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1  # 10% на валидацию
TEST_SPLIT = 0.1  # 10% на тестирование


# Функция разбиения данных
def split_dataset(img_dir, mask_dir, train_img_dir, train_mask_dir, val_img_dir, val_mask_dir, test_img_dir,
                  test_mask_dir):
    # Создаём папки, если их нет
    for folder in [train_img_dir, train_mask_dir, val_img_dir, val_mask_dir, test_img_dir, test_mask_dir]:
        os.makedirs(folder, exist_ok=True)

    # Получаем список всех изображений
    images = sorted(os.listdir(img_dir))
    masks = sorted(os.listdir(mask_dir))

    # Убеждаемся, что изображения и маски соответствуют
    assert len(images) == len(masks), "Ошибка: Количество изображений и масок не совпадает!"

    # Перемешиваем список для случайного разбиения
    random.shuffle(images)

    # Определяем размеры выборок
    total_size = len(images)
    train_size = int(total_size * TRAIN_SPLIT)
    val_size = int(total_size * VAL_SPLIT)

    train_images = images[:train_size]
    val_images = images[train_size:train_size + val_size]
    test_images = images[train_size + val_size:]  # Остаток уходит в тест

    # Функция перемещения файлов
    def move_files(file_list, src_img_dir, src_mask_dir, dst_img_dir, dst_mask_dir):
        for img_name in file_list:
            mask_name = img_name.replace(".jpg", ".png")  # Предполагаем, что маски в PNG
            shutil.move(os.path.join(src_img_dir, img_name), os.path.join(dst_img_dir, img_name))
            shutil.move(os.path.join(src_mask_dir, mask_name), os.path.join(dst_mask_dir, mask_name))

    # Перемещаем файлы
    move_files(train_images, img_dir, mask_dir, train_img_dir, train_mask_dir)
    move_files(val_images, img_dir, mask_dir, val_img_dir, val_mask_dir)
    move_files(test_images, img_dir, mask_dir, test_img_dir, test_mask_dir)

    print(f"Разбиение завершено!")
    print(f"{train_size} изображений в тренировочной выборке")
    print(f"{val_size} изображений в валидационной выборке")
    print(f"{len(test_images)} изображений в тестовой выборке")


# Запускаем разбиение
split_dataset(IMG_DIR, MASK_DIR, TRAIN_IMG_DIR, TRAIN_MASK_DIR, VAL_IMG_DIR, VAL_MASK_DIR, TEST_IMG_DIR, TEST_MASK_DIR)
