import cv2
import numpy as np
import os

def overlay_mask(image, mask, color=(0, 255, 0), alpha=0.5):
    """
    Накладывает бинарную маску на изображение с полупрозрачным цветом.

    :param image: Исходное изображение (BGR)
    :param mask: Бинарная маска (grayscale)
    :param color: Цвет для наложения (BGR)
    :param alpha: Прозрачность маски (0 - прозрачная, 1 - непрозрачная)
    :return: Изображение с наложенной маской
    """
    mask_colored = np.zeros_like(image)
    mask_colored[:] = color

    mask_alpha = (mask > 0).astype(np.uint8) * 255
    mask_colored = cv2.bitwise_and(mask_colored, mask_colored, mask=mask_alpha)

    overlay = cv2.addWeighted(image, 1, mask_colored, alpha, 0)

    return overlay

def process_folder(image_folder, mask_folder, output_folder, color=(0, 0, 255), alpha=0.5):
    """
    Обрабатывает папку с изображениями и соответствующими масками.

    :param image_folder: Папка с изображениями
    :param mask_folder: Папка с масками (названия должны совпадать)
    :param output_folder: Папка для сохранения результата
    :param color: Цвет наложенной маски (BGR)
    :param alpha: Прозрачность маски
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('png', 'jpg', 'jpeg'))]

    for img_name in image_files:
        img_path = os.path.join(image_folder, img_name)
        mask_path = os.path.join(mask_folder, img_name)

        if not os.path.exists(mask_path):
            print(f"Нет маски для {img_name}, пропускаем.")
            continue

        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            print(f"Ошибка загрузки {img_name}, пропускаем.")
            continue

        result = overlay_mask(image, mask, color=color, alpha=alpha)

        output_path = os.path.join(output_folder, img_name)
        cv2.imwrite(output_path, result)
        print(f"Обработано: {img_name}")

# Параметры
image_folder = "MostImportant"  # Папка с изображениями
mask_folder = "generated_masks"  # Папка с масками
output_folder = "overlayed_masks"  # Папка для сохраненных изображений
mask_color = (0, 0, 255)  # Красный цвет маски
mask_alpha = 0.5  # Прозрачность

# Запуск обработки
process_folder(image_folder, mask_folder, output_folder, color=mask_color, alpha=mask_alpha)
