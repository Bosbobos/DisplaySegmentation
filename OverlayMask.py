import cv2
import numpy as np
import os


def overlay_mask(image, mask, color=(0, 255, 0), alpha=0.5):
    """
    Накладывает бинарную маску на изображение с полупрозрачным цветом.
    """
    if mask is None:
        raise ValueError("Маска не загружена или повреждена!")

    mask_colored = np.zeros_like(image, dtype=np.uint8)
    mask_colored[:] = color

    # Убедимся, что mask - одноканальное изображение
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    mask_alpha = (mask > 0).astype(np.uint8) * 255

    # Применяем битовую маску
    mask_colored = cv2.bitwise_and(mask_colored, mask_colored, mask=mask_alpha)

    # Накладываем маску
    overlay = cv2.addWeighted(image, 1, mask_colored, alpha, 0)

    return overlay


def process_folder(image_folder, mask_folder, output_folder, color=(0, 0, 255), alpha=0.5):
    """
    Обрабатывает папку с изображениями и соответствующими масками.
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

        if mask.shape[:2] != image.shape[:2]:

            # Маска приведена к размеру изображения
            resized_mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            result_resized_to_image = overlay_mask(image, resized_mask, color=color, alpha=alpha)
            output_path_resized_to_image = os.path.join(output_folder,
                                                        f"{os.path.splitext(img_name)[0]}_resized_to_image.png")
            cv2.imwrite(output_path_resized_to_image, result_resized_to_image)
        else:
            result = overlay_mask(image, mask, color=color, alpha=alpha)
            output_path = os.path.join(output_folder, img_name)
            cv2.imwrite(output_path, result)

        #print(f"Обработано: {img_name}")


# Параметры
image_folder = "test_images"  # Папка с изображениями
mask_folder = "generated_masks"  # Папка с масками
output_folder = "overlayed_masks"  # Папка для сохраненных изображений
mask_color = (0, 0, 255)  # Красный цвет маски
mask_alpha = 0.5  # Прозрачность

# Запуск обработки
process_folder(image_folder, mask_folder, output_folder, color=mask_color, alpha=mask_alpha)
