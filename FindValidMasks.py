import os
import cv2
import shutil

# Пути к папкам (замени на свои)
MASKS_FOLDER = "coco_tv_masks"
IMAGES_FOLDER = "coco_tv_images"
OUTPUT_MASKS_FOLDER = "valid_masks"
OUTPUT_IMAGES_FOLDER = "valid_images"

# Создаем папки, если их нет
os.makedirs(OUTPUT_MASKS_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_IMAGES_FOLDER, exist_ok=True)


def relaxed_analyze_mask(mask):
    """Менее строгая проверка маски: учитывает форму, отверстия и заполненность"""
    if mask is None:
        return False

    h, w = mask.shape
    total_area = h * w

    # Преобразуем в бинарное изображение
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Находим контуры
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 0.05 * total_area]  # Фильтр 5% площади

    if len(contours) != 1:
        return False

    cnt = contours[0]
    area = cv2.contourArea(cnt)

    if area > 0.95 * total_area:
        return False

    # Проверка формы
    x, y, w_obj, h_obj = cv2.boundingRect(cnt)
    aspect_ratio = max(w_obj, h_obj) / min(w_obj, h_obj + 1e-5)

    if aspect_ratio > 2:  # Менее строгая проверка формы
        return False

    # Проверяем отверстия
    holes, hierarchy = cv2.findContours(binary_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    hole_areas = [cv2.contourArea(h) for h in holes[1:]]  # Пропускаем основной контур

    if len(hole_areas) > 2 or sum(hole_areas) > 0.05 * total_area:  # Разрешаем до 2 отверстий, но не больше 5% площади
        return False

    # Заполняемость bounding box
    bbox_area = w_obj * h_obj
    fill_ratio = area / bbox_area

    if fill_ratio < 0.8:  # Разрешаем ≥ 80% заполненности
        return False

    return True


# Обработка файлов
for mask_filename in os.listdir(MASKS_FOLDER):
    mask_path = os.path.join(MASKS_FOLDER, mask_filename)

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if relaxed_analyze_mask(mask):
        shutil.copy(mask_path, os.path.join(OUTPUT_MASKS_FOLDER, mask_filename))

        image_path = os.path.join(IMAGES_FOLDER, mask_filename.replace(".png", ".jpg"))  # Или другой формат
        if os.path.exists(image_path):
            shutil.copy(image_path, os.path.join(OUTPUT_IMAGES_FOLDER, os.path.basename(image_path)))

print("Фильтрация завершена! Теперь должно пройти больше изображений.")
