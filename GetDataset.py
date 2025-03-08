import os
import requests
import numpy as np
import cv2
import concurrent.futures
from pycocotools.coco import COCO
from pycocotools import mask as mask_utils

# Пути
ANNOTATIONS_DIR = "annotations"  # Папка с аннотациями
ANNOTATIONS_PATH = os.path.join(ANNOTATIONS_DIR, "instances_train2017.json")
ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

OUTPUT_IMAGES_DIR = "coco_tv_images"  # Куда сохранять изображения
OUTPUT_MASKS_DIR = "coco_tv_masks"  # Куда сохранять маски
NUM_THREADS = 8  # Количество потоков

# Создаём папки, если их нет
os.makedirs(ANNOTATIONS_DIR, exist_ok=True)
os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)
os.makedirs(OUTPUT_MASKS_DIR, exist_ok=True)

# ✅ Функция для скачивания аннотаций COCO
def download_annotations():
    zip_path = os.path.join(ANNOTATIONS_DIR, "annotations_trainval2017.zip")
    if not os.path.exists(ANNOTATIONS_PATH):  # Если JSON ещё не скачан
        print("📥 Аннотации COCO не найдены, скачиваем...")
        response = requests.get(ANNOTATIONS_URL, stream=True)
        if response.status_code == 200:
            with open(zip_path, "wb") as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            print("✅ Аннотации загружены, распаковываем...")
            import zipfile
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall('./')
            os.remove(zip_path)  # Удаляем ZIP после распаковки
        else:
            print("❌ Ошибка загрузки аннотаций COCO")
            exit()
    else:
        print("✅ Аннотации COCO уже загружены.")

# Скачиваем аннотации, если их нет
download_annotations()

# Загружаем аннотации COCO
coco = COCO(ANNOTATIONS_PATH)

# Найдём категорию "TV"
cat_ids = coco.getCatIds(catNms=["tv"])
if not cat_ids:
    print("Категория 'TV' не найдена в COCO!")
    exit()

# Получаем все изображения, содержащие "TV"
img_ids = coco.getImgIds(catIds=cat_ids)
images = coco.loadImgs(img_ids)

print(f"Найдено {len(images)} изображений с 'TV'. Начинаем загрузку...")

# Функция скачивания и создания маски
def process_image(img_data):
    img_filename = img_data["file_name"]
    img_url = img_data["coco_url"]
    img_id = img_data["id"]

    save_img_path = os.path.join(OUTPUT_IMAGES_DIR, img_filename)
    save_mask_path = os.path.join(OUTPUT_MASKS_DIR, img_filename.replace(".jpg", ".png"))

    # Если изображение скачано, но маски нет – создаём маску
    if os.path.exists(save_img_path) and not os.path.exists(save_mask_path):
        pass  # Переходим к созданию маски

    # Если изображение и маска уже есть – пропускаем
    elif os.path.exists(save_img_path) and os.path.exists(save_mask_path):
        return f"🔹 Уже обработано: {img_filename}"

    # Скачивание изображения (если не скачано)
    else:
        response = requests.get(img_url, stream=True, timeout=10)
        if response.status_code == 200:
            with open(save_img_path, "wb") as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
        else:
            return f"❌ Ошибка скачивания: {img_filename}"

    # Создаём пустую маску
    mask = np.zeros((img_data["height"], img_data["width"]), dtype=np.uint8)

    # Получаем аннотации
    ann_ids = coco.getAnnIds(imgIds=img_id, catIds=cat_ids)
    anns = coco.loadAnns(ann_ids)

    for ann in anns:
        if "segmentation" in ann:
            segmentation = ann["segmentation"]

            try:
                # 🎯 1. Polygon (список координат)
                if isinstance(segmentation, list):
                    for seg in segmentation:
                        pts = np.array(seg, np.int32).reshape((-1, 1, 2))
                        cv2.fillPoly(mask, [pts], 1)  # 1 - экран, 0 - фон

                # 🎯 2. Обычный RLE (словарь)
                elif isinstance(segmentation, dict):
                    rle_mask = mask_utils.decode(segmentation)
                    mask[rle_mask > 0] = 1  # 1 - экран

                # 🎯 3. Список RLE (его нужно декодировать)
                elif isinstance(segmentation, list) and isinstance(segmentation[0], dict):
                    rle_combined = mask_utils.frPyObjects(segmentation, img_data["height"], img_data["width"])
                    rle_mask = mask_utils.decode(rle_combined)
                    mask[rle_mask > 0] = 1  # 1 - экран

                else:
                    print(f"⚠️ Неподдерживаемый формат сегментации в файле: {img_filename}")

            except Exception as e:
                print(f"❌ Ошибка обработки маски для {img_filename}: {e}")
                continue  # Пропускаем это изображение

    # Проверяем, не пустая ли маска
    if np.sum(mask) == 0:
        print(f"⚠️ Пустая маска (удалена): {img_filename}")
        return f"⚠️ Пустая маска (удалена): {img_filename}"

    # Сохраняем маску
    cv2.imwrite(save_mask_path, mask * 255)  # 0 (фон), 255 (экран)

    return f"✅ Готово: {img_filename}"

# Запускаем многопоточное скачивание и обработку
with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
    results = list(executor.map(process_image, images))

# Вывод результатов
for res in results:
    print(res)

print("🎉 Все изображения загружены и маски созданы!")
