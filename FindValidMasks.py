import os
import shutil
import cv2
import numpy as np


def is_valid_mask(mask, min_area_ratio=0.01):
    # Находим контуры на маске
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Проверяем, что ровно один контур
    if len(contours) != 1:
        return False

    contour = contours[0]

    # Проверяем, что контур выпуклый
    if not cv2.isContourConvex(contour):
        return False

    # Аппроксимируем контур до многоугольника
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    # Проверяем, что это четырехугольник
    if len(approx) != 4:
        return False

    # Проверяем, что все углы на изображении
    height, width = mask.shape[:2]
    for point in approx:
        x, y = point[0]
        if x < 0 or y < 0 or x >= width or y >= height:
            return False

    # Проверяем, что площадь маски больше минимального порога
    image_area = height * width
    area = cv2.contourArea(contour)
    if area < min_area_ratio * image_area:
        return False

    return True


def process_masks(input_folder, output_folder, image_folder, output_image_folder, min_area_ratio=0.01):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if not os.path.exists(output_image_folder):
        os.makedirs(output_image_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith('.png'):
            image_path = os.path.join(input_folder, filename)
            mask = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            if mask is None:
                continue

            if is_valid_mask(mask, min_area_ratio):
                shutil.copy(image_path, os.path.join(output_folder, filename))

                # Копирование соответствующего изображения
                image_file_path = os.path.join(image_folder, filename)
                if os.path.exists(image_file_path):
                    shutil.copy(image_file_path, os.path.join(output_image_folder, filename.replace('.png', '.jpg')))


# Укажите пути к папкам
input_folder = 'coco_tv_masks'  # Папка с масками
output_folder = 'valid_masks'  # Папка для отфильтрованных масок
image_folder = 'coco_tv_images'  # Папка с исходными изображениями
output_image_folder = 'valid_images'  # Папка для соответствующих изображений

process_masks(input_folder, output_folder, image_folder, output_image_folder)
