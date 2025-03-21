import os

import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from shapely.geometry import Polygon

# Константы:
BOX_THICKNESS = 3
TEXT_THICKNESS = 6
TEXT_SCALE = 2
TEXT_COORDS = (1700, 150)
IMAGE_FIGSIZE = (8, 6)
ROI_SIZE = 300
HEELS_CONFIDENCE = 0.5
BOOTS_CONFIDENCE = 0.6


# Доп. функции, которые используются дальше при решении конкретных задач:
def get_box_corners_coords(xywh_coords):
    corners_coords = []
    corners_coords.append((xywh_coords[0] - xywh_coords[2] / 2,
                           xywh_coords[1] - xywh_coords[3] / 2))
    corners_coords.append((xywh_coords[0] + xywh_coords[2] / 2,
                           xywh_coords[1] - xywh_coords[3] / 2))
    corners_coords.append((xywh_coords[0] + xywh_coords[2] / 2,
                           xywh_coords[1] + xywh_coords[3] / 2))
    corners_coords.append((xywh_coords[0] - xywh_coords[2] / 2,
                           xywh_coords[1] + xywh_coords[3] / 2))
    return corners_coords


def xywh_to_xyxy(xywh_coords):
    xyxy_coords = [
        int(xywh_coords[0] - xywh_coords[2] / 2),
        int(xywh_coords[1] - xywh_coords[3] / 2),
        int(xywh_coords[0] + xywh_coords[2] / 2),
        int(xywh_coords[1] + xywh_coords[3] / 2),
    ]
    return xyxy_coords


def intersection(clear_zone, xyxy_coords):
    area_of_intersection = 0
    for i in range(xyxy_coords[0], xyxy_coords[2]):
        for j in range(xyxy_coords[1], xyxy_coords[3]):
            if clear_zone[j, i] == 255.0:
                area_of_intersection += 1
    return area_of_intersection


def union(area_of_clear_zone, xyxy_coords, area_of_intersection):
    area_of_box = (int(abs(xyxy_coords[2] - xyxy_coords[0])) * int(abs(xyxy_coords[3] - xyxy_coords[1])))
    area_of_union = area_of_clear_zone + area_of_box - area_of_intersection
    return area_of_union


def IoU(clear_zone, area_of_clear_zone, xyxy_coords):
    area_of_intersection = intersection(clear_zone, xyxy_coords)
    area_of_union = union(area_of_clear_zone, xyxy_coords, area_of_intersection)
    return area_of_intersection / area_of_union


def box_drawing(image, xyxy_coords, cls):
    color = (0, 0, 255) if cls == 0 else (0, 255, 0)
    image = cv2.rectangle(image, (int(xyxy_coords[0]), int(xyxy_coords[1])),
                          (int(xyxy_coords[2]), int(xyxy_coords[3])), color, thickness=BOX_THICKNESS)
    return image


def text_drawing(image, text):
    color = (0, 0, 255) if text == 'ALARM' else (0, 255, 0)
    image = cv2.putText(image, text, TEXT_COORDS, cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=TEXT_SCALE, color=color, thickness=TEXT_THICKNESS)
    return image


def show_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=IMAGE_FIGSIZE)
    plt.imshow(image)
    plt.axis('off')
    plt.show()


# Задача 1
# Нахождение позы человека. Для этого используйте предоставленную готовую модель - YOLOv12-S pose-estimination.
# Вам важно найти местоположение пяток человека в пространстве для контроля пребывания их в чистой или грязной
# зоне (как правило это последние по индексу keypoints в result-е инференса модели).
def get_heels(model, confidence, image_path):
    pose_result = model(image_path, conf=confidence, verbose=False)[0]
    all_heels = []
    for keypoints in pose_result.keypoints.xy:
        all_heels.append(keypoints[-1].tolist())
        all_heels.append(keypoints[-2].tolist())
    annotated_img = pose_result.plot()
    return all_heels, annotated_img


# Задача 2
# Нахождение и детекция обуви. Необходимо распознать на картинке два класса, грязную (dirt boot) и чистую обувь
# (green boot). Для облегчения выполнения задачи модель детекции обуви уже дана (на базе YOLOv12-S).
def get_boots(model, confidence, image_path):
    boots_result = model(image_path, conf=confidence, verbose=False)[0]
    boots_list = []
    for box in boots_result.boxes:
        boots_list.append((int(box.cls), box.xywh[0].tolist()))
    return boots_list


# Задача 3
# Напишите решение, которое способно определить, находится ли dirt boot или green boot в области ROI,
# центром которой является пятка скелета. Размеры 'ROI' определите как полигон (набор связанных точек).
def get_heels_boots_match(heels, boots, heel_ROI_coords):
    heels_boots_list = []
    for heel in heels:
        heel_x, heel_y = float(heel[0]), float(heel[1])
        heel_boots_list = [(heel_x, heel_y)]
        current_ROI_coords = [(heel_x + coord[0], heel_y + coord[1]) for coord in heel_ROI_coords]
        for boot in boots:
            ROI_polygon = Polygon(current_ROI_coords)
            boot_coords = get_box_corners_coords(boot[1])
            boot_polygon = Polygon(boot_coords)
            if ROI_polygon.contains(boot_polygon):
                heel_boots_list.append(boot)
        heels_boots_list.append(heel_boots_list)
    return heels_boots_list


# Продолжение задачи 3. Подпункты:
# Необходимо, чтобы ваше решение отрабатывало следующие ситуации:
# - если пятка находится в грязной зоне, обувь находится в грязной зоне (с некоторым IoU), и обувь находится рядом
#   к пятке - выдать ALARM
# - если пятка находится в чистой зоне, обувь находится в чистой зоне (с некоторым IoU), и обувь находится рядом
#   к пятке - выдать GOOD
# - грязная обувь в чистой зоне без человека не считается за нарушение. Чистая обувь в грязной и с человеком,
#   и без человека - не нарушение.
def get_status(heels_boots_list, clear_zone, area_of_clear_zone, image, iou=0.001):
    """
    Функция может вернуть одно из трёх значений: ALARM (первый пункт), GOOD (второй пункт)
    и пустую строку (все остальные случаи). Помимо этого возвращает изображение с
    нарисованными поверх него ограничивающими рамками и текстом.
    """
    current_status = ''
    for heel_boots in heels_boots_list:
        if len(heel_boots) < 2:
            continue
        heel_x, heel_y = int(heel_boots[0][0]), int(heel_boots[0][1])
        for boot in heel_boots[1:]:
            if clear_zone[heel_y, heel_x] == 255.0:
                boot_class = boot[0]
                xyxy_boot_coords = xywh_to_xyxy(boot[1])
                if IoU(clear_zone, area_of_clear_zone, xyxy_boot_coords) > iou:
                    if not boot_class:
                        current_status = 'ALARM'
                    elif boot_class:
                        current_status = 'GOOD' if current_status != 'ALARM' else current_status
                    image = box_drawing(image, xyxy_boot_coords, boot_class)
                    image = text_drawing(image, current_status)
    return current_status, image


# Все задачи вместе:
def main():
    filenames = os.listdir('test_data')
    pose_model = YOLO('yolo11s-pose.pt')
    boots_model = YOLO('yolo12s-boots.pt')
    clear_zone = cv2.imread('clear_zone_mask.jpg', cv2.IMREAD_GRAYSCALE)
    area_of_clear_zone = np.count_nonzero(clear_zone[clear_zone == 255.0])
    # Наверное, можно по-другому посчитать площадь чистой зоны, но решил в эту сторону не думать.
    # Камера в этой задаче снимает одно и то же место, и в реальной работе площадь чистой зоны не придётся рассчитывать
    # каждый раз заново. Достаточно один раз вычислить её в начале работы программы, а затем использовать для всех
    # изображений, приходящих с камеры.
    for filename in filenames:
        image_path = os.path.join('test_data', filename)
        heels, annotated_image = get_heels(pose_model, HEELS_CONFIDENCE, image_path)
        boots = get_boots(boots_model, BOOTS_CONFIDENCE, image_path)
        heels_boots_list = get_heels_boots_match(heels, boots, heel_ROI_coords=[(-ROI_SIZE, -ROI_SIZE),
                                                                                (-ROI_SIZE, ROI_SIZE),
                                                                                (ROI_SIZE, ROI_SIZE),
                                                                                (ROI_SIZE, -ROI_SIZE)])
        result, annotated_image = get_status(heels_boots_list, clear_zone, area_of_clear_zone, annotated_image)
        #show_image(annotated_image) В docker'е всегда проблемно работать с GUI и его элементами, поэтому проще отключить.
        print(f'{filename} - {result}')


if __name__ == '__main__':
    main()
