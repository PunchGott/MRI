import os
import sys
import time
import sqlite3
import json
import tkinter as tk
from tkinter import filedialog
from datetime import datetime

import cv2
import numpy as np
from ultralytics import YOLO


# ---------- Работа с базой данных ----------
def init_database(db_path="results.db"):
    """
    Создаёт таблицу для хранения результатов, если она не существует.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS segmentation_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_name TEXT NOT NULL,
            image_path TEXT NOT NULL,
            processed_at TEXT NOT NULL,
            num_objects INTEGER,
            total_time REAL,
            boxes_json TEXT,          -- JSON-массив с координатами рамок
            confidences_json TEXT,    -- JSON-массив с уверенностями
            masks_json TEXT,          -- JSON-массив с контурами (список точек)
            model_used TEXT
        )
    ''')
    conn.commit()
    conn.close()


def save_result_to_db(db_path, image_name, image_path, num_objects, total_time,
                      boxes, confidences, contours_list, model_name="yolov8n-seg.pt"):
    """
    Сохраняет результаты обработки одного изображения в базу данных.
    
    Args:
        db_path (str): путь к файлу БД.
        image_name (str): имя файла.
        image_path (str): полный путь к исходному изображению.
        num_objects (int): количество обнаруженных объектов.
        total_time (float): время обработки в секундах.
        boxes (np.ndarray или None): массив рамок [x1,y1,x2,y2].
        confidences (np.ndarray или None): массив уверенностей.
        contours_list (list): список контуров (каждый контур - список точек).
        model_name (str): название использованной модели.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Преобразуем данные в JSON
    boxes_json = json.dumps(boxes.tolist()) if boxes is not None else None
    conf_json = json.dumps(confidences.tolist()) if confidences is not None else None
    contours_json = json.dumps(contours_list) if contours_list else None
    
    processed_at = datetime.now().isoformat()
    
    cursor.execute('''
        INSERT INTO segmentation_results 
        (image_name, image_path, processed_at, num_objects, total_time,
         boxes_json, confidences_json, masks_json, model_used)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (image_name, image_path, processed_at, num_objects, total_time,
          boxes_json, conf_json, contours_json, model_name))
    
    conn.commit()
    conn.close()


# ---------- Вспомогательные функции ----------
def select_folder():
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(title="Выберите папку с МРТ-изображениями")
    root.destroy()
    return folder_path if folder_path else None


def extract_contours_from_masks(masks, image_shape):
    """
    Извлекает контуры из масок сегментации.
    
    Args:
        masks (torch.Tensor или None): тензор масок.
        image_shape (tuple): (height, width) исходного изображения.
    
    Returns:
        list: список контуров, каждый представлен списком точек [x, y].
    """
    if masks is None:
        return []
    
    if hasattr(masks, 'data'):
        masks_np = masks.data.cpu().numpy()
    else:
        masks_np = np.array(masks)
    
    contours_list = []
    for mask in masks_np:
        binary_mask = (mask > 0.5).astype(np.uint8) * 255
        # Находим внешние контуры
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Берём самый большой контур (обычно один)
            c = max(contours, key=cv2.contourArea)
            # Преобразуем в список точек (x, y)
            points = c.squeeze().tolist()
            if isinstance(points, list) and points:
                contours_list.append(points)
            else:
                # Если вырожденный случай, пропускаем
                pass
    return contours_list


def draw_contours_on_image(image, masks, boxes=None, class_ids=None, confidences=None):
    result_img = image.copy()
    if masks is None:
        return result_img
    
    if hasattr(masks, 'data'):
        masks_np = masks.data.cpu().numpy()
    else:
        masks_np = np.array(masks)
    
    for i, mask in enumerate(masks_np):
        binary_mask = (mask > 0.5).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result_img, contours, -1, (0, 255, 0), 2)
        
        if boxes is not None and i < len(boxes):
            x1, y1, x2, y2 = map(int, boxes[i])
            cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 1)
            if class_ids is not None and confidences is not None and i < len(confidences):
                label = f"tumor {confidences[i]:.2f}"
                cv2.putText(result_img, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return result_img


def process_images(input_dir, output_dir, model, db_path="results.db"):
    supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
    
    try:
        files = [f for f in os.listdir(input_dir) 
                if f.lower().endswith(supported_extensions)]
    except OSError as e:
        print(f"Ошибка при чтении директории {input_dir}: {e}")
        return 0, 0.0
    
    if not files:
        print("В выбранной папке не найдено изображений с поддерживаемыми расширениями.")
        return 0, 0.0
    
    print(f"Найдено изображений для обработки: {len(files)}")
    print("-" * 50)
    
    os.makedirs(output_dir, exist_ok=True)
    
    processed_count = 0
    total_time = 0.0
    
    for idx, filename in enumerate(files, 1):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        print(f"[{idx}/{len(files)}] Обработка: {filename}")
        
        try:
            image = cv2.imread(input_path)
            if image is None:
                print(f"  Предупреждение: не удалось загрузить изображение {filename}")
                continue
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            start_time = time.time()
            
            results = model.predict(
                source=image_rgb,
                imgsz=640,
                conf=0.25,
                verbose=False
            )
            
            end_time = time.time()
            inference_time = end_time - start_time
            total_time += inference_time
            
            # Извлечение данных
            if results and len(results) > 0:
                result = results[0]
                masks = result.masks
                boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else None
                confidences = result.boxes.conf.cpu().numpy() if result.boxes is not None else None
                class_ids = result.boxes.cls.cpu().numpy().astype(int) if result.boxes is not None else None
                
                # Извлекаем контуры для БД
                contours_list = extract_contours_from_masks(masks, image.shape[:2])
                
                num_objects = len(boxes) if boxes is not None else 0
                print(f"  Обнаружено объектов: {num_objects}, время: {inference_time:.2f} сек.")
                
                # Визуализация
                result_image = draw_contours_on_image(image, masks, boxes, class_ids, confidences)
            else:
                num_objects = 0
                contours_list = []
                result_image = image
                print(f"  Объектов не обнаружено, время: {inference_time:.2f} сек.")
            
            # Сохранение изображения
            cv2.imwrite(output_path, result_image)
            
            # Сохранение в БД
            save_result_to_db(
                db_path=db_path,
                image_name=filename,
                image_path=input_path,
                num_objects=num_objects,
                total_time=inference_time,
                boxes=boxes if 'boxes' in locals() else None,
                confidences=confidences if 'confidences' in locals() else None,
                contours_list=contours_list,
                model_name="yolov8n-seg.pt"
            )
            
            processed_count += 1
            
        except Exception as e:
            print(f"  Ошибка при обработке {filename}: {e}")
            try:
                if 'image' in locals() and image is not None:
                    cv2.imwrite(output_path, image)
            except:
                pass
            continue
    
    print("-" * 50)
    print(f"Обработка завершена. Обработано изображений: {processed_count}")
    print(f"Общее время: {total_time:.2f} сек.")
    print(f"Результаты сохранены в: {output_dir}")
    print(f"Данные сохранены в базу данных: {db_path}")
    
    return processed_count, total_time


def main():
    print("=" * 60)
    print("   Сегментация опухолей на МРТ с использованием YOLOv8")
    print("   (отрисовка контуров + сохранение в БД)")
    print("=" * 60)
    print()
    
    # Инициализация БД
    db_path = "segmentation_results.db"
    init_database(db_path)
    print(f"База данных инициализирована: {db_path}")
    print()
    
    print("Выберите папку с МРТ-изображениями для обработки...")
    input_dir = select_folder()
    if not input_dir:
        print("Папка не выбрана. Программа завершает работу.")
        sys.exit(0)
    print(f"Выбрана папка: {input_dir}")
    print()
    
    print("Загрузка модели YOLOv8 для сегментации...")
    try:
        model = YOLO('yolov8n-seg.pt')
        print("Модель успешно загружена.")
    except Exception as e:
        print(f"Ошибка при загрузке модели: {e}")
        sys.exit(1)
    print()
    
    output_dir = os.path.join(input_dir, 'results')
    print(f"Результаты будут сохранены в: {output_dir}")
    print()
    
    print("Начинается обработка изображений...")
    print()
    
    processed, total_time = process_images(input_dir, output_dir, model, db_path)
    
    print()
    print("=" * 60)
    print("Программа завершила работу.")
    print("=" * 60)


if __name__ == "__main__":
    main()
