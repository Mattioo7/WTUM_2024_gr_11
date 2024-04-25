from ultralytics import YOLO
import torch
import os
import cv2

# Ścieżka do wytrenowanych wag i zestawu walidacyjnego
weights_path = 'C:/Users/Jakub/Desktop/WTUM_2024_gr_11-main/train80e/weights/best.pt'
val_dir = 'C:/Users/Jakub/Desktop/ValidationSet'

# Wczytanie modelu
model = YOLO(weights_path)

# Zestaw walidacyjny
val_images = [os.path.join(val_dir, x) for x in os.listdir(val_dir) if x.endswith(('.png', '.jpg', '.jpeg'))]

# Wyniki
results = []
for img_path in val_images:
    # Wczytywanie obrazu
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Konwersja do RGB

    # Inferencja (przewidywanie)
    pred = model(img)

    # Zapisywanie wyników
    results.append(pred)

