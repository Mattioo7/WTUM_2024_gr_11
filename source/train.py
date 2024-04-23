import torch
from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.yaml")  # build a new model from scratch
    results = model.train(data="config.yaml", epochs=1)  # train the model

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()  # Dodaj tę linię
    main()
