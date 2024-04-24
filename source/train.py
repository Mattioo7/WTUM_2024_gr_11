import torch
from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")  
    #results = model.train(data="config.yaml", epochs=11)  # train the model
    results = model("35.jpg")

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()  # Dodaj tę linię
    main()
