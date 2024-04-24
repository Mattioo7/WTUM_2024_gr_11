import torch
from ultralytics import YOLO

def main():
    model = YOLO("runs/detect/train/weights/best.pt")  # build a new model from scratch
    results = model("https://incor.com.pl/wp-content/uploads/2022/04/pion-s1-z1.jpg", save=True) # train the model

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()  # Dodaj tę linię
    main()
