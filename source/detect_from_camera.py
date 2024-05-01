from ultralytics import YOLO
import cv2
import math

# start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

cap.set(3, 640)
cap.set(4, 480)

# model
model = YOLO("C:/Users/Jakub/Desktop/Nowy folder/WTUM_2024_gr_11/source/runs/detect/train80e/weights/best.pt")

# object classes
classNames = {0: 'data', 1: 'A-1', 2: 'A-11', 3: 'A-11a', 4: 'A-12a', 5: 'A-14', 6: 'A-15', 7: 'A-16', 8: 'A-17', 9: 'A-18b', 10: 'A-2', 11: 'A-20', 12: 'A-21', 13: 'A-24', 14: 'A-29', 15: 'A-3', 16: 'A-30', 17: 'A-32', 18: 'A-4', 19: 'A-6a', 20: 'A-6b', 21: 'A-6c', 22: 'A-6d', 23: 'A-6e', 24: 'A-7', 25: 'A-8', 26: 'B-1', 27: 'B-18', 28: 'B-2', 29: 'B-20', 30: 'B-21', 31: 'B-22', 32: 'B-23', 33: 'B-25', 34: 'B-26', 35: 'B-27', 36: 'B-33', 37: 'B-34', 38: 'B-36', 39: 'B-41', 40: 'B-42', 41: 'B-43', 42: 'B-44', 43: 'B-5', 44: 'B-6-B-8-B-9', 45: 'B-8', 46: 'B-9', 47: 'C-10', 48: 'C-12', 49: 'C-13', 50: 'C-13-C-16', 51: 'C-13a', 52: 'C-13a-C-16a', 53: 'C-16', 54: 'C-2', 55: 'C-4', 56: 'C-5', 57: 'C-6', 58: 'C-7', 59: 'C-9', 60: 'D-1', 61: 'D-14', 62: 'D-15', 63: 'D-18', 64: 'D-18b', 65: 'D-2', 66: 'D-21', 67: 'D-23', 68: 'D-23a', 69: 'D-24', 70: 'D-26', 71: 'D-26b', 72: 'D-26c', 73: 'D-27', 74: 'D-28', 75: 'D-29', 76: 'D-3', 77: 'D-40', 78: 'D-41', 79: 'D-42', 80: 'D-43', 81: 'D-4a', 82: 'D-4b', 83: 'D-51', 84: 'D-52', 85: 'D-53', 86: 'D-6', 87: 'D-6b', 88: 'D-7', 89: 'D-8', 90: 'D-9', 91: 'D-tablica', 92: 'G-1a', 93: 'G-3'}

while True:
    success, img = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue
    
    results = model(img, stream=True)

    # coordinates
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

            # put box in cam
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # confidence
            confidence = math.ceil((box.conf[0]*100))/100
            print("Confidence --->", confidence)

            # class name
            cls = int(box.cls[0])
            class_name = classNames.get(cls, f"Unknown class: {cls}")
            print("Class name -->", class_name)

            # object details
            org = (x1, y1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2

            cv2.putText(img, class_name, org, font, fontScale, color, thickness)

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()