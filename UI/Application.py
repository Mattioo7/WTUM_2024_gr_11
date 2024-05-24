import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO

class VideoPlayer:
    def __init__(self, root):
        self.root = root
        self.root.title("Simple Video Player with YOLOv8 Object Detection")

        # Set fixed player size
        self.canvas_width = 640
        self.canvas_height = 480

        # Load YOLOv8 models
        self.localization_model = YOLO('Models/sign_localization.pt')
        self.classification_model = YOLO('Models/sign_classification_medium.pt')

        # Create GUI elements
        self.canvas = tk.Canvas(root, bg='black', width=self.canvas_width, height=self.canvas_height)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.control_frame = tk.Frame(root)
        self.control_frame.pack(fill=tk.X)

        self.play_button = tk.Button(self.control_frame, text="Play", command=self.play)
        self.play_button.pack(side=tk.LEFT)

        self.pause_button = tk.Button(self.control_frame, text="Pause", command=self.pause)
        self.pause_button.pack(side=tk.LEFT)

        self.stop_button = tk.Button(self.control_frame, text="Stop", command=self.stop)
        self.stop_button.pack(side=tk.LEFT)

        self.open_button = tk.Button(self.control_frame, text="Open", command=self.open_file)
        self.open_button.pack(side=tk.LEFT)

        self.camera_button = tk.Button(self.control_frame, text="Camera", command=self.camera_on)
        self.camera_button.pack(side=tk.LEFT)

        self.volume_slider = tk.Scale(self.control_frame, from_=0, to=100, orient=tk.HORIZONTAL)
        self.volume_slider.set(50)
        self.volume_slider.pack(side=tk.RIGHT)

        self.video_source = None
        self.video_capture = None
        self.is_paused = False
        self.frame = None
        self.camera = None

        self.update_canvas()

    def play(self):
        if self.video_capture is None and self.camera is None:
            return
        self.is_paused = False

    def pause(self):
        self.is_paused = True

    def stop(self):
        self.is_paused = True
        if self.video_capture:
            self.video_capture.release()
            self.video_capture = None
        if self.camera:
            self.camera.release()
            self.camera = None
        self.canvas.delete("all")
        self.camera_button["state"] = 'active'

    def open_file(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.stop()
            self.camera_button["state"] = 'disabled'
            self.video_source = file_path
            self.video_capture = cv2.VideoCapture(self.video_source)
            self.play()

    def camera_on(self):
        self.stop()
        self.camera_button["state"] = 'disabled'
        self.camera = cv2.VideoCapture(0)
        self.play()

    def resize_frame(self, frame):
        h, w = frame.shape[:2]
        aspect_ratio = w / h

        if w > self.canvas_width or h > self.canvas_height:
            if aspect_ratio > 1:
                # Width is the limiting factor
                new_w = self.canvas_width
                new_h = int(new_w / aspect_ratio)
            else:
                # Height is the limiting factor
                new_h = self.canvas_height
                new_w = int(new_h * aspect_ratio)
        else:
            new_w, new_h = w, h

        resized_frame = cv2.resize(frame, (new_w, new_h))
        return resized_frame

    def update_canvas(self):
        if not self.is_paused and ((self.video_capture and self.video_capture.isOpened()) or (self.camera and self.camera.isOpened())):
            ret, frame = None, None
            if(self.video_capture):
                ret, frame = self.video_capture.read()
            else:
                ret, frame = self.camera.read()
            if ret:
                # Perform object detection with YOLOv8
                results = self.localization_model(frame)

                # Draw bounding boxes on the frame
                for result in results:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        localized_sign = frame[y1:y2, x1:x2]
                        sign_class_results = self.classification_model(localized_sign)
                        try:
                            sign_id = int(sign_class_results[0].boxes.cls[0].item())
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, f'{sign_class_results[0].names[sign_id]}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        except:
                            print('Could not classify a sign')

                # # Resize frame to fit the canvas while maintaining aspect ratio
                frame = self.resize_frame(frame)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img)
                #
                # # Clear the canvas and update with the new frame
                self.canvas.delete("all")
                self.canvas.create_image(self.canvas_width//2, self.canvas_height//2, anchor=tk.CENTER, image=imgtk)
                self.root.imgtk = imgtk  # Keep a reference to avoid garbage collection

        # Call update_canvas again after 33ms (approximately 30 FPS)
        self.root.after(33, self.update_canvas)

if __name__ == "__main__":
    root = tk.Tk()
    player = VideoPlayer(root)
    root.mainloop()
