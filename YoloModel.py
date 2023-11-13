from ultralytics import YOLO
import cv2

class YOLOModel:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def get_predictions(self, color_image):
        cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)  # ensure correct color format
        return self.model(color_image, stream=True)
