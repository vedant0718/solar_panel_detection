from ultralytics import YOLO
import os

def train_model():
    model = YOLO('yolov8n.pt')
    model.train(data='dataset/data.yaml', epochs=50, imgsz=416, batch=16)
    model.save('results/models/solar_panel_yolo/weights/best.pt')
    return "Model training completed and best weights saved."
