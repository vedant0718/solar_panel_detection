from ultralytics import YOLO
import os

def train_model():
    # Initialize model with a pre-trained YOLO model (e.g., yolov8n.pt)
    model = YOLO('yolov8n.pt')
    # Train the model using the dataset configuration from data.yaml
    model.train(data='dataset/data.yaml', epochs=50, imgsz=416, batch=16)
    # Save the best model weights to the designated folder
    model.save('results/models/solar_panel_yolo/weights/best.pt')
    return "Model training completed and best weights saved."
