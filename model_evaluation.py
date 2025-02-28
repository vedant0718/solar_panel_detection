import os
import glob
import random
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ultralytics import YOLO
import supervision
from supervision.metrics.detection import ConfusionMatrix
from utils import draw_boxes, format_change
from fundamental_functions import generate_test_data

def visualize_predictions():
    """
    Loads the trained model and for each randomly selected test image:
    - Draws predicted boxes in red.
    - Reads the corresponding ground truth file from dataset/labels/test,
      converts YOLO-format labels to bounding boxes, and draws them in green.
    Returns a list of matplotlib figures.
    """
    model_path = os.path.join('results', 'models', 'solar_panel_yolo', 'weights', 'best.pt')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found at {model_path}")
    model = YOLO(model_path)
    
    test_images = glob.glob(os.path.join('dataset', 'images', 'test', "*.tif"))
    if len(test_images) == 0:
        raise FileNotFoundError("No test images found in the 'dataset/images/test' folder.")
    
    sample_images = random.sample(test_images, min(4, len(test_images)))
    figures = []
    for img_path in sample_images:
        image = cv2.imread(img_path)
        if image is None:
            continue
        
        # Get predicted boxes (red)
        results = model(image)
        if results and len(results) > 0 and results[0].boxes is not None:
            pred_boxes = results[0].boxes.xyxy.cpu().numpy()
        else:
            pred_boxes = []
        draw_boxes(image, pred_boxes, (0, 0, 255), "Prediction")
        
        # Load ground truth boxes from corresponding label file in dataset/labels/test (green)
        base = os.path.splitext(os.path.basename(img_path))[0]
        gt_label_path = os.path.join('dataset', 'labels', 'test', base + ".txt")
        if os.path.exists(gt_label_path):
            with open(gt_label_path, 'r') as f:
                gt_data = f.readlines()
            gt_boxes = []
            h, w = image.shape[:2]
            for line in gt_data:
                parts = line.strip().split()
                if len(parts) == 5:
                    # YOLO format: class, x_center, y_center, width, height (normalized)
                    x_center, y_center, box_w, box_h = map(float, parts[1:])
                    x_center_abs = x_center * w
                    y_center_abs = y_center * h
                    box_w_abs = box_w * w
                    box_h_abs = box_h * h
                    x_min = x_center_abs - box_w_abs/2
                    y_min = y_center_abs - box_h_abs/2
                    x_max = x_center_abs + box_w_abs/2
                    y_max = y_center_abs + box_h_abs/2
                    gt_boxes.append([x_min, y_min, x_max, y_max])
            draw_boxes(image, gt_boxes, (0, 255, 0), "Ground Truth")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        fig = plt.figure(figsize=(8, 6))
        plt.imshow(image_rgb)
        plt.title(f"Predictions & Ground Truth: {os.path.basename(img_path)}")
        plt.axis("off")
        figures.append(fig)
        plt.close(fig)
    return figures

def evaluate_model():
    """
    Generates synthetic test data, converts boxes to the expected tensor format,
    computes confusion matrix metrics over a range of IoU and confidence thresholds,
    and returns precision/recall tables along with default metrics.
    """
    model_path = os.path.join('results', 'models', 'solar_panel_yolo', 'weights', 'best.pt')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found at {model_path}")
    
    gt_data, pred_data, pred_scores_data = generate_test_data()
    
    gt_tensors = []
    pred_tensors = []
    from utils import format_change
    for gt_boxes in gt_data:
        boxes = [format_change(box) for box in gt_boxes]
        gt_tensors.append(np.array([box + [0] for box in boxes]))
    for pred_boxes, confs in zip(pred_data, pred_scores_data):
        boxes = [format_change(box) for box in pred_boxes]
        pred_tensors.append(np.array([box + [0, conf] for box, conf in zip(boxes, confs)]))
    
    cm_default = ConfusionMatrix.from_tensors(
        predictions=pred_tensors,
        targets=gt_tensors,
        classes=["object"],
        conf_threshold=0.5,
        iou_threshold=0.5
    )
    M_default = cm_default.matrix
    TP_default = M_default[1, 1]
    FN_default = M_default[1, 0]
    FP_default = M_default[0, 1]
    precision_default = TP_default / (TP_default + FP_default) if (TP_default + FP_default) > 0 else 0
    recall_default = TP_default / (TP_default + FN_default) if (TP_default + FN_default) > 0 else 0
    
    iou_thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    conf_thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    results = {}
    for iou_thresh in iou_thresholds:
        results[iou_thresh] = {}
        for conf_thresh in conf_thresholds:
            cm = ConfusionMatrix.from_tensors(
                predictions=pred_tensors,
                targets=gt_tensors,
                classes=["object"],
                conf_threshold=conf_thresh,
                iou_threshold=iou_thresh
            )
            M = cm.matrix
            TP = M[1, 1]
            FN = M[1, 0]
            FP = M[0, 1]
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            results[iou_thresh][conf_thresh] = {"precision": precision, "recall": recall, "f1": f1}
    
    precision_df = pd.DataFrame({conf: {iou: results[iou][conf]["precision"] for iou in iou_thresholds} for conf in conf_thresholds})
    recall_df = pd.DataFrame({conf: {iou: results[iou][conf]["recall"] for iou in iou_thresholds} for conf in conf_thresholds})
    f1_df = pd.DataFrame({conf: {iou: results[iou][conf]["f1"] for iou in iou_thresholds} for conf in conf_thresholds})
    
    return precision_df, recall_df, f1_df, precision_default, recall_default
