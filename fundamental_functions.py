import os
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import supervision
from shapely.geometry import Polygon

def generate_test_data(num_images=10, box_size=20, img_size=100, num_boxes=10):
    """
    Randomly generates test data:
    - num_images: number of synthetic images.
    - Each image gets num_boxes ground truth and predicted boxes (with random confidence scores).
    Returns lists for ground truth boxes, predicted boxes, and predicted scores.
    """
    gt_data = []
    pred_data = []
    pred_scores_data = []
    min_cen = box_size // 2
    max_cen = img_size - (box_size // 2)

    for i in range(num_images):
        gt_boxes = []
        pred_boxes = []
        pred_boxes_scores = []
        for j in range(num_boxes):
            x_cen = random.randint(min_cen, max_cen)
            y_cen = random.randint(min_cen, max_cen)
            gt_boxes.append([x_cen, y_cen, box_size, box_size])
        for j in range(num_boxes):
            x_cen = random.randint(min_cen, max_cen)
            y_cen = random.randint(min_cen, max_cen)
            pred_boxes.append([x_cen, y_cen, box_size, box_size])
            pred_boxes_scores.append(random.random())
        gt_data.append(gt_boxes)
        pred_data.append(pred_boxes)
        pred_scores_data.append(pred_boxes_scores)
    return gt_data, pred_data, pred_scores_data

def compute_iou(box1, box2):
    """
    Computes Intersection over Union (IoU) between two boxes.
    Boxes are in YOLO format: [x_center, y_center, width, height].
    Uses the shapely library.
    """
    x1_cen, y1_cen, w1, h1 = box1
    x2_cen, y2_cen, w2, h2 = box2
    
    x1_min = x1_cen - w1 / 2
    x1_max = x1_cen + w1 / 2
    y1_min = y1_cen - h1 / 2
    y1_max = y1_cen + h1 / 2
    
    x2_min = x2_cen - w2 / 2
    x2_max = x2_cen + w2 / 2
    y2_min = y2_cen - h2 / 2
    y2_max = y2_cen + h2 / 2
    
    poly1 = Polygon([(x1_min, y1_min), (x1_max, y1_min), (x1_max, y1_max), (x1_min, y1_max)])
    poly2 = Polygon([(x2_min, y2_min), (x2_max, y2_min), (x2_max, y2_max), (x2_min, y2_max)])
    inter_area = poly1.intersection(poly2).area
    union_area = poly1.union(poly2).area
    return inter_area / union_area if union_area != 0 else 0

def ap_voc11(precision, recall):
    """Compute Average Precision using Pascal VOC 11-point interpolation."""
    ap = 0.0
    for t in np.arange(0.0, 1.1, 0.1):
        p = np.max(precision[recall >= t]) if np.sum(recall >= t) > 0 else 0
        ap += p / 11.0
    return ap

def ap_coco101(precision, recall):
    """Compute Average Precision using COCO 101-point interpolation."""
    ap = 0.0 
    for t in np.arange(0.0, 1.01, 0.01):
        p = np.max(precision[recall >= t]) if np.sum(recall >= t) > 0 else 0
        ap += p / 101.0
    return ap

def area_under_curve_ap(precision, recall):
    """
    Compute the Area Under the Precision-Recall Curve as AP.
    """
    mod_rec = np.concatenate(([0.0], recall, [1.0]))
    mod_prec = np.concatenate(([0.0], precision, [0.0]))
    for i in range(mod_prec.size - 1, 0, -1):
        mod_prec[i - 1] = max(mod_prec[i - 1], mod_prec[i])
    rec_chg_indices = np.where(mod_rec[1:] != mod_rec[:-1])[0]
    area = np.sum((mod_rec[rec_chg_indices + 1] - mod_rec[rec_chg_indices]) * mod_prec[rec_chg_indices + 1])
    return area

def format_change(box):
    """
    Converts a box from YOLO center format to [x_min, y_min, x_max, y_max] format.
    """
    x_min = box[0] - box[2] / 2
    y_min = box[1] - box[3] / 2
    x_max = box[0] + box[2] / 2
    y_max = box[1] + box[3] / 2
    return [x_min, y_min, x_max, y_max]

def compute_precisionRecall(gt_data, pred_data, pred_scores_data, iou_threshold=0.5):
    """
    Compute precision and recall values for object detection results.
    
    Args:
        gt_data: List of ground truth boxes per image.
        pred_data: List of prediction boxes per image.
        pred_scores_data: List of confidence scores per image.
        iou_threshold: IoU threshold to consider a match.
        
    Returns:
        precision, recall, scores arrays.
    """
    # Collect all predictions across all images
    predictions = []
    for img_idx, (gt_boxes, pred_boxes, pred_scores) in enumerate(zip(gt_data, pred_data, pred_scores_data)):
        for pred_idx, (pred_box, pred_score) in enumerate(zip(pred_boxes, pred_scores)):
            predictions.append({
                'img_idx': img_idx,
                'pred_idx': pred_idx,
                'pred_box': pred_box,
                'pred_score': pred_score
            })
    
    # Sort predictions by confidence score (highest first)
    predictions = sorted(predictions, key=lambda x: x['pred_score'], reverse=True)
    
    # Total number of ground truth boxes
    gt_boxes_total = sum(len(gt_boxes) for gt_boxes in gt_data)
    
    # Arrays to track true positives and false positives
    tp = np.zeros(len(predictions))
    fp = np.zeros(len(predictions))
    
    # Track which ground truth boxes have been matched
    visited_gt = [set() for _ in range(len(gt_data))]
    
    # Process each prediction in order of confidence
    for i, pred in enumerate(predictions):
        img_idx = pred['img_idx']
        pred_box = pred['pred_box']
        gt_boxes = gt_data[img_idx]
        
        max_iou = 0
        max_iou_gt_idx = -1
        for gt_idx, gt_box in enumerate(gt_boxes):
            if gt_idx in visited_gt[img_idx]:
                continue
            iou = compute_iou(gt_box, pred_box)
            if iou > max_iou:
                max_iou = iou
                max_iou_gt_idx = gt_idx
        if max_iou >= iou_threshold and max_iou_gt_idx != -1:
            tp[i] = 1
            visited_gt[img_idx].add(max_iou_gt_idx)
        else:
            fp[i] = 1
    
    cumulative_tp = np.cumsum(tp)
    cumulative_fp = np.cumsum(fp)
    precision = cumulative_tp / (cumulative_tp + cumulative_fp + 1e-10)
    recall = cumulative_tp / gt_boxes_total if gt_boxes_total > 0 else np.zeros_like(cumulative_tp)
    scores = [pred['pred_score'] for pred in predictions]
    
    precision = np.concatenate(([1.0], precision))
    recall = np.concatenate(([0.0], recall))
    scores = [1.0] + scores
    
    return precision, recall, scores

def compute_iou_supervision_per_image(gt_data, pred_data):
    """
    Computes IoU using the supervision library for each image.
    """
    ious_per_image = []
    for gt_boxes, pred_boxes in zip(gt_data, pred_data):
        gt_boxes_new = np.array([format_change(box) for box in gt_boxes])
        pred_boxes_new = np.array([format_change(box) for box in pred_boxes])
        sup_iou = supervision.detection.utils.box_iou_batch(gt_boxes_new, pred_boxes_new)
        ious_per_image.append(sup_iou)
    return ious_per_image

def plot_precision_recall_curve(precision, recall, ap_voc, ap_coco, ap_area):
    """
    Plots the precision-recall curve along with AP values.
    Returns a matplotlib figure.
    """
    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(10, 7))
    plt.plot(recall, precision, color='darkorange', lw=2, label='Precision-Recall Curve')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    legend_text = (
        f'VOC 11-point AP: {ap_voc:.4f}\n'
        f'COCO 101-point AP: {ap_coco:.4f}\n'
        f'Area under PR curve AP: {ap_area:.4f}'
    )
    plt.legend([legend_text], loc='lower left', fontsize=10, frameon=True)
    plt.tight_layout()
    return fig
