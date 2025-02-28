import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def explore_data():
    """
    Computes area statistics for solar panels.
    - Converts YOLO normalized values to absolute pixel dimensions (assuming 416x416 images),
      then converts to meters using a conversion factor (0.31).
    - Returns a histogram figure and a stats dictionary (mean, std, total instances).
    """
    label_path = os.path.join("dataset", "labels", "labels_native")
    
    areas = []
    for file in os.listdir(label_path):
        if file.endswith(".txt"):
            with open(os.path.join(label_path, file), 'r') as f:
                data = f.readlines()
                for info in data:
                    parts = info.strip().split()
                    if len(parts) == 5:
                        # YOLO format: class, x_center, y_center, width, height (normalized)
                        width_pix = float(parts[3]) * 416
                        height_pix = float(parts[4]) * 416
                        width_m = width_pix * 0.31
                        height_m = height_pix * 0.31
                        area = width_m * height_m
                        areas.append(area)
    
    areas = np.array(areas)
    mean_area = np.mean(areas) if len(areas) > 0 else 0
    std_area = np.std(areas) if len(areas) > 0 else 0
    total_instances = len(areas)
    
    fig = plt.figure(figsize=(10, 6))
    plt.hist(areas, bins=30, color='skyblue', edgecolor='black')
    plt.xlabel('Area (m²)')
    plt.ylabel('Instances')
    plt.title('Histogram of Solar Panel Areas')
    plt.grid(True, alpha=0.3)
    
    stats = {"mean_area": mean_area, "std_area": std_area, "total_instances": total_instances}
    return fig, stats

def value_counts_labels():
    """
    Computes the value counts of labels per image.
    Returns a dictionary where the keys are the number of labels in an image and
    the values are the number of images with that count.
    """
    label_path = os.path.join("dataset", "labels", "labels_native")
    labels_count = {}
    for file in os.listdir(label_path):
        if file.endswith(".txt"):
            with open(os.path.join(label_path, file), 'r') as f:
                data = f.readlines()
                count = len(data)
                labels_count[file] = count
    value_counts = {}
    for count in labels_count.values():
        value_counts[count] = value_counts.get(count, 0) + 1
    return value_counts
