import cv2

def draw_boxes(image, boxes, color, label):
    """
    Draws bounding boxes on an image.
    - boxes: list or array of [x_min, y_min, x_max, y_max]
    - color: (B, G, R) tuple for the rectangle
    - label: text label to put near each box.
    """
    for box in boxes:
        x_min, y_min, x_max, y_max = map(int, box)
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
        cv2.putText(image, label, (x_min, max(y_min - 5, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def format_change(box):
    """
    Converts a box from YOLO center format to [x_min, y_min, x_max, y_max].
    """
    x_min = box[0] - box[2] / 2
    y_min = box[1] - box[3] / 2
    x_max = box[0] + box[2] / 2
    y_max = box[1] + box[3] / 2
    return [x_min, y_min, x_max, y_max]
