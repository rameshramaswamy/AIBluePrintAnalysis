import cv2
import numpy as np
from typing import List, Tuple, Dict
import random

class Visualizer:
    """
    Helper class to draw bounding boxes and annotations on architectural drawings.
    """
    
    def __init__(self, class_map: Dict[str, int]):
        self.class_map = class_map
        # Reverse map: ID -> Name
        self.id_to_name = {v: k for k, v in class_map.items()}
        # Generate random colors for each class
        self.colors = {v: [random.randint(0, 255) for _ in range(3)] for v in class_map.values()}

    def draw_bboxes(self, image: np.ndarray, bboxes: List[List[float]], confidences: List[float] = None, class_ids: List[int] = None) -> np.ndarray:
        """
        Draws bounding boxes on the image.
        bboxes format: [x1, y1, x2, y2] (pixel coordinates, not normalized)
        """
        img_copy = image.copy()
        
        if class_ids is None:
            class_ids = [0] * len(bboxes)
        
        for idx, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = map(int, bbox)
            cls_id = int(class_ids[idx])
            color = self.colors.get(cls_id, (0, 255, 0))
            
            # Draw Rectangle
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
            
            # Label
            label = self.id_to_name.get(cls_id, f"Class {cls_id}")
            if confidences:
                label += f" {confidences[idx]:.2f}"
                
            # Text Background
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img_copy, (x1, y1 - 20), (x1 + w, y1), color, -1)
            cv2.putText(img_copy, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
        return img_copy

    @staticmethod
    def yolo_to_pixels(yolo_box: List[float], img_w: int, img_h: int) -> List[float]:
        """Convert YOLO [cx, cy, w, h] normalized to [x1, y1, x2, y2] pixels"""
        cx, cy, w, h = yolo_box
        x1 = int((cx - w/2) * img_w)
        y1 = int((cy - h/2) * img_h)
        x2 = int((cx + w/2) * img_w)
        y2 = int((cy + h/2) * img_h)
        return [x1, y1, x2, y2]