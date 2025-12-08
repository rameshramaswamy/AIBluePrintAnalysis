import numpy as np
import torch
import torchvision.ops as ops
from typing import List, Dict

class PredictionMerger:
    """
    Enterprise-grade merging using Weighted Box Fusion (WBF) strategy.
    WBF yields better accuracy for tiled inference than standard NMS.
    """

    @staticmethod
    def merge_detections(all_boxes: List[List[float]], 
                         all_scores: List[float], 
                         all_classes: List[int], 
                         iou_threshold: float = 0.5) -> Dict[str, np.ndarray]:
        
        if not all_boxes:
            return {'boxes': np.array([]), 'scores': np.array([]), 'classes': np.array([])}

        boxes = np.array(all_boxes)
        scores = np.array(all_scores)
        labels = np.array(all_classes)

        # 1. Separate by class (WBF must be per-class)
        final_boxes, final_scores, final_labels = [], [], []
        unique_labels = np.unique(labels)

        for label in unique_labels:
            idxs = np.where(labels == label)[0]
            cls_boxes = boxes[idxs]
            cls_scores = scores[idxs]

            # 2. Cluster boxes based on IoU
            # We use a simplified WBF approach: 
            # If boxes overlap significantly, average their coordinates weighted by confidence.
            
            # Sort by score desc
            order = cls_scores.argsort()[::-1]
            cls_boxes = cls_boxes[order]
            cls_scores = cls_scores[order]
            
            keep_boxes = []
            keep_scores = []
            
            while len(cls_boxes) > 0:
                # Take the highest confidence box
                current_box = cls_boxes[0]
                current_score = cls_scores[0]
                
                # Find overlaps
                ious = PredictionMerger._compute_iou(current_box, cls_boxes)
                mask = ious > iou_threshold
                
                # WBF Logic: Average coordinates of matching boxes
                matching_boxes = cls_boxes[mask]
                matching_scores = cls_scores[mask]
                
                # Weighted Average
                w_sum = np.sum(matching_scores)
                weighted_box = np.sum(matching_boxes * matching_scores[:, None], axis=0) / w_sum
                
                # Update score (can use max, avg, or boosted)
                # We use avg score for stability, or min(1.0, avg + boost)
                new_score = np.mean(matching_scores)

                keep_boxes.append(weighted_box)
                keep_scores.append(new_score)
                
                # Remove processed boxes
                cls_boxes = cls_boxes[~mask]
                cls_scores = cls_scores[~mask]

            # Add to final list
            final_boxes.extend(keep_boxes)
            final_scores.extend(keep_scores)
            final_labels.extend([label] * len(keep_boxes))

        return {
            'boxes': np.array(final_boxes),
            'scores': np.array(final_scores),
            'classes': np.array(final_labels)
        }

    @staticmethod
    def _compute_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """Numpy vectorised IoU calculation"""
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])
        
        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        area_box = (box[2] - box[0]) * (box[3] - box[1])
        area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        
        union = area_box + area_boxes - intersection
        return intersection / (union + 1e-6)