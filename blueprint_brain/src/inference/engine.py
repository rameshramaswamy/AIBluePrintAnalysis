import torch
import numpy as np
import logging
import math
from typing import Dict, Any, List
from blueprint_brain.config.settings import settings
from blueprint_brain.src.models.detector import BlueprintDetector
from blueprint_brain.src.utils.postprocessing import PredictionMerger
from blueprint_brain.src.core.exceptions import ModelInferenceError

logger = logging.getLogger(__name__)

class InferenceEngine:
    _instance = None
    
    def __init__(self, model_path: str = None, batch_size: int = 16):
        self.model_path = model_path or settings.DEFAULT_MODEL_VERSION
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.detector = None
        self.merger = PredictionMerger()
        self.batch_size = batch_size
        
        self.tile_size = settings.TILE_SIZE
        self.stride = int(self.tile_size * (1 - settings.TILE_OVERLAP))

    def _load_model(self):
        if self.detector is None:
            self.detector = BlueprintDetector(model_version=self.model_path)

    def process_full_image(self, image: np.ndarray) -> Dict[str, Any]:
        self._load_model()
        h_img, w_img = image.shape[:2]
        
        # 1. Generate All Tile Coordinates
        tile_coords = []
        for y in range(0, h_img, self.stride):
            for x in range(0, w_img, self.stride):
                x_end = min(x + self.tile_size, w_img)
                y_end = min(y + self.tile_size, h_img)
                x_start = x_end - self.tile_size
                y_start = y_end - self.tile_size
                if x_start >= 0 and y_start >= 0:
                    tile_coords.append((x_start, y_start, x_end, y_end))

        # 2. Batch Inference Loop
        global_boxes, global_scores, global_classes = [], [], []
        
        total_batches = math.ceil(len(tile_coords) / self.batch_size)
        logger.info(f"Inference: {len(tile_coords)} tiles in {total_batches} batches.")

        for i in range(0, len(tile_coords), self.batch_size):
            batch_coords = tile_coords[i : i + self.batch_size]
            batch_imgs = []
            
            # Prepare Batch
            for (x1, y1, x2, y2) in batch_coords:
                batch_imgs.append(image[y1:y2, x1:x2])
            
            # Predict Batch (One GPU Call)
            results = self.detector.predict_batch(
                batch_imgs, 
                conf_threshold=settings.CONFIDENCE_THRESHOLD
            )
            
            # Parse Results
            for j, res in enumerate(results):
                if len(res.boxes) == 0: continue
                
                # Get offset from coordinates
                x_start, y_start, _, _ = batch_coords[j]
                
                boxes = res.boxes.xyxy.cpu().numpy()
                scores = res.boxes.conf.cpu().numpy()
                classes = res.boxes.cls.cpu().numpy()
                
                # Shift coordinates to global
                boxes[:, [0, 2]] += x_start
                boxes[:, [1, 3]] += y_start
                
                global_boxes.extend(boxes)
                global_scores.extend(scores)
                global_classes.extend(classes)

        # 3. WBF Merge
        return self.merger.merge_detections(global_boxes, global_scores, global_classes)