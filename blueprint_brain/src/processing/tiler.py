import cv2
import numpy as np
import logging
from shapely.geometry import box, Polygon
from pathlib import Path
from typing import List, Dict, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

from blueprint_brain.config.settings import settings
from blueprint_brain.src.core.exceptions import ProcessingError

logger = logging.getLogger(__name__)

class ImageTiler:
    """
    Enterprise-grade image slicer with multiprocessing support.
    """
    
    def __init__(self, tile_size: int = 640, overlap: float = 0.2):
        self.tile_size = tile_size
        self.stride = int(tile_size * (1 - overlap))

    def _process_single_image(self, task_payload: Dict) -> str:
        """
        Worker function designed to run in a separate process.
        Returns: Status string
        """
        try:
            image_path = task_payload['image_path']
            polygons_by_class = task_payload['polygons']
            output_dir = task_payload['output_dir']
            class_map = task_payload['class_map']
            base_filename = task_payload['base_filename']

            img = cv2.imread(str(image_path))
            if img is None:
                raise ProcessingError(f"Could not read image: {image_path}")

            h_img, w_img, _ = img.shape
            tile_idx = 0
            
            # Pre-calculate labels to avoid I/O in loop if possible, 
            # but for YOLO we usually write files immediately.
            
            for y in range(0, h_img, self.stride):
                for x in range(0, w_img, self.stride):
                    x_end = min(x + self.tile_size, w_img)
                    y_end = min(y + self.tile_size, h_img)
                    x_start = x_end - self.tile_size
                    y_start = y_end - self.tile_size
                    
                    if x_start < 0 or y_start < 0: continue

                    # Crop & Geometry
                    tile_img = img[y_start:y_end, x_start:x_end]
                    tile_poly = box(x_start, y_start, x_end, y_end)
                    
                    yolo_labels = []
                    
                    for cls_name, polys in polygons_by_class.items():
                        if cls_name not in class_map: continue
                        cls_id = class_map[cls_name]
                        
                        for poly in polys:
                            if not tile_poly.intersects(poly): continue
                            intersection = tile_poly.intersection(poly)
                            
                            # Filter small artifacts (<5% of object or tiny area)
                            if intersection.area < 50: continue

                            # Normalize Coordinates
                            minx, miny, maxx, maxy = intersection.bounds
                            w = maxx - minx
                            h = maxy - miny
                            cx = minx + w/2 - x_start
                            cy = miny + h/2 - y_start
                            
                            # YOLO Format: class cx cy w h (normalized)
                            label_line = (
                                f"{cls_id} "
                                f"{np.clip(cx/self.tile_size, 0, 1):.6f} "
                                f"{np.clip(cy/self.tile_size, 0, 1):.6f} "
                                f"{np.clip(w/self.tile_size, 0, 1):.6f} "
                                f"{np.clip(h/self.tile_size, 0, 1):.6f}"
                            )
                            yolo_labels.append(label_line)

                    if yolo_labels:
                        t_name = f"{base_filename}_{tile_idx}"
                        cv2.imwrite(str(output_dir / "images" / f"{t_name}.jpg"), tile_img)
                        with open(output_dir / "labels" / f"{t_name}.txt", 'w') as f:
                            f.write("\n".join(yolo_labels))
                        tile_idx += 1
            
            return f"Success: {base_filename} generated {tile_idx} tiles"

        except Exception as e:
            logger.error(f"Failed to process {task_payload.get('base_filename', 'unknown')}: {str(e)}")
            return f"Error: {str(e)}"

    def process_batch(self, tasks: List[Dict]):
        """
        Executes tiling in parallel using ProcessPoolExecutor.
        """
        logger.info(f"Starting batch processing with {settings.NUM_WORKERS} workers...")
        
        with ProcessPoolExecutor(max_workers=settings.NUM_WORKERS) as executor:
            futures = [executor.submit(self._process_single_image, task) for task in tasks]
            
            for future in as_completed(futures):
                result = future.result()
                # meaningful logging for monitoring
                if "Error" in result:
                    logger.warning(result)