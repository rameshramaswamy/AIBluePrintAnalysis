import numpy as np
import logging
from typing import List, Dict, Any
from shapely.geometry import Point

from blueprint_brain.src.ocr.engine import TextEntity
from blueprint_brain.src.ocr.cleaner import TextCleaner, TextType
from blueprint_brain.src.logic.geometry import GeometryUtils
from blueprint_brain.src.logic.scale import ScaleEngine
from blueprint_brain.src.core.exceptions import LogicFusionError

logger = logging.getLogger(__name__)

class FusionAssembler:
    """
    Orchestrator for merging Vision and Logic.
    """
    
    def __init__(self, scale_value: float = 10.0):
        # In production, scale_value might come from user input or metadata
        self.geo = GeometryUtils()
        self.scale = ScaleEngine(pixels_per_foot=scale_value)

    def assemble_floorplan(self, 
                           image_shape: tuple,
                           room_mask: np.ndarray, 
                           detections: List[Dict], 
                           ocr_results: List[TextEntity]) -> Dict[str, Any]:
        try:
            h, w = image_shape[:2]
            
            # 1. Convert Mask to Polygons
            room_polys = self.geo.mask_to_polygons(room_mask)
            logger.info(f"Fusion: Processed {len(room_polys)} room polygons.")

            # 2. Prepare Data for Spatial Indexing
            # We want to map Text Labels AND Detected Objects to Rooms
            
            # A. Prepare Text
            room_label_points = []
            for txt in ocr_results:
                if TextCleaner.classify_text(txt.text) == TextType.ROOM_LABEL:
                    # Store tuple (Point, DataDict)
                    room_label_points.append((txt.center_point, {'entity': txt}))

            # B. Prepare Objects (Doors, etc.)
            object_points = []
            for det in detections:
                # Calc center
                bbox = det['bbox']
                cx = (bbox[0] + bbox[2]) / 2
                cy = (bbox[1] + bbox[3]) / 2
                object_points.append((Point(cx, cy), {'det': det}))

            # 3. Spatial Matching (Fast)
            # Map Labels -> Rooms
            mapped_labels = self.geo.match_points_to_polygons(room_label_points, room_polys)
            
            # Map Objects -> Rooms
            mapped_objects = self.geo.match_points_to_polygons(object_points, room_polys)

            # 4. Construct Output Structure
            rooms_data = []
            
            for idx, poly in enumerate(room_polys):
                # Get Labels for this room index
                labels = [d['entity'] for d in mapped_labels if d.get('container_index') == idx]
                
                # Pick best label (Highest confidence)
                if labels:
                    best_label = max(labels, key=lambda x: x.confidence)
                    room_name = best_label.text
                    conf = best_label.confidence
                else:
                    room_name = f"Room {idx+1}"
                    conf = 0.0

                # Get Objects for this room index
                objs = [d['det']['label'] for d in mapped_objects if d.get('container_index') == idx]

                # Area
                area_sqft = self.scale.calculate_area_sqft(poly.area)

                rooms_data.append({
                    "id": f"room_{idx}",
                    "label": room_name,
                    "confidence": conf,
                    "area_sqft": area_sqft,
                    "objects": objs,
                    "polygon": list(poly.exterior.coords)
                })

            # 5. Metadata
            total_sqft = sum(r['area_sqft'] or 0 for r in rooms_data)
            
            return {
                "status": "success",
                "meta": {
                    "image_size": [w, h],
                    "total_sqft": round(total_sqft, 2),
                    "room_count": len(rooms_data)
                },
                "data": rooms_data
            }

        except Exception as e:
            logger.error(f"Fusion assembly failed: {e}")
            raise LogicFusionError(f"Critical failure in logic layer: {e}")