import cv2
import numpy as np
import logging
from typing import List, Tuple, Optional
from shapely.geometry import Polygon, Point
from shapely.validation import make_valid

from blueprint_brain.src.core.exceptions import GeometryError

logger = logging.getLogger(__name__)

class GeometryUtils:
    """
    High-performance geometric operations.
    """

    @staticmethod
    def mask_to_polygons(binary_mask: np.ndarray, min_area: int = 500) -> List[Polygon]:
        """
        Robustly converts binary masks to validated Shapely polygons.
        """
        try:
            # Find Contours
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            polygons = []
            for cnt in contours:
                if cv2.contourArea(cnt) < min_area:
                    continue
                    
                # Simplify (RDP algorithm)
                epsilon = 0.005 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                
                if len(approx) < 3: continue

                points = approx.reshape(-1, 2)
                poly = Polygon(points)

                # Defensive Geometry: Fix bowties/self-intersections
                if not poly.is_valid:
                    poly = make_valid(poly)
                    # make_valid might return MultiPolygon, take largest
                    if poly.geom_type == 'MultiPolygon':
                        poly = max(poly.geoms, key=lambda p: p.area)
                
                polygons.append(poly)
                    
            return polygons
        except Exception as e:
            raise GeometryError(f"Mask to Polygon conversion failed: {e}")

    @staticmethod
    def match_points_to_polygons(points_with_data: List[Tuple[Point, dict]], 
                                 polygons: List[Polygon]) -> List[dict]:
        """
        Uses Spatial Index (STRtree) to map M points to N polygons efficiently.
        Returns: List of dicts where data is enriched with 'polygon_index'.
        """
        from shapely.strtree import STRtree
        
        if not polygons or not points_with_data:
            return points_with_data

        # Build Tree
        tree = STRtree(polygons)
        
        for pt, data in points_with_data:
            # Query tree for candidate polygons (bounding box intersection)
            query_geom = pt
            candidate_indices = tree.query(query_geom)
            
            # Precise check
            found_idx = None
            for idx in candidate_indices:
                if polygons[idx].contains(pt):
                    found_idx = int(idx)
                    break
            
            data['container_index'] = found_idx
            
        return [d for _, d in points_with_data]