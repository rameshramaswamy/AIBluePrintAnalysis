import os
import numpy as np
import xml.etree.ElementTree as ET
from shapely.geometry import Polygon
from typing import List, Dict, Tuple
import logging

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CubiCasaParser:
    """
    Parses CubiCasa5k SVG files into geometric polygons and bounding boxes.
    """

    def __init__(self, class_map: Dict[str, int]):
        self.class_map = class_map
        # CubiCasa often uses these SVG identifiers
        self.svg_mapping = {
            'Wall': ['OuterWall', 'Wall'],
            'Window': ['Window'],
            'Door': ['Door'],
            'Room': ['Room'],
            'Icon': ['FixedFurniture', 'Electrical'] # Simplified for example
        }

    def parse_svg(self, svg_path: str) -> Dict[str, List[Polygon]]:
        """
        Parses an SVG file and returns a dictionary of Shapely Polygons by class.
        """
        if not os.path.exists(svg_path):
            raise FileNotFoundError(f"SVG not found: {svg_path}")

        tree = ET.parse(svg_path)
        root = tree.getroot()
        
        # Namespace handling for SVG
        ns = {'svg': 'http://www.w3.org/2000/svg'}
        
        extracted_data = {k: [] for k in self.class_map.keys()}

        # Iterate through relevant groups
        for cat_name, identifiers in self.svg_mapping.items():
            if cat_name not in self.class_map:
                continue
                
            for identifier in identifiers:
                # Find all groups or paths with this ID (simplified logic)
                # In real CubiCasa, classes are often in <g id="Wall">
                elements = root.findall(f".//*[@id='{identifier}']", ns)
                
                # Also check class attributes (CubiCasa variation)
                elements += root.findall(f".//*[@class='{identifier}']", ns)

                for elem in elements:
                    # Extract polygon points
                    polys = elem.findall('.//svg:polygon', ns)
                    for poly in polys:
                        points_str = poly.get('points')
                        if points_str:
                            points = self._str_to_points(points_str)
                            if len(points) >= 3:
                                extracted_data[cat_name].append(Polygon(points))
                                
        return extracted_data

    def _str_to_points(self, points_str: str) -> List[Tuple[float, float]]:
        """Converts SVG point string 'x1,y1 x2,y2' to list of tuples."""
        try:
            points = []
            for pair in points_str.strip().split(' '):
                x, y = map(float, pair.split(','))
                points.append((x, y))
            return points
        except ValueError:
            return []

    def get_bboxes_yolo(self, polygons: List[Polygon], img_width: int, img_height: int) -> List[Tuple]:
        """
        Converts polygons to YOLO Bounding Box format: [x_center, y_center, width, height] normalized.
        """
        bboxes = []
        for poly in polygons:
            minx, miny, maxx, maxy = poly.bounds
            
            # Normalization
            dw = 1. / img_width
            dh = 1. / img_height
            
            w = maxx - minx
            h = maxy - miny
            x = minx + (w / 2)
            y = miny + (h / 2)
            
            bboxes.append((x * dw, y * dh, w * dw, h * dh))
            
        return bboxes