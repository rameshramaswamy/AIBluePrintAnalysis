import cv2
import numpy as np
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

class AdaptiveSlicer:
    """
    Optimizes inference by only generating tiles for areas with content.
    Uses contour detection and entropy filtering to skip empty whitespace.
    """
    
    @staticmethod
    def get_roi_tiles(image: np.ndarray, 
                      tile_size: int = 640, 
                      overlap: float = 0.2) -> List[Tuple[int, int, int, int]]:
        """
        Returns list of (x1, y1, x2, y2) tuples for tiles that contain actual data.
        """
        h_img, w_img = image.shape[:2]
        stride = int(tile_size * (1 - overlap))
        
        # 1. Create a "Content Mask" (Downscaled for speed)
        # We shrink image to 1/10th size to quickly find ink density
        scale = 0.1
        small_h, small_w = int(h_img * scale), int(w_img * scale)
        if small_h == 0 or small_w == 0: return [] # Too small

        small_img = cv2.resize(image, (small_w, small_h))
        gray = cv2.cvtColor(small_img, cv2.COLOR_BGR2GRAY)
        
        # Invert: Ink (black) becomes bright, paper (white) becomes dark
        # Adaptive threshold handles varying lighting/scan quality
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY_INV, 11, 2)
        
        # Dilate to connect broken lines into solid blocks of "content"
        kernel = np.ones((5,5), np.uint8)
        dilated = cv2.dilate(binary, kernel, iterations=2)
        
        tile_coords = []
        skipped_count = 0

        # 2. Iterate Grid
        for y in range(0, h_img, stride):
            for x in range(0, w_img, stride):
                x_end = min(x + tile_size, w_img)
                y_end = min(y + tile_size, h_img)
                x_start = x_end - tile_size
                y_start = y_end - tile_size
                
                if x_start < 0 or y_start < 0: continue
                
                # 3. Check Content Mask
                # Map coordinates to the small mask
                sx1, sy1 = int(x_start * scale), int(y_start * scale)
                sx2, sy2 = int(x_end * scale), int(y_end * scale)
                
                # Check pixel density in the mask region
                mask_roi = dilated[sy1:sy2, sx1:sx2]
                if mask_roi.size == 0: continue
                
                # If > 1% of the tile has ink, keep it. Else skip.
                ink_ratio = np.count_nonzero(mask_roi) / mask_roi.size
                
                if ink_ratio > 0.01: 
                    tile_coords.append((x_start, y_start, x_end, y_end))
                else:
                    skipped_count += 1

        logger.info(f"Adaptive Slicer: Kept {len(tile_coords)} tiles, Skipped {skipped_count} empty tiles.")
        return tile_coords