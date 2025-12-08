import re
import logging
from typing import Optional

from blueprint_brain.src.core.exceptions import ScaleCalibrationError

logger = logging.getLogger(__name__)

class ScaleEngine:
    """
    Manages pixel-to-foot conversions with sanity checks.
    """

    def __init__(self, pixels_per_foot: Optional[float] = None):
        self.pixels_per_foot = pixels_per_foot
        self.min_sanity_sqft = 5.0   # Smallest reasonable room (closet)
        self.max_sanity_sqft = 10000.0 # Largest reasonable room (hall)

    def set_scale(self, pixels_per_foot: float):
        if pixels_per_foot <= 0:
            raise ScaleCalibrationError("Scale must be positive")
        self.pixels_per_foot = pixels_per_foot
        logger.info(f"Scale calibrated: {self.pixels_per_foot:.2f} px/ft")

    def calculate_area_sqft(self, pixel_area: float) -> Optional[float]:
        if not self.pixels_per_foot:
            return None
        
        sqft = pixel_area / (self.pixels_per_foot ** 2)
        
        # Sanity Check (Logging only, don't crash)
        if sqft > self.max_sanity_sqft:
            logger.warning(f"Calculated area {sqft:.2f} sqft exceeds sanity limit. Scale might be wrong.")
        
        return round(sqft, 2)