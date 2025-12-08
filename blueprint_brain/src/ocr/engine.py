import numpy as np
import logging
from typing import List, Optional
from pydantic import BaseModel, Field
from paddleocr import PaddleOCR
from shapely.geometry import Point

from blueprint_brain.config.settings import settings
from blueprint_brain.src.core.exceptions import OCRAnalysisError

logger = logging.getLogger(__name__)

class TextEntity(BaseModel):
    """Standardized Text Object"""
    text: str
    confidence: float
    bbox: List[int] = Field(..., description="[x1, y1, x2, y2]")
    center: List[float] = Field(..., description="[cx, cy]")

    @property
    def center_point(self) -> Point:
        return Point(self.center[0], self.center[1])

class OCREngine:
    """
    Thread-safe Singleton wrapper for PaddleOCR.
    """
    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(OCREngine, cls).__new__(cls)
        return cls._instance

    def _get_model(self):
        """Lazy loader for the heavy OCR model"""
        if self._model is None:
            try:
                logger.info(f"Loading OCR Model (Mode: {'GPU' if settings.USE_AMP else 'CPU'})...")
                # use_angle_cls=True is vital for rotated architectural text
                self._model = PaddleOCR(
                    use_angle_cls=True, 
                    lang='en', 
                    show_log=False,
                    use_gpu=False # Set True if Docker container has CUDA
                )
            except Exception as e:
                raise OCRAnalysisError(f"Failed to initialize PaddleOCR: {e}")
        return self._model

    def analyze_image(self, image: np.ndarray) -> List[TextEntity]:
        """
        Extracts text with high-performance error handling.
        """
        if image is None or image.size == 0:
            logger.warning("OCR received empty image.")
            return []

        model = self._get_model()
        
        try:
            # cls=True enables orientation classification (0, 90, 180, 270)
            result = model.ocr(image, cls=True)
            
            entities = []
            # PaddleOCR returns None if no text found
            if not result or result[0] is None:
                return []

            for line in result[0]:
                coords, (text_str, confidence) = line
                
                # Robust Coordinate Parsing
                xs = [pt[0] for pt in coords]
                ys = [pt[1] for pt in coords]
                x1, x2 = min(xs), max(xs)
                y1, y2 = min(ys), max(ys)
                
                cx = x1 + (x2 - x1) / 2
                cy = y1 + (y2 - y1) / 2

                # Filter low confidence noise
                if confidence < settings.CONFIDENCE_THRESHOLD:
                    continue

                entities.append(TextEntity(
                    text=text_str.strip(),
                    confidence=confidence,
                    bbox=[int(x1), int(y1), int(x2), int(y2)],
                    center=[cx, cy]
                ))
            
            logger.debug(f"OCR extracted {len(entities)} text entities.")
            return entities

        except Exception as e:
            logger.error(f"OCR Inference Failed: {e}")
            raise OCRAnalysisError(f"OCR processing crashed: {e}")