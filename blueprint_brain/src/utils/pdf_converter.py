import numpy as np
import cv2
from pdf2image import convert_from_path
from pathlib import Path
from typing import List

class PDFConverter:
    """
    Converts PDF documents to a list of OpenCV images (numpy arrays).
    """
    
    @staticmethod
    def to_images(pdf_path: Path, dpi: int = 200) -> List[np.ndarray]:
        """
        Convert PDF to list of numpy arrays (BGR format for OpenCV).
        High DPI (200-300) is crucial for small details in blueprints.
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
            
        # Convert PDF to PIL Images
        try:
            pil_images = convert_from_path(str(pdf_path), dpi=dpi)
        except Exception as e:
            raise RuntimeError(f"Failed to convert PDF. Is poppler installed? Error: {e}")

        opencv_images = []
        for pil_img in pil_images:
            # Convert PIL RGB to OpenCV BGR
            open_cv_image = np.array(pil_img) 
            open_cv_image = open_cv_image[:, :, ::-1].copy() 
            opencv_images.append(open_cv_image)
            
        return opencv_images