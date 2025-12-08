class BlueprintAIException(Exception):
    """Base exception for the platform"""
    pass

class DataIngestionError(BlueprintAIException):
    """Raised when input files (PDF/SVG) are corrupted or missing"""
    pass

class ProcessingError(BlueprintAIException):
    """Raised during the tiling or image manipulation phase"""
    pass

class ModelInferenceError(BlueprintAIException):
    """Raised when YOLO/PyTorch fails during prediction"""
    pass

class OCRAnalysisError(BlueprintAIException):
    """Raised when OCR model fails or returns malformed data"""
    pass

class GeometryError(BlueprintAIException):
    """Raised when polygon conversion or spatial operations fail"""
    pass

class ScaleCalibrationError(BlueprintAIException):
    """Raised when scale cannot be determined or calculated area is physically impossible"""
    pass