import os
import yaml
from pathlib import Path
from ultralytics import YOLO
import logging
from typing import Dict, Any

# Configure Logging
logger = logging.getLogger("BlueprintDetector")
logging.basicConfig(level=logging.INFO)

class BlueprintDetector:
    """
    Wrapper for YOLOv8 Object Detection Model.
    Handles configuration generation, training, and inference.
    """

    def __init__(self, model_version: str = "yolov8n.pt"):
        """
        Args:
            model_version: 'yolov8n.pt' (nano) for speed, 'yolov8x.pt' (extra large) for accuracy.
        """
        self.model_name = model_version
        self.model = YOLO(model_version)
        logger.info(f"Initialized YOLO model: {model_version}")

    def _generate_data_yaml(self, data_path: Path, class_map: Dict[str, int]) -> Path:
        """
        Generates the 'data.yaml' file required by YOLO.
        """
        # Ensure absolute paths
        abs_path = data_path.resolve()
        
        config = {
            'path': str(abs_path),
            'train': 'images', # Ultralytics looks for images/ relative to path
            'val': 'images',   # Using same set for demo (Use 'val' folder in prod)
            'names': {v: k for k, v in class_map.items()}
        }
        
        yaml_path = data_path / "dataset.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f)
            
        return yaml_path

    def train(self, 
              data_path: Path, 
              class_map: Dict[str, int], 
              epochs: int = 50, 
              img_size: int = 640,
              batch_size: int = 16):
        """
        Starts the training process.
        """
        logger.info("Generating dataset configuration...")
        yaml_path = self._generate_data_yaml(data_path, class_map)
        
        logger.info(f"Starting training for {epochs} epochs...")
        results = self.model.train(
            data=str(yaml_path),
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            plots=True,
            device='0' if self._has_gpu() else 'cpu',
            project='blueprint_brain_runs',
            name='v1_baseline'
        )
        logger.info("Training complete.")
        return results

    def predict(self, image_path: str, conf_threshold: float = 0.25) -> Any:
        """
        Runs inference on a single image.
        """
        results = self.model.predict(
            source=image_path,
            conf=conf_threshold,
            save=False,
            verbose=False
        )
        return results[0] # Return the first result object

    def export(self, format: str = 'onnx'):
        """Exports the model for production deployment."""
        self.model.export(format=format)

    @staticmethod
    def _has_gpu():
        import torch
        return torch.cuda.is_available() or torch.backends.mps.is_available()