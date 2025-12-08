import sys
import cv2
import glob
from pathlib import Path
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent))

from blueprint_brain.config.settings import settings
from blueprint_brain.src.models.detector import BlueprintDetector
from blueprint_brain.src.utils.visualizer import Visualizer

def main():
    # 1. Load Trained Model
    # After training, YOLO saves weights in runs/detect/train/weights/best.pt
    # For now, we load the base model if training hasn't happened, or point to trained weights
    weights_path = Path("blueprint_brain_runs/v1_baseline/weights/best.pt")
    
    if weights_path.exists():
        print(f"Loading trained weights from {weights_path}")
        model = BlueprintDetector(model_version=str(weights_path))
    else:
        print("Trained weights not found. Loading pre-trained base model (predictions will be random/coco classes).")
        model = BlueprintDetector(model_version="yolov8n.pt")

    viz = Visualizer(settings.CLASS_MAP)

    # 2. Get a Test Image (A tile from our processed data)
    test_images = list(settings.PROCESSED_PATH.glob("images/*.jpg"))
    if not test_images:
        print("No test images found.")
        return

    # Pick the first 3 images to test
    for img_path in test_images[:3]:
        print(f"Predicting on {img_path.name}...")
        
        # 3. Run Inference
        result = model.predict(str(img_path), conf_threshold=0.3)
        
        # 4. Extract Boxes
        boxes = result.boxes.xyxy.cpu().numpy() # [x1, y1, x2, y2]
        confs = result.boxes.conf.cpu().numpy()
        cls_ids = result.boxes.cls.cpu().numpy().astype(int)
        
        # 5. Visualize
        img = cv2.imread(str(img_path))
        annotated_img = viz.draw_bboxes(img, boxes, confs, cls_ids)
        
        # Save output
        output_name = f"pred_{img_path.name}"
        cv2.imwrite(output_name, annotated_img)
        print(f"Saved prediction to {output_name}")

if __name__ == "__main__":
    main()