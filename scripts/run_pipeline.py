import sys
import cv2
import json
import argparse
from pathlib import Path
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from blueprint_brain.config.settings import settings
from blueprint_brain.src.utils.pdf_converter import PDFConverter
from blueprint_brain.src.inference.engine import InferenceEngine
from blueprint_brain.src.utils.visualizer import Visualizer

def main():
    parser = argparse.ArgumentParser(description="Blueprint AI Inference Pipeline")
    parser.add_argument("--input", type=str, required=True, help="Path to PDF or Image file")
    parser.add_argument("--weights", type=str, default="yolov8n.pt", help="Path to trained .pt file")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path("output_results")
    output_dir.mkdir(exist_ok=True)

    # 1. Initialize Engine
    print(f"Loading Engine with weights: {args.weights}")
    engine = InferenceEngine(
        model_path=args.weights,
        tile_size=settings.TILE_SIZE,
        overlap=settings.TILE_OVERLAP
    )
    viz = Visualizer(settings.CLASS_MAP)

    # 2. Load Input
    images = []
    if input_path.suffix.lower() == ".pdf":
        print("Converting PDF to images...")
        try:
            images = PDFConverter.to_images(input_path)
        except Exception as e:
            print(f"Error converting PDF: {e}")
            return
    elif input_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
        img = cv2.imread(str(input_path))
        images = [img]
    else:
        print("Unsupported format")
        return

    # 3. Process each page
    for idx, img in enumerate(images):
        print(f"Processing Page {idx+1} ({img.shape[1]}x{img.shape[0]} px)...")
        
        # RUN PIPELINE
        results = engine.process_full_image(img)
        
        # 4. JSON Output
        det_count = len(results['boxes'])
        print(f"Detected {det_count} objects.")
        
        json_output = []
        for i in range(det_count):
            box = results['boxes'][i].tolist()
            cls_id = int(results['classes'][i])
            score = float(results['scores'][i])
            label = settings.CLASS_MAP.get(str(cls_id), "Unknown") # Inverse lookup in real app
            
            json_output.append({
                "label": label,
                "confidence": score,
                "bbox": box
            })
            
        json_path = output_dir / f"{input_path.stem}_page{idx+1}.json"
        with open(json_path, 'w') as f:
            json.dump(json_output, f, indent=2)

        # 5. Visual Output
        annotated_img = viz.draw_bboxes(
            img, 
            results['boxes'], 
            results['scores'], 
            results['classes']
        )
        
        out_img_path = output_dir / f"{input_path.stem}_page{idx+1}_analyzed.jpg"
        cv2.imwrite(str(out_img_path), annotated_img)
        print(f"Saved results to {output_dir}")

if __name__ == "__main__":
    main()