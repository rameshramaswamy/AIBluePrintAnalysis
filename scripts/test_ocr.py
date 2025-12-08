import sys
import cv2
import argparse
from pathlib import Path

# Add project root
sys.path.append(str(Path(__file__).parent.parent))

from blueprint_brain.src.ocr.engine import OCREngine
from blueprint_brain.src.ocr.cleaner import TextCleaner, TextType

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to blueprint image")
    args = parser.parse_args()
    
    img_path = Path(args.image)
    if not img_path.exists():
        print("Image not found")
        return

    # 1. Load Image
    img = cv2.imread(str(img_path))
    
    # 2. Run OCR
    engine = OCREngine()
    print("Running OCR (this may take a moment for the first run)...")
    entities = engine.analyze_image(img)
    
    print(f"Found {len(entities)} text entities.")
    
    # 3. Visualize & Classify
    for ent in entities:
        t_type = TextCleaner.classify_text(ent.text)
        
        color = (0, 0, 255) # Red (Noise)
        if t_type == TextType.ROOM_LABEL:
            color = (0, 255, 0) # Green
        elif t_type == TextType.DIMENSION:
            color = (255, 0, 0) # Blue
        elif t_type == TextType.SCALE_MARKER:
            color = (0, 255, 255) # Yellow

        # Draw box
        x1, y1, x2, y2 = ent.bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Draw Text
        label = f"{ent.text} ({t_type.value})"
        cv2.putText(img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        print(f"[{t_type.value.upper()}] {ent.text} (Conf: {ent.confidence:.2f})")

    # Save Output
    out_name = f"ocr_debug_{img_path.name}"
    cv2.imwrite(out_name, img)
    print(f"Saved debug image to {out_name}")

if __name__ == "__main__":
    main()