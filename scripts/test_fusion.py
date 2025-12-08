import sys
import numpy as np
import cv2
import json
from pathlib import Path

# Add project root
sys.path.append(str(Path(__file__).parent.parent))

from blueprint_brain.src.fusion.assembler import FusionAssembler
from blueprint_brain.src.logic.scale import ScaleEngine
from blueprint_brain.src.ocr.engine import TextEntity

def main():
    print("Testing Logic Layer (Fusion)...")

    # 1. Create Mock Data
    img_size = (1000, 1000)
    
    # Mock Mask: Create a white square (Room) on black background
    # Room is 200x200 pixels -> 40,000 px area
    mock_mask = np.zeros(img_size, dtype=np.uint8)
    cv2.rectangle(mock_mask, (100, 100), (300, 300), 255, -1) # Room 1
    cv2.rectangle(mock_mask, (400, 100), (600, 400), 255, -1) # Room 2 (Larger)

    # Mock OCR: Text inside the squares
    mock_ocr = [
        TextEntity(text="Master Bed", confidence=0.95, bbox=[120, 120, 200, 150], center=[160, 135]), # Inside Room 1
        TextEntity(text="Kitchen", confidence=0.90, bbox=[450, 150, 500, 180], center=[475, 165]),    # Inside Room 2
        TextEntity(text="Hallway", confidence=0.80, bbox=[800, 800, 850, 820], center=[825, 810])     # Outside (Orphan)
    ]

    # Mock Detections (YOLO)
    mock_dets = [
        {'label': 'toilet', 'bbox': [110, 110, 130, 130]}, # Inside Room 1
        {'label': 'sink', 'bbox': [410, 110, 430, 130]}    # Inside Room 2
    ]

    # 2. Initialize Assembler with a fake scale
    # Let's say 10 pixels = 1 foot.
    scale_eng = ScaleEngine(pixels_per_unit=10.0) 
    assembler = FusionAssembler(scale_engine=scale_eng)

    # 3. Run Logic
    result = assembler.assemble_floorplan(
        image_shape=img_size,
        room_mask=mock_mask,
        detections=mock_dets,
        ocr_results=mock_ocr
    )

    # 4. Verify Output
    print(json.dumps(result, indent=2))

    # Assertions for automated validation
    rooms = result['rooms']
    assert len(rooms) == 2, "Should detect 2 rooms"
    
    room1 = next(r for r in rooms if r['label'] == "Master Bed")
    # Area: 200x200 = 40,000 px. Scale 10px/ft -> 20x20 ft = 400 sqft.
    assert 390 < room1['area_sqft'] < 410, f"Area calc wrong: {room1['area_sqft']}"
    assert "toilet" in room1['objects_contained'], "Object containment failed"

    print("\nSUCCESS: Logic Layer assembled the floorplan correctly.")

if __name__ == "__main__":
    main()