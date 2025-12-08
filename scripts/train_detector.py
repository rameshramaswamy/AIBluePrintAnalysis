import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from blueprint_brain.config.settings import settings
from blueprint_brain.src.models.detector import BlueprintDetector

def main():
    # 1. Config
    # We point to 'data/processed' where we saved tiles in Iteration 1
    data_dir = settings.PROCESSED_PATH
    
    # Check if data exists
    if not (data_dir / "images").exists():
        print(f"Error: Data not found at {data_dir}. Run 'scripts/prepare_data.py' first.")
        return

    # 2. Initialize Model
    # Using 'yolov8s.pt' (Small) - Good balance of speed/accuracy for PoC
    detector = BlueprintDetector(model_version="yolov8s.pt")
    
    # 3. Train
    print("Initializing Training Pipeline...")
    detector.train(
        data_path=data_dir,
        class_map=settings.CLASS_MAP,
        epochs=10,  # Set to 10 for quick validation, 100+ for production
        img_size=settings.TILE_SIZE,
        batch_size=8 # Lower batch size if running on CPU/Small GPU
    )

    print("Training finished. Check 'blueprint_brain_runs/v1_baseline' for metrics.")

if __name__ == "__main__":
    main()