import sys
import glob
from pathlib import Path
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from blueprint_brain.config.settings import settings
from blueprint_brain.src.ingestion.cubicasa_loader import CubiCasaParser
from blueprint_brain.src.processing.tiler import ImageTiler

def main():
    # Setup Paths
    raw_path = settings.RAW_DATA_PATH
    output_path = settings.PROCESSED_PATH
    
    (output_path / "images").mkdir(parents=True, exist_ok=True)
    (output_path / "labels").mkdir(parents=True, exist_ok=True)
    
    print(f"Starting Data Preparation from {raw_path}...")
    
    # Initialize Components
    parser = CubiCasaParser(class_map=settings.CLASS_MAP)
    tiler = ImageTiler(tile_size=settings.TILE_SIZE, overlap=settings.TILE_OVERLAP)
    
    # Mock finding files (In real life, walk the directory)
    # Assumes structure: raw/id/model.svg and raw/id/F1_original.png
    # Here we simulate finding one for demonstration if files don't exist
    sample_files = list(raw_path.glob("*/*.svg"))
    
    if not sample_files:
        print("No SVG files found. Ensure CubiCasa5k is extracted in data/raw/")
        print("Debug: Generating a dummy synthetic image/svg for testing...")
        # create_dummy_data() # (Function to create fake data for testing logic)
        return

    for svg_file in tqdm(sample_files):
        try:
            folder = svg_file.parent
            # CubiCasa usually has 'F1_original.png' or similar
            img_file = list(folder.glob("*.png"))[0] 
            
            # 1. Parse SVG -> Polygons
            polygons = parser.parse_svg(str(svg_file))
            
            # 2. Tile Image & Generate Labels
            tiler.process_image(
                image_path=img_file,
                polygons_by_class=polygons,
                class_map=settings.CLASS_MAP,
                output_dir=output_path,
                base_filename=folder.name
            )
            
        except Exception as e:
            print(f"Error processing {svg_file}: {e}")

    print("Data Preparation Complete. Ready for training.")

if __name__ == "__main__":
    main()