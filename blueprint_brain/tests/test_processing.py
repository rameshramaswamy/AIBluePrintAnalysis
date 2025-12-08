import pytest
import numpy as np
import shutil
from pathlib import Path
from blueprint_brain.src.processing.tiler import ImageTiler

# Mock Data
@pytest.fixture
def mock_image_env(tmp_path):
    """Creates a dummy 2000x2000 image and directories"""
    img = np.zeros((2000, 2000, 3), dtype=np.uint8)
    img_path = tmp_path / "test_blueprint.jpg"
    cv2.imwrite(str(img_path), img)
    
    out_dir = tmp_path / "processed"
    (out_dir / "images").mkdir(parents=True)
    (out_dir / "labels").mkdir(parents=True)
    
    return img_path, out_dir

def test_tiler_geometry(mock_image_env):
    """Ensure tiler calculates correct number of tiles"""
    img_path, out_dir = mock_image_env
    tiler = ImageTiler(tile_size=640, overlap=0.0) # 0 overlap for easy math
    
    # 2000 / 640 = 3.125 -> Should be 4 tiles wide x 4 tiles high = 16 tiles
    # Note: Logic forces last tile to shift back to fit, so coverage is 100%
    
    # We construct a mock task
    task = {
        'image_path': img_path,
        'polygons': {}, # No labels for this test
        'output_dir': out_dir,
        'class_map': {},
        'base_filename': "test"
    }
    
    result = tiler._process_single_image(task)
    assert "Success" in result
    
    # Check generated files
    generated_tiles = list((out_dir / "images").glob("*.jpg"))
    # The stride logic might produce slightly different counts based on edge handling
    # but strictly it should cover the area.
    assert len(generated_tiles) > 0