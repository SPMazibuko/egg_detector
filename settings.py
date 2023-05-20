from pathlib import Path
import sys

# Get the absolute path of the current file
file_path = Path(__file__).resolve()

# Get the parent directory of the current file
root_path = file_path.parent

# Add the root path to the sys.path list if it is not already there
if root_path not in sys.path:
    sys.path.append(str(root_path))

# Get the relative path of the root directory with respect to the current working directory
ROOT = root_path.relative_to(Path.cwd())

# Source
IMAGE = 'Image'
WEBCAM = 'Webcam'

SOURCES_LIST = [IMAGE, WEBCAM]

# images
IMAGES_DIR = ROOT / 'images'
DEFAULT_IMAGE = IMAGES_DIR / '1.jpg'
DEFAULT_DETECT_IMAGE = IMAGES_DIR / '1_detected.jpeg'

# model
MODEL_DIR = ROOT / 'weights'
DETECTION_MODEL = MODEL_DIR / 'best.pt'

# Detected/segmented image dirpath locator
DETECT_LOCATOR = 'detect'

# Webcam
WEBCAM_PATH = 0
