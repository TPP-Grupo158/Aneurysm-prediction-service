from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import os
from dotenv import load_dotenv
import torch

BASE_DIR = Path(__file__).resolve().parent.parent

# Constants
MODEL_PATHS = {
        'vessel_ROI_seg': os.getenv("VESSEL_ROI_SEG"),
        'aneurysm_cls_1': os.getenv("ANEURYSM_CLS_1"),
        'aneurysm_cls_2': os.getenv("ANEURYSM_CLS_2"),
        'plane_2d_cls':  os.getenv("PLANE_2D_CLS"),
}
SHARED_DIR = Path('.')
TEMP_DIR = Path('tmp')
ID_COL = 'SeriesInstanceUID'
LABEL_COLS = [
    'Left Infraclinoid Internal Carotid Artery',
    'Right Infraclinoid Internal Carotid Artery',
    'Left Supraclinoid Internal Carotid Artery',
    'Right Supraclinoid Internal Carotid Artery',
    'Left Middle Cerebral Artery',
    'Right Middle Cerebral Artery',
    'Anterior Communicating Artery',
    'Left Anterior Cerebral Artery',
    'Right Anterior Cerebral Artery',
    'Left Posterior Communicating Artery',
    'Right Posterior Communicating Artery',
    'Basilar Tip',
    'Other Posterior Circulation',
    'Aneurysm Present',
]

# Define global variables at the module level
GLOBAL_VESSEL_ROI_PREDICTOR = None
GLOBAL_ANEURYSM_PREDICTOR_ALL_FOLDS = None
CLS_2D_PREDICTOR = None

USE_NUM_GPUS = 1
NUM_INFER_WORKERS = 1
COMPILE_NETWORK = False

executor = ThreadPoolExecutor(max_workers=NUM_INFER_WORKERS)

def get_device(gpu_id: int = 0) -> torch.device:
    """
    Get the computation device, with validation for GPU availability.
    """
    if torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
        return torch.device(f"cuda:{gpu_id}")
    return torch.device("cpu")

# Get the device (this part of the original code is fine)
DEVICE = get_device(gpu_id=0)