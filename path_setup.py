from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent
CURRENT_DIR = str(BASE_DIR / "Intracranial_Aneurysm_Detection")
sys.path.append(CURRENT_DIR)
sys.path.append(CURRENT_DIR+"/inference_prediction")
sys.path.append(CURRENT_DIR+'/nnXNet')

sys.path.append(CURRENT_DIR+'/wheels_20251001/dicom2nifti_20250917')

sys.path.append(CURRENT_DIR+'/wheels_20251001/acvl_utils-0.2.5')

sys.path.append(CURRENT_DIR+'/wheels_20251001/batchgenerators-0.25.1')

sys.path.append(CURRENT_DIR+'/wheels_20251001/dynamic_network_architectures-0.3.1')

print(sys.path)