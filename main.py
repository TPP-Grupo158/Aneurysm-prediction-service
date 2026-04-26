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


from Intracranial_Aneurysm_Detection.inference_prediction.inference import predict, predict_from_nifti
from fastapi import FastAPI, HTTPException, Response, status, Cookie, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from typing import Annotated
import shutil
import os
import zipfile
import uuid
BASE_TMP_PATH = "/tmp/dicoms"

origins = [
    "http://localhost:8001",  # Your Vite dev server
    "http://127.0.0.1:8001",
    "*"
]

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,           # Allow specific origins
    allow_credentials=True,
    allow_methods=["*"],             # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],             # Allow all headers (Content-Type, etc.)
)

def cleanup_folder(folder_path: str):
    """Función para borrar la carpeta después de procesarla"""
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        print(f"Carpeta borrada: {folder_path}")

@app.get("/predict/dicom")
async def predict_with_dicom(file: UploadFile = File(...)):
    if not file.filename.endswith(".zip"):
        raise HTTPException(status_code=400, detail="El archivo debe ser un .zip")
    unique_id = str(uuid.uuid4())
    extraction_path = os.path.join(BASE_TMP_PATH, unique_id)
    os.makedirs(extraction_path, exist_ok=True)
    zip_path = os.path.join(extraction_path, "temp_upload.zip")
    try:
        # 3. Guardar el archivo ZIP recibido en disco
        with open(zip_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 4. Descomprimir
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extraction_path)
        
        os.remove(zip_path)

        contenido = os.listdir(extraction_path)

        result=predict(extraction_path)


        return {"prediction":result}

    except Exception as e:
        cleanup_folder(extraction_path)
        raise HTTPException(status_code=500, detail=f"Error procesando zip: {str(e)}")

@app.get("/predict/nifti")
async def predict_with_dicom_niftis(file: UploadFile = File(...)):
    temp_path = f"/tmp/{file.filename}"
    
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    result=predict_from_nifti(temp_path).to_dicts()[0]
    shutil.rmtree('/tmp', ignore_errors=True)
    return {"prediction":result}

@app.get("/ping/")
async def ping():
    return{"result": "pong"}