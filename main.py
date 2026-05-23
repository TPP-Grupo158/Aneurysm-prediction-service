from pathlib import Path
import sys
from schemas import PredictionResponse
from common.minio_controller import upload_predictions, upload_file
from common.mongo_db import save_prediction_metadata
import json
from io import BytesIO

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
from fastapi import FastAPI, HTTPException, Form, status, Cookie, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from typing import Annotated
import shutil
import os
import zipfile
import uuid
BASE_TMP_PATH = "/tmp/dicoms"
TASK_ANEURYSM = "aneurysm"

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

@app.post("/predict/dicom")
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


        return {"prediction":result.to_dicts()[0]}

    except Exception as e:
        cleanup_folder(extraction_path)
        raise HTTPException(status_code=500, detail=f"Error procesando zip: {str(e)}")

@app.post("/predict/nifti")
async def predict_with_dicom_niftis(doctor_id: str = Form(...), paciente_id: str = Form(...),file: UploadFile = File(...)):
    temp_path = f"/tmp/{file.filename}"
    study_id = str(uuid.uuid4())
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        result = predict_from_nifti(temp_path).to_dicts()[0]
        minio_origin_path = f"{paciente_id}/{TASK_ANEURYSM}/{study_id}/file_t1.nii.gz"
        predicition_path = f"{paciente_id}/{TASK_ANEURYSM}/{study_id}/prediction.json"
        url_origin=upload_file(temp_path, minio_origin_path)
        url=upload_predictions(result, predicition_path)
        db_id = save_prediction_metadata(
            doctor_id=doctor_id,
            paciente_id=paciente_id,
            task_type=TASK_ANEURYSM,
            input_images={"t1": url_origin},
            prediction_url=url,
        )
        return PredictionResponse(
            status="success",
            db_id=db_id,
            paciente_id=paciente_id,
            doctor_id=doctor_id,
            original_images={"t1": url_origin},
            prediction_result=url,
            task= TASK_ANEURYSM,
            modalities_used=["t1"],
        )
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.get("/ping/")
async def ping():
    return{"result": "pong"}