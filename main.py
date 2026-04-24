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
@app.get("/predict/dicom")
async def predict_with_dicom():
    return {"prediction": "result"}

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