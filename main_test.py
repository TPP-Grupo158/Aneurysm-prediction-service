import pytest
from unittest.mock import patch, MagicMock  # <--- ESTO ES LO QUE FALTA
from fastapi.testclient import TestClient
from main import app
from schemas import PredictionResponse
import sys
import polars as pl

sys.modules["Intracranial_Aneurysm_Detection.inference_prediction.inference"] = MagicMock()
sys.modules["common.minio_controller"] = MagicMock()
sys.modules["common.mongo_db"] = MagicMock()

client = TestClient(app)

@patch("main.upload_file")
@patch("main.upload_predictions")
@patch("main.predict_from_nifti")
@patch("main.save_prediction_metadata")
def test_predict_nifti_success(mock_save, mock_predict, mock_upload_p, mock_upload_f):
    mock_predict.return_value = pl.DataFrame([{
    "x": [0.2],
    "y": [0.111],
    "z": [0.978]
    }])
    mock_upload_f.return_value = "http://minio/origin"
    mock_upload_p.return_value = "http://minio/pred"
    mock_save.return_value = "db_123"
    
    data = {"doctor_id": "1", "paciente_id": "1111111"}
    files = {"file": ("test.nii.gz", b"binary", "application/octet-stream")}
    
    response = client.post("/predict/nifti", data=data, files=files)
    
    
    assert response.status_code == 200
    assert response.json() == {
        "status": "success",
        "db_id": "db_123",
        "paciente_id": "1111111",
        "doctor_id": "1",
        "original_images": {"t1": "http://minio/origin"},
        "prediction_result": "http://minio/pred",
        "task": "aneurysm",
        "modalities_used": ["t1"],
    }   