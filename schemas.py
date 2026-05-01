from pydantic import BaseModel, HttpUrl
from typing import List, Dict
from enum import Enum
from datetime import datetime


class PredictionResponse(BaseModel):
    status: str
    db_id: str
    paciente_id: str
    doctor_id: str
    original_images: Dict[str, str]
    prediction_result: dict
    task: str
    modalities_used: List[str]