from pymongo import MongoClient
from common.config import settings
from datetime import datetime
from typing import Dict, Optional, Any
import math

client = MongoClient(settings.MONGO_URI)
db = client[settings.DB_NAME]
collection = db["predictions"]

# ==========================================
# CREACIÓN DE ÍNDICES EN MONGO:
# ==========================================
# Al crear índices, las búsquedas por paciente o médico van  pasar de 
# complejidad O(N) a O(log N), optimizando la base de datos drásticamente.
collection.create_index("paciente_id")
collection.create_index("doctor_id")
collection.create_index([("created_at", -1)])


def save_prediction_metadata(
    doctor_id: str, 
    paciente_id: str, 
    task_type: str, 
    input_images: Dict[str, str], 
    prediction_url: Optional[str],
    status: str = "completed"
):
    """Registra el evento de predicción en MongoDB Atlas."""
    record = {
        "doctor_id": doctor_id,
        "paciente_id": paciente_id,
        "task_type": task_type,
        "created_at": datetime.utcnow(),
        "prediction_image": prediction_url,
        "status": status
    }
    
    # Inyectamos dinámicamente las URLs de entrada
    for modality, url in input_images.items():
        record[f"original_image_{modality}"] = url
        
    result = collection.insert_one(record)
    return str(result.inserted_id)