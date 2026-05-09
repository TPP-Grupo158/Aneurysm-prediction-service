from minio import Minio
from minio.error import S3Error
from common.config import settings
import json
from io import BytesIO

# Cliente MinIO
client = Minio(
    settings.MINIO_ENDPOINT,
    access_key=settings.MINIO_ACCESS_KEY,
    secret_key=settings.MINIO_SECRET_KEY,
    secure=False  # False porque estamos en local sin HTTPS
)

def upload_predictions(data: dict, object_name: str) -> str:
    """
    Sube un json de predicciones a MinIO y retorna la URL pública para el frontend.
    """
    try:
        # Subir el archivo (ya sabemos que el bucket existe por el lifespan)
        json_data = json.dumps(data).encode('utf-8')
        json_buffer = BytesIO(json_data)
        client.put_object(
            settings.MINIO_BUCKET,
            object_name,
            data=json_buffer,
            length=len(json_data),
            content_type="application/json" # Importante para que el navegador lo lea bien
        )
        url = f"{settings.MINIO_PUBLIC_URL}/{settings.MINIO_BUCKET}/{object_name}"
        return url

    except Exception as e:
        print(f"Error subiendo a MinIO: {e}")
        raise e
    
def upload_file(file_path: str, object_name: str) -> str:
    """
    Sube un archivo a MinIO y retorna la URL pública para el frontend.
    """
    try:
        # Subir el archivo (ya sabemos que el bucket existe por el lifespan)
        client.fput_object(
            settings.MINIO_BUCKET,
            object_name,
            file_path,
            content_type="application/octet-stream"
        )

        url = f"{settings.MINIO_PUBLIC_URL}/{settings.MINIO_BUCKET}/{object_name}"
        return url

    except Exception as e:
        print(f"Error subiendo a MinIO: {e}")
        raise e