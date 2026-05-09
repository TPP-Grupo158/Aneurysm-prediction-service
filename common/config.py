import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    PROJECT_NAME: str = "Medical AI Inference Service"

    # Mongo
    MONGO_URI: str = os.getenv("MONGO_URI")
    DB_NAME: str = "medical_db"

    MINIO_ENDPOINT: str = os.getenv("MINIO_ENDPOINT")

    MINIO_ACCESS_KEY: str = os.getenv("MINIO_ACCESS_KEY")
    MINIO_SECRET_KEY: str = os.getenv("MINIO_SECRET_KEY")
    MINIO_BUCKET: str = "medical-images"

    MINIO_PUBLIC_URL: str = "http://localhost:9000"

settings = Settings()
