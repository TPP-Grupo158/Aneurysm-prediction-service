# 1. Usa la imagen de runtime, pero asegúrate de no reinstalar lo que ya trae
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /app

# 2. Evitar generar archivos .pyc y buffers de log para ahorrar espacio y mejorar performance
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. Instalación de dependencias de sistema en un solo paso
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 4. Copiar solo lo necesario para instalar dependencias primero
COPY Intracranial_Aneurysm_Detection/wheels_20251001 /tmp/wheels
COPY Intracranial_Aneurysm_Detection/requirements.txt .
COPY Intracranial_Aneurysm_Detection/nnXNet /app/nnXNet

# 5. Instalación de Python optimizada
# Eliminamos la reinstalación de torch porque ya viene en la imagen base
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir /tmp/wheels/*.whl --no-deps && \
    pip install --no-cache-dir connected-components-3d fastapi[standard] uvicorn python-dotenv pymongo minio pytest httpx && \
    pip install --no-cache-dir torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.6.0+cu124.html && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install -e /app/nnXNet && \
    rm -rf /tmp/wheels

# 6. Copia selectiva (Crucial para bajar esos 1.2 GB)
COPY . .

EXPOSE 8045

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8045"]