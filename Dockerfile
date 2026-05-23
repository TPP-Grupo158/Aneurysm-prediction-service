# Use a base that supports Python 3.11 more easily or install 3.11 specifically
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

# 2. Limpieza de sistema inmediata
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
# Copy requirements and install python packages
COPY Intracranial_Aneurysm_Detection/wheels_20251001 /app/wheels
COPY Intracranial_Aneurysm_Detection/requirements.txt /app/requirements.txt
COPY Intracranial_Aneurysm_Detection/nnXNet /app/Intracranial_Aneurysm_Detection/nnXNet
RUN python3.11 -m pip install --no-cache-dir --upgrade pip && \
    python3.11 -m pip install /app/wheels/*.whl --no-deps && \
    # Install torch first
    python3.11 -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124 && \
    python3.11 -m pip install connected-components-3d && \
    python3.11 -m pip install fastapi[standard] uvicorn python-dotenv pymongo minio && \
    # Install PyG dependencies from the specialized PyG index
    python3.11 -m pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.6.0+cu124.html && \
    # Now install the rest of your requirements
    python3.11 -m pip install --no-cache-dir -r /app/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124 && \
    python3.11 -m pip install -e /app/Intracranial_Aneurysm_Detection/nnXNet
# Copy the rest of the application
COPY . .

# Expose the FastAPI port
EXPOSE 8045

# Run the application (equivalent to your 'make build' command)
CMD ["python3.11", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8045"]