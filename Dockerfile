# Use a base that supports Python 3.11 more easily or install 3.11 specifically
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Argentina/Buenos_Aires

WORKDIR /app

# Install Python 3.11 and system dependencies
RUN apt-get update && \
    ln -fs /usr/share/zoneinfo/$TZ /etc/localtime && \
    apt-get install -y \
    software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    curl && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and install python packages
COPY Intracranial_Aneurysm_Detection/wheels_20251001 /app/wheels
COPY Intracranial_Aneurysm_Detection/requirements.txt /app/requirements.txt
COPY Intracranial_Aneurysm_Detection/nnXNet /app/Intracranial_Aneurysm_Detection/nnXNet
RUN python3.11 -m pip install --no-cache-dir --upgrade pip && \
    python3.11 -m pip install /app/wheels/*.whl --no-deps && \
    # Install torch first
    python3.11 -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124 && \
    python3.11 -m pip install connected-components-3d && \
    # Install PyG dependencies from the specialized PyG index
    python3.11 -m pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.6.0+cu124.html && \
    # Now install the rest of your requirements
    python3.11 -m pip install --no-cache-dir -r /app/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124 && \
    python3.11 -m pip install -e /app/Intracranial_Aneurysm_Detection/nnXNet
# Copy the rest of the application
COPY . .

# Expose the FastAPI port
EXPOSE 8000

# Run the application (equivalent to your 'make build' command)
CMD ["fastapi", "run", "main.py", "--host", "0.0.0.0", "--port", "8000"]