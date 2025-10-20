# ==============================
# Dockerfile para endpoint FLUX img2img (SEGURO)
# ==============================

# Base CUDA con Ubuntu (compatible con RunPod)
FROM nvidia/cuda:13.0.1-cudnn-runtime-ubuntu22.04

# Evita prompts interactivos
ENV DEBIAN_FRONTEND=noninteractive

# Instalar dependencias básicas
RUN apt-get update && apt-get install -y \
    git \
    python3-pip \
    python3-dev \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Crear carpeta de la app
WORKDIR /app

# Copiar archivos
COPY main.py /app/

# Instalar librerías de Python necesarias (PyTorch)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Instalar librerías de Python (Diffusers, FastAPI, etc.)
RUN pip install --no-cache-dir diffusers transformers accelerate safetensors fastapi uvicorn pillow huggingface_hub requests

# ==============================
# Configuración de Modelos y Token
# =========================================================================

# El token de Hugging Face se inyecta desde RunPod y lo toma FastAPI
ARG HUGGINGFACE_HUB_TOKEN
ENV HUGGINGFACE_HUB_TOKEN=${HUGGINGFACE_HUB_TOKEN}

# ==============================
# Configuración de FastAPI
# ==============================

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
