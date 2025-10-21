# ==============================
# Dockerfile para endpoint FLUX img2img (FASTAPI)
# ==============================

# Usamos una imagen CUDA base (compatible con tu setup)
FROM nvidia/cuda:13.0.1-cudnn-runtime-ubuntu22.04

# Variables de Entorno
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Instalar dependencias básicas
RUN apt-get update && apt-get install -y git python3-pip python3-dev wget curl && rm -rf /var/lib/apt/lists/*
# Crear carpeta de la app e instalar Python
WORKDIR /app

# Instalar librerías de Python (PyTorch, Diffusers, FastAPI)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir diffusers transformers accelerate safetensors fastapi uvicorn pillow huggingface_hub requests

# Copiar la aplicación principal
COPY main.py /app/

# ==============================
# Configuración de FastAPI y RunPod
# ==============================

# El token de Hugging Face se inyecta desde RunPod
ARG HUGGINGFACE_HUB_TOKEN
ENV HUGGINGFACE_HUB_TOKEN=${HUGGINGFACE_HUB_TOKEN}

# Puerto estándar para el tráfico de la API
EXPOSE 8080

# El comando de inicio será manejado por el Start Command en la UI de RunPod.
# Sin embargo, establecemos el CMD por si la UI lo omite.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
