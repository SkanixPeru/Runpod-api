# ==============================
# Dockerfile para endpoint FLUX img2img (SEGURO Y PERSISTENTE)
# ==============================

# Base CUDA con Ubuntu
FROM nvidia/cuda:13.0.1-cudnn-runtime-ubuntu22.04

# Variables de Entorno
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Instalar dependencias básicas (añadimos wget y curl)
RUN apt-get update && apt-get install -y \
    git \
    python3-pip \
    python3-dev \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Crear carpeta de la app y establecer /app como WORKDIR
WORKDIR /app

# Instalar librerías de Python
# Mantenemos las librerías en una sola capa (por eficiencia)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir diffusers transformers accelerate safetensors fastapi uvicorn pillow huggingface_hub requests

# ==============================
# LÓGICA DE PERSISTENCIA DE MODELOS
# ==============================

# Copiar el script de descarga DEBE hacerse ANTES del CMD
COPY download_models.sh /

# Dar permisos de ejecución al script
RUN chmod +x /download_models.sh

# Copiar la aplicación principal
COPY main.py /app/

# ==============================
# Configuración de FastAPI y RunPod
# ==============================

# El puerto estándar de RunPod para los workers
EXPOSE 8080

# El comando de inicio será manejado por el Start Command en la UI
# Asegúrate de usar el Start Command: /download_models.sh && uvicorn main:app --host 0.0.0.0 --port 8080
# El CMD es el fallback si no se usa Start Command
CMD ["/download_models.sh", "&&", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
