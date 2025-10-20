# ==============================
# Dockerfile para endpoint NSFW img2img / text2img (CORREGIDO)
# ==============================

# Base CUDA con Ubuntu (compatible con RunPod)
FROM nvidia/cuda:13.0.1-cudnn-runtime-ubuntu22.04

# Evita prompts interactivos
ENV DEBIAN_FRONTEND=noninteractive

# Instalar dependencias básicas (añadimos 'huggingface-cli' para descargar)
RUN apt-get update && apt-get install -y \
    git \
    python3-pip \
    python3-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Crear carpeta de la app
WORKDIR /app

# Copiar archivos
COPY main.py /app/

# Instalar librerías de Python necesarias
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# Añadimos 'huggingface-cli' y 'huggingface_hub' para descargar los modelos
RUN pip install --no-cache-dir diffusers transformers accelerate safetensors fastapi uvicorn pillow huggingface_hub

# ==============================
# Descargar modelos automáticamente (SOLUCIÓN AL ERROR 401)
# ==============================

# Transferir la variable de entorno del token de RunPod al ambiente
ARG HUGGINGFACE_HUB_TOKEN
ENV HUGGINGFACE_HUB_TOKEN=${HUGGINGFACE_HUB_TOKEN}

# 0. VERIFICACIÓN DEL TOKEN (NUEVO PASO)
RUN sh -c 'echo "🔑 Token HUGGINGFACE_HUB_TOKEN:" && \
    if [ -n "$HUGGINGFACE_HUB_TOKEN" ]; then \
        echo "Token encontrado, longitud: ${#HUGGINGFACE_HUB_TOKEN} caracteres. (Las primeras 5 letras son: ${HUGGINGFACE_HUB_TOKEN:0:5}...)"; \
    else \
        echo "ERROR: La variable de entorno HUGGINGFACE_HUB_TOKEN no está definida. La descarga fallará."; \
        exit 1; \
    fi'


# 1. Modelo base (FLUX-1-dev) - Usa snapshot_download para autenticación y descarga
RUN mkdir -p /app/models
RUN python3 - <<'EOF'
from huggingface_hub import snapshot_download
import os

token = os.getenv("HUGGINGFACE_HUB_TOKEN")
print(f"Descargando modelo base con token (trunco en Python): {token[:10]}...")

# Utilizamos snapshot_download para forzar la descarga del repositorio completo
snapshot_download(
    repo_id="black-forest-labs/FLUX.1-dev",
    local_dir="/app/models/FLUX.1-dev",
    allow_patterns=["*"],
    token=token
)
print("✅ Modelo base FLUX.1-dev descargado en /app/models/FLUX.1-dev.")
EOF

# 2. Descargar el LoRA NSFW (Flux Uncensored)
RUN mkdir -p /app/models/loras && \
    wget -O /app/models/loras/Flux-uncensored-v2.safetensors "https://huggingface.co/enhanceaiteam/Flux-uncensored-v2/resolve/main/lora.safetensors"

# ==============================
# Configuración de FastAPI
# ==============================

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
