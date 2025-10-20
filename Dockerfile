# ==============================
# Dockerfile para endpoint NSFW img2img / text2img
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
    && rm -rf /var/lib/apt/lists/*

# Crear carpeta de la app
WORKDIR /app

# Copiar archivos
COPY main.py /app/

# Instalar librerías de Python necesarias
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir diffusers transformers accelerate safetensors fastapi uvicorn pillow huggingface_hub

# ==============================
# Descargar modelos automáticamente
# ==============================

# Variable de entorno del token (la recibirá RunPod Serverless)
ARG HUGGINGFACE_HUB_TOKEN
ENV HUGGINGFACE_HUB_TOKEN=${HUGGINGFACE_HUB_TOKEN}

# Logueo automático con Hugging Face
RUN huggingface-cli login --token ${HUGGINGFACE_HUB_TOKEN} || true

# Modelo base (FLUX-1-dev)
RUN python3 - <<'EOF'
from diffusers import AutoPipelineForText2Image
import torch, os

token = os.getenv("HUGGINGFACE_HUB_TOKEN")
print(f"Descargando modelo base con token: {token[:10]}...")

pipe = AutoPipelineForText2Image.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16,
    use_auth_token=token
)
EOF

# Descargar el LoRA NSFW (Flux Uncensored)
RUN mkdir -p /app/models && \
    wget -O /app/models/Flux-uncensored-v2.safetensors "https://huggingface.co/enhanceaiteam/Flux-uncensored-v2/resolve/main/lora.safetensors"

# ==============================
# Configuración de FastAPI
# ==============================

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
