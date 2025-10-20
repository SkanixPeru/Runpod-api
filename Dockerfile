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
# Descargar modelos automáticamente
# ==============================

# Modelo base (FLUX-1-dev)
RUN python3 - <<'EOF'
from diffusers import AutoPipelineForText2Image
import torch

token = "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"
print(f"Descargando modelo base con token: {token[:10]}...")

pipe = AutoPipelineForText2Image.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16,
    use_auth_token=token
)
EOF

# Descargar LoRA NSFW
RUN mkdir -p /app/models && \
    wget --header="Authorization: Bearer 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" \
    -O /app/models/Flux-uncensored-v2.safetensors \
    https://huggingface.co/enhanceaiteam/Flux-uncensored-v2/resolve/main/lora.safetensors

# Limpieza de token (seguridad)
RUN rm -rf ~/.cache/huggingface && \
    echo "Token eliminado del entorno ✅"

# ==============================
# Configuración de FastAPI
# ==============================

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
