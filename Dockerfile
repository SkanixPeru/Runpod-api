# ==============================
# Dockerfile para NSFW img2img endpoint
# ==============================

# Base CUDA con Ubuntu
FROM nvidia/cuda:13.0.1-cudnn-runtime-ubuntu22.04

# Evita preguntas de instalación
ENV DEBIAN_FRONTEND=noninteractive

# Actualizar e instalar dependencias básicas
RUN apt-get update && apt-get install -y \
    git \
    python3-pip \
    python3-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copiar requerimientos y main.py
WORKDIR /app
COPY main.py /app/

# Instalar librerías de Python
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir diffusers transformers accelerate safetensors fastapi uvicorn pillow

# ==============================
# Descargar modelos automáticamente
# ==============================
# Modelo base
RUN python3 -c "from diffusers import AutoPipelineForText2Image; AutoPipelineForText2Image.from_pretrained('black-forest-labs/FLUX.1-dev', torch_dtype='auto')"

# LoRA NSFW (Flux Uncensored)
RUN mkdir -p /app/models && \
    wget -O /app/models/Flux-uncensored-v2.safetensors https://huggingface.co/enhanceaiteam/Flux-uncensored-v2/resolve/main/lora.safetensors

# Puerto donde correrá FastAPI
EXPOSE 8000

# Comando para iniciar el servidor
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
