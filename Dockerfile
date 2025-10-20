FROM nvidia/cuda:12.1.105-cudnn8-runtime-ubuntu22.04

# Evita preguntas de instalación
ENV DEBIAN_FRONTEND=noninteractive

# Instalar dependencias básicas
RUN apt-get update && apt-get install -y \
    git python3-pip python3-venv ffmpeg libsm6 libxext6 && \
    rm -rf /var/lib/apt/lists/*

# Copiar archivos
WORKDIR /app
COPY . /app

# Instalar requirements
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Exponer puerto de FastAPI
EXPOSE 8000

# Ejecutar FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
