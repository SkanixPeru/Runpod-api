FROM nvidia/cuda:12.1.105-cudnn8-runtime-ubuntu22.04

# Instalar Python
RUN apt-get update && apt-get install -y python3 python3-pip git

# Copiar archivos
WORKDIR /app
COPY . /app

# Instalar dependencias
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

# Exponer puerto FastAPI
EXPOSE 8000

# Ejecutar servidor
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
