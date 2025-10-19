# 1. Usar una imagen base de Python
FROM python:3.9-slim

# 2. Establecer el directorio de trabajo
WORKDIR /app

# 3. Instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Crear el directorio para el modelo
RUN mkdir /app/model_sd3_5_large

# 5. Autenticarse y descargar el modelo (¡La parte importante!)
#    Runpod pasará tu secreto HF_TOKEN como un argumento de construcción.
ARG HF_TOKEN
ENV HF_HOME /tmp/.hf_cache # Evitar problemas de permisos de root para cache de HF
RUN python -c "import os; from huggingface_hub import login; token = os.environ.get('HF_TOKEN'); \
    if token: login(token=token, write_permission=True); print('Hugging Face login successful.') \
    else: print('HF_TOKEN not set, skipping login. Download might fail if repo is gated.')"

RUN python -c "from huggingface_hub import snapshot_download; \
    print('Starting model download...'); \
    snapshot_download(repo_id='stabilityai/stable-diffusion-3.5-large', \
                      local_dir='/app/model_sd3_5_large', \
                      local_dir_use_symlinks=False, \
                      resume_download=True); \
    print('Model download complete.')"

# 6. Copiar el script de tu API (app.py)
COPY app.py .

# 7. Exponer el puerto del servidor
EXPOSE 8080

# 8. Comando para iniciar el servidor
#    Usamos el puerto 8000, que es común para Runpod Serverless
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
