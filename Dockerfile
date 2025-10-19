# 1. Usar una imagen base de Python
FROM python:3.9-slim

# 2. Establecer el directorio de trabajo
WORKDIR /app

# 3. Instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Crear el directorio para el modelo
#    Esto se hará automáticamente por app.py si no existe
#    RUN mkdir /app/model_sd3_5_large

# 5. ¡ELIMINADO! Ya no se autentica ni descarga el modelo durante la construcción.

# 6. Copiar el script de tu API (app.py)
COPY app.py .

# 7. Exponer el puerto del servidor
EXPOSE 8000

# 8. Comando para iniciar el servidor
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
