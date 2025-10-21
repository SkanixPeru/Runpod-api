#!/bin/bash

# --- CONFIGURACIÓN DE RUTAS ---
# Directorio donde RunPod monta tu Network Volume
MODEL_DIR="/workspace/huggingface_cache" 
LORA_REPO="enhanceaiteam/Flux-uncensored-v2"
LORA_FILENAME="lora.safetensors"
# Ruta completa donde el LORA debería existir en el disco
LORA_PATH="$MODEL_DIR/$LORA_FILENAME" 
LORA_DOWNLOAD_URL="https://huggingface.co/$LORA_REPO/resolve/main/$LORA_FILENAME"

# 1. Crear directorios persistentes
mkdir -p "$MODEL_DIR"

echo "================================================="
echo "VERIFICANDO PERSISTENCIA DE MODELOS EN WORKSPACE"
echo "================================================="

# 2. Verificar y Descargar LORA (Si no existe)
if [ -f "$LORA_PATH" ]; then
    echo "✅ LoRA encontrado. Saltando descarga."
else
    echo "❌ LoRA no encontrado. Descargando desde Hugging Face..."
    # Usamos wget para descargar el archivo directamente al volumen de red
    wget -O "$LORA_PATH" "$LORA_DOWNLOAD_URL"
    
    if [ $? -eq 0 ]; then
        echo "✅ Descarga del LoRA completada."
    else
        echo "❌ ERROR de descarga del LoRA. Abortando inicio."
        # Salir con código de error para que RunPod sepa que falló
        exit 1
    fi
fi

echo "================================================="
echo "PREPARACIÓN TERMINADA. Iniciando servidor FastAPI."
echo "================================================="

# Este script termina aquí. El comando de inicio en RunPod continuará con Uvicorn.
