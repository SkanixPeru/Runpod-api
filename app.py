import os
import io
import torch
import base64
import re
from PIL import Image
from fastapi import FastAPI, Response
from pydantic import BaseModel, Field # Importa Pydantic para manejar el JSON
from diffusers import DiffusionPipeline, StableDiffusionImg2ImgPipeline
from huggingface_hub import snapshot_download, login
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# --- Definir la estructura JSON de ENTRADA ---
# Esto coincide con el estándar de Runpod
class Item(BaseModel):
    image: str = Field(..., description="La imagen codificada en Base64 (ej: data:image/jpeg;base64,...)")
    prompt: str
    denoising_strength: float = 0.5
    num_inference_steps: int = 25

class RunpodInput(BaseModel):
    input: Item

# --- Lógica de descarga y autenticación ---
# ¡CAMBIO IMPORTANTE!
# Ahora apunta a tu "disco duro" (Network Storage).
LOCAL_MODEL_DIR = "/runpod-volume/model_sd_3_5_large" 
HF_MODEL_ID = "stabilityai/stable-diffusion-3.5-large"

# Esto solo se ejecutará LA PRIMERA VEZ que llames a la API.
# Descargará el modelo al "disco duro".
if not os.path.exists(LOCAL_MODEL_DIR):
    os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
    logger.info("Directorio del modelo no encontrado en el disco duro, iniciando descarga (esto tardará varios minutos)...")

    hf_token = os.environ.get('HF_TOKEN')
    if hf_token:
        try:
            login(token=hf_token, write_permission=True)
            logger.info("Hugging Face login successful.")
        except Exception as e:
            logger.error(f"Error logging into Hugging Face: {e}")
    else:
        logger.warning("HF_TOKEN environment variable not set.")

    try:
        logger.info(f"Descargando modelo {HF_MODEL_ID} a {LOCAL_MODEL_DIR}...")
        snapshot_download(
            repo_id=HF_MODEL_ID,
            local_dir=LOCAL_MODEL_DIR,
            local_dir_use_symlinks=False,
            resume_download=True
        )
        logger.info("Descarga del modelo al disco duro completa.")
    except Exception as e:
        logger.error(f"Error fatal al descargar el modelo de Hugging Face: {e}")
else:
    # A partir de la SEGUNDA llamada, entrará aquí (instantáneo)
    logger.info("Directorio del modelo encontrado en el disco duro. Cargando...")

# --- Cargar los Pipelines ---
text2img_pipe = None
img2img_pipe = None
try:
    logger.info("Cargando pipelines en la GPU...")
    text2img_pipe = DiffusionPipeline.from_pretrained(
        LOCAL_MODEL_DIR, torch_dtype=torch.float16, use_safetensors=True, variant="fp16"
    ).to("cuda")
    img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        LOCAL_MODEL_DIR, torch_dtype=torch.float16, use_safetensors=True, variant="fp16"
    ).to("cuda")
    logger.info("¡Éxito! Modelos cargados en GPU.")
except Exception as e:
    logger.error(f"Error fatal al cargar modelos SD3.5 en GPU: {e}")

# --- Endpoints de la API ---

@app.get("/health")
async def health():
    if img2img_pipe is not None:
        return {"status": "ok", "message": "Pipelines loaded."}
    else:
        return Response(content='{"error":"Pipelines not loaded"}', status_code=503, media_type="application/json")

@app.post("/predict")
async def predict(runpod_input: RunpodInput): # <-- Acepta el JSON
    """Genera o edita una imagen desde un input JSON con Base64."""
    item = runpod_input.input # Extrae el objeto "input"
    
    logger.info(f"Modo Img2Img (JSON). Strength: {item.denoising_strength}")
    try:
        # Decodificar la imagen Base64
        img_data_str = item.image.split(',')[-1]
        img_data_bytes = base64.b64decode(img_data_str)
        init_image = Image.open(io.BytesIO(img_data_bytes)).convert("RGB")
        init_image = init_image.resize((1024, 1024))

        if img2img_pipe is None:
            logger.error("Img2Img pipeline no está cargado.")
            return Response(content='{"error":"Img2Img pipeline not loaded"}', status_code=503, media_type="application/json")

        with torch.no_grad():
            output_image = img2img_pipe(
                prompt=item.prompt,
                image=init_image,
                strength=item.denoising_strength,
                num_inference_steps=item.num_inference_steps
            ).images[0]
        logger.info("Generación Img2Img completada.")

        # Devolver la imagen
        img_byte_arr = io.BytesIO()
        output_image.save(img_byte_arr, format="PNG")
        img_byte_arr = img_byte_arr.getvalue()
        return Response(content=img_byte_arr, media_type="image/png")

    except Exception as e:
        logger.error(f"Error durante la predicción Img2Img: {e}")
        return Response(content=f'{{"error":"Internal server error: {e}"}}', status_code=500, media_type="application/json")
