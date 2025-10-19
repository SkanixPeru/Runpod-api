import os
import io
import torch
from PIL import Image
from fastapi import FastAPI, Response, UploadFile, File, Form
from diffusers import DiffusionPipeline, StableDiffusionImg2ImgPipeline
from huggingface_hub import snapshot_download, login
import logging

# Configurar logging básico
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Ruta local donde se guardará el modelo en el contenedor
LOCAL_MODEL_DIR = "/app/model_sd3_5_large"
HF_MODEL_ID = "stabilityai/stable-diffusion-3.5-large"

# --- Lógica de descarga y autenticación al iniciar el contenedor ---
# Esto se ejecuta UNA VEZ cuando el contenedor arranca (el "arranque en frío")
# Se asegura de que el modelo esté disponible antes de cargar los pipelines.

if not os.path.exists(LOCAL_MODEL_DIR):
    os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
    
    logger.info("Directorio del modelo no encontrado, iniciando descarga...")

    # Autenticación con Hugging Face
    hf_token = os.environ.get('HF_TOKEN')
    if hf_token:
        try:
            login(token=hf_token, write_permission=True)
            logger.info("Hugging Face login successful.")
        except Exception as e:
            logger.error(f"Error logging into Hugging Face: {e}")
            # Continuar, la descarga podría fallar si el repo es gated.
    else:
        logger.warning("HF_TOKEN environment variable not set. Download might fail if model is gated.")

    # Descargar el modelo
    try:
        logger.info(f"Descargando modelo {HF_MODEL_ID} a {LOCAL_MODEL_DIR}...")
        snapshot_download(
            repo_id=HF_MODEL_ID,
            local_dir=LOCAL_MODEL_DIR,
            local_dir_use_symlinks=False,
            resume_download=True
        )
        logger.info("Descarga del modelo completa.")
    except Exception as e:
        logger.error(f"Error fatal al descargar el modelo de Hugging Face: {e}")
        # Si la descarga falla aquí, el contenedor no podrá iniciar correctamente.
        # Esto resultará en que el endpoint /health no responda.
else:
    logger.info("Directorio del modelo encontrado localmente. No se necesita descarga.")


# --- Cargar los Pipelines ---
# Carga los modelos desde el disco local del contenedor a la GPU.
text2img_pipe = None
img2img_pipe = None

try:
    logger.info("Cargando pipelines en la GPU...")
    # Cargar el pipeline Text2Img
    text2img_pipe = DiffusionPipeline.from_pretrained(
        LOCAL_MODEL_DIR,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16" # Usar la variante fp16 si está disponible
    ).to("cuda")

    # Cargar el pipeline Img2Img
    img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        LOCAL_MODEL_DIR,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16"
    ).to("cuda")

    logger.info("¡Éxito! Modelos Stable Diffusion 3.5 Large (Text2Img & Img2Img) cargados en GPU.")
except Exception as e:
    logger.error(f"Error fatal al cargar modelos SD3.5 en GPU: {e}.")
    # Si falla aquí, el pod probablemente se reiniciará.


# --- Endpoints de la API ---

@app.get("/health")
async def health():
    """Endpoint de chequeo de salud que Runpod usará."""
    if img2img_pipe is not None and text2img_pipe is not None:
        return {"status": "ok", "message": "Pipelines loaded."}
    else:
        return Response(content="Error: Pipelines not loaded.", status_code=503)


@app.post("/predict")
async def predict(
    prompt: str = Form(...),
    image: UploadFile = File(None),
    denoising_strength: float = Form(0.5),
    num_inference_steps: int = Form(25)
):
    """Genera o edita una imagen."""
    logger.info(f"Recibida solicitud /predict. Prompt: {prompt[:30]}...")
    output_image = None

    if image and image.filename != '':
        # --- MODO Img2Img (Edición) ---
        logger.info(f"Modo Img2Img. Strength: {denoising_strength}, Steps: {num_inference_steps}")
        try:
            input_image_bytes = await image.read()
            init_image = Image.open(io.BytesIO(input_image_bytes)).convert("RGB")
            init_image = init_image.resize((1024, 1024))
    
            if img2img_pipe is None:
                logger.error("Img2Img pipeline no está cargado.")
                return Response(content="Error: Img2Img pipeline no cargado.", status_code=503)
    
            with torch.no_grad():
                output_image = img2img_pipe(
                    prompt=prompt,
                    image=init_image,
                    strength=denoising_strength,
                    num_inference_steps=num_inference_steps
                ).images[0]
            logger.info("Generación Img2Img completada.")

        except Exception as e:
            logger.error(f"Error durante la predicción Img2Img: {e}")
            return Response(content=f"Error interno: {e}", status_code=500)
            
    else:
        # --- MODO Text2Img (Generación) ---
        logger.info(f"Modo Text2Img. Steps: {num_inference_steps}")
        if text2img_pipe is None:
            logger.error("Text2Img pipeline no está cargado.")
            return Response(content="Error: Text2Img pipeline no cargado.", status_code=503)
        
        with torch.no_grad():
            output_image = text2img_pipe(
                prompt=prompt,
                num_inference_steps=num_inference_steps
            ).images[0]
        logger.info("Generación Text2Img completada.")

    if output_image:
        img_byte_arr = io.BytesIO()
        output_image.save(img_byte_arr, format="PNG")
        img_byte_arr = img_byte_arr.getvalue()
        return Response(content=img_byte_arr, media_type="image/png")
    else:
        return Response(content="Error: No se pudo generar la imagen.", status_code=500)
