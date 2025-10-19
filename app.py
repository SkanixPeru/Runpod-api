import os
import io
import torch
from PIL import Image
from fastapi import FastAPI, Response, UploadFile, File, Form
from diffusers import DiffusionPipeline, StableDiffusionImg2ImgPipeline
import logging

# Configurar logging básico
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Ruta local donde el Dockerfile descargó el modelo
LOCAL_MODEL_DIR = "/app/model_sd3_5_large"

# --- Cargar los Pipelines al iniciar ---
# Esto sucede UNA VEZ cuando el contenedor arranca (el "arranque en frío")
logger.info(f"Checking for model in {LOCAL_MODEL_DIR}...")

text2img_pipe = None
img2img_pipe = None

if not os.path.exists(LOCAL_MODEL_DIR):
    logger.error(f"¡Error Crítico! El directorio del modelo no existe: {LOCAL_MODEL_DIR}")
else:
    logger.info("Directorio del modelo encontrado. Cargando pipelines en la GPU...")
    try:
        # Cargar el pipeline Text2Img (por si acaso)
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
        logger.error(f"Error fatal al cargar modelos SD3.5 en GPU: {e}")
        # Si esto falla, el endpoint /health no funcionará correctamente

# --- Endpoints de la API ---

@app.get("/health")
async def health():
    """Endpoint de chequeo de salud que Runpod usará."""
    # Devuelve 200 OK si los modelos se cargaron correctamente
    if img2img_pipe is not None and text2img_pipe is not None:
        return {"status": "ok", "message": "Pipelines loaded."}
    else:
        # Devuelve un error si los modelos no se cargaron
        return Response(content="Error: Pipelines not loaded.", status_code=503)


@app.post("/predict")
async def predict(
    prompt: str = Form(...),
    image: UploadFile = File(None),
    denoising_strength: float = Form(0.5), # Control para Img2Img
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
            
            # Redimensionar la imagen de entrada a 1024x1024 (óptimo para SD3.5)
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

    # Devolver la imagen generada
    if output_image:
        img_byte_arr = io.BytesIO()
        output_image.save(img_byte_arr, format="PNG")
        img_byte_arr = img_byte_arr.getvalue()
        return Response(content=img_byte_arr, media_type="image/png")
    else:
        return Response(content="Error: No se pudo generar la imagen.", status_code=500)
