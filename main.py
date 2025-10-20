import os
import torch
from diffusers import StableDiffusionImg2ImgPipeline, AutoPipelineForText2Image
from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
from PIL import Image
from io import BytesIO
from huggingface_hub import hf_hub_download, login
from typing import Optional

app = FastAPI()

# -----------------------------
# Configuración de modelos y rutas
# -----------------------------
MODEL_NAME = "black-forest-labs/FLUX.1-dev"
LORA_NAME = "enhanceaiteam/Flux-uncensored-v2"
LORA_FILENAME = "lora.safetensors"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AUTH_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN")
pipeline = None # Pipeline inicializado como None

# RUTA CRÍTICA: RunPod monta el Network Volume en /workspace
WORKSPACE_DIR = "/workspace"
HF_CACHE_DIR = os.path.join(WORKSPACE_DIR, "huggingface_cache")

# -----------------------------
# Inicialización y Carga de Pipeline (RUNTIME)
# Se ejecuta al iniciar el servidor Uvicorn.
@app.on_event("startup")
def load_models_on_startup():
    global pipeline
    try:
        # 1. Crear el directorio de caché si no existe
        if not os.path.exists(HF_CACHE_DIR):
            os.makedirs(HF_CACHE_DIR, exist_ok=True)
            print(f"✅ Creado directorio de caché persistente: {HF_CACHE_DIR}")

        if AUTH_TOKEN:
            # Login para autenticar el acceso al modelo gated (IMPORTANTE)
            login(token=AUTH_TOKEN, add_to_git_credential=False)
            print("✅ Logueado en Hugging Face.")
        else:
            print("⚠️ HUGGINGFACE_HUB_TOKEN no definido. Fallará si el modelo es gated.")

        # 2. Descargar y Cargar Modelo Base
        # Usamos 'cache_dir' para guardar el modelo permanentemente en el Network Volume
        print(f"📥 Descargando/cargando modelo base {MODEL_NAME}...")
        pipe_t2i = AutoPipelineForText2Image.from_pretrained(
            MODEL_NAME, 
            cache_dir=HF_CACHE_DIR, # <--- Corregido: Usa la ruta persistente
            torch_dtype=torch.bfloat16,
            token=AUTH_TOKEN
        ).to(DEVICE)
        
        # Convertir a pipeline Img2Img
        pipeline = StableDiffusionImg2ImgPipeline(**pipe_t2i.components)
        
        # 3. Descargar y Cargar LoRA
        # Usamos 'cache_dir' para guardar el LoRA permanentemente en el Network Volume
        print(f"📥 Descargando/cargando LoRA {LORA_NAME}...")
        lora_local_path = hf_hub_download(
            repo_id=LORA_NAME, 
            filename=LORA_FILENAME, 
            cache_dir=HF_CACHE_DIR, # <--- Corregido: Usa la ruta persistente
            token=AUTH_TOKEN
        )

        pipeline.load_lora_weights(lora_local_path)
        pipeline.safety_checker = None
        pipeline.enable_attention_slicing()

        print("✅ Pipeline listo para generar imágenes.")

    except Exception as e:
        print(f"❌ ERROR CRÍTICO DURANTE LA CARGA DEL MODELO: {e}")
        pipeline = None # Deja el pipeline en None si hay error

# -----------------------------
# Endpoint img2img
# -----------------------------
@app.post("/generate")
async def generate_image(
    file: UploadFile,
    prompt: str = Form(...),
    # Prompt negativo extenso para prevenir mutaciones faciales/corporales
    negative_prompt: str = Form("mutations, disfigured, distorted face, strange anatomy, ugly, blurry hands, extra limbs, changing composition, 3d art, illustration, painting, sketch, low quality, bad anatomy, bad hands, missing fingers, extra fingers, fewer fingers, cropped, worst quality, low resolution, jpeg artifacts, signature, watermark, username, error, text, logo"),
    strength: float = Form(0.35), # <--- CLAVE: Control de iluminación
    guidance_scale: float = Form(7.5),
    num_inference_steps: int = Form(30)
):
    global pipeline
    if pipeline is None:
        raise HTTPException(status_code=500, detail="El pipeline del modelo no está cargado. Error de autenticación o descarga en el servidor.")
        
    # Leer imagen enviada
    img = Image.open(BytesIO(await file.read())).convert("RGB")

    # Generar imagen
    try:
        result = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=img,
            strength=strength, 
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps
        ).images[0]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error durante la generación: {e}")

    # Guardar temporalmente en /tmp
    output_path = "/tmp/output.png"
    result.save(output_path)

    return FileResponse(output_path, media_type="image/png")
