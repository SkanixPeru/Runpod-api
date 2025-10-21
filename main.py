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
LORA_REPO = "enhanceaiteam/Flux-uncensored-v2"
LORA_FILENAME = "lora.safetensors"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AUTH_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN")
pipeline = None # Pipeline inicializado como None

# RUTA CRÍTICA: RunPod monta el Network Volume en /workspace
WORKSPACE_DIR = "/workspace"
HF_CACHE_DIR = os.path.join(WORKSPACE_DIR, "huggingface_cache")
# Ruta del LORA, asumimos que fue descargado por download_models.sh
LORA_LOCAL_PATH = os.path.join(HF_CACHE_DIR, LORA_FILENAME)


# -----------------------------
# Inicialización y Carga de Pipeline (RUNTIME)
@app.on_event("startup")
def load_models_on_startup():
    global pipeline
    try:
        # 1. Autenticación y creación de directorios
        if not os.path.exists(HF_CACHE_DIR):
            os.makedirs(HF_CACHE_DIR, exist_ok=True)
            print(f"✅ Creado directorio de caché persistente: {HF_CACHE_DIR}")

        if AUTH_TOKEN:
            login(token=AUTH_TOKEN, add_to_git_credential=False)
            print("✅ Logueado en Hugging Face.")
        else:
            print("⚠️ HUGGINGFACE_HUB_TOKEN no definido.")

        # 2. Descargar/Cargar Modelo Base
        # Esto usa el cache_dir persistente. Solo descarga si no existe.
        print(f"📥 Descargando/cargando modelo base {MODEL_NAME}...")
        pipe_t2i = AutoPipelineForText2Image.from_pretrained(
            MODEL_NAME, 
            cache_dir=HF_CACHE_DIR, # <--- Usa la ruta persistente
            torch_dtype=torch.bfloat16,
            token=AUTH_TOKEN
        ).to(DEVICE)
        
        pipeline = StableDiffusionImg2ImgPipeline(**pipe_t2i.components)
        
        # 3. Cargar LoRA DESDE LA RUTA PERMANENTE
        # El download_models.sh asegura que este archivo ya existe en el disco.
        if os.path.exists(LORA_LOCAL_PATH):
            print(f"🔄 Cargando LoRA localmente desde {LORA_LOCAL_PATH}...")
            pipeline.load_lora_weights(LORA_LOCAL_PATH)
        else:
            print(f"⚠️ LoRA no encontrado en el disco persistente: {LORA_LOCAL_PATH}")
        
        pipeline.safety_checker = None
        pipeline.enable_attention_slicing()
        print("✅ Pipeline listo para generar imágenes.")

    except Exception as e:
        print(f"❌ ERROR CRÍTICO DURANTE LA CARGA DEL MODELO: {e}")
        pipeline = None 

# -----------------------------
# Endpoint img2img
# -----------------------------
@app.post("/generate")
async def generate_image(
    file: UploadFile,
    prompt: str = Form(...),
    negative_prompt: str = Form("mutations, disfigured, distorted face, strange anatomy, ugly, blurry hands, extra limbs, changing composition, 3d art, illustration, painting, sketch, low quality, bad anatomy, bad hands, missing fingers, extra fingers, fewer fingers, cropped, worst quality, low resolution, jpeg artifacts, signature, watermark, username, error, text, logo"),
    strength: float = Form(0.35), 
    guidance_scale: float = Form(7.5),
    num_inference_steps: int = Form(30)
):
    global pipeline
    if pipeline is None:
        # Este error ahora es muy probable que signifique un fallo de autenticación o descarga
        raise HTTPException(status_code=500, detail="El pipeline del modelo no está cargado. Verifique logs y token de Hugging Face.")
        
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

    # Devuelve la imagen como un archivo PNG
    return FileResponse(output_path, media_type="image/png")