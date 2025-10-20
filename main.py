import os
import torch
from diffusers import StableDiffusionImg2ImgPipeline, AutoPipelineForText2Image
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import FileResponse
from PIL import Image
from io import BytesIO
import requests
from typing import Optional

# --- CONFIGURACIÓN DE RUTAS LOCALES Y PIPELINE ---
# Usamos rutas locales para los modelos que descargamos en el Dockerfile
MODEL_PATH = "models/FLUX.1-dev"
LORA_PATH = "models/loras/Flux-uncensored-v2.safetensors"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI()

# -----------------------------
# Cargar pipeline
# -----------------------------
print("⚡ Cargando modelo base (local) y LoRA...")
try:
    # 1. Cargar el pipeline base T2I de forma local
    pipe_t2i = AutoPipelineForText2Image.from_pretrained(
        MODEL_PATH, 
        torch_dtype=torch.bfloat16,
        local_files_only=True # Forzar a cargar desde la carpeta local descargada
    ).to(DEVICE)
    
    # 2. Convertir a pipeline Img2Img
    pipeline = StableDiffusionImg2ImgPipeline(**pipe_t2i.components)
    
    # 3. Cargar el LoRA localmente
    pipeline.load_lora_weights(LORA_PATH, adapter_name="flux_uncensored")
    pipeline.safety_checker = None
    pipeline.enable_attention_slicing()

    print("✅ Pipeline listo")

except Exception as e:
    print(f"❌ ERROR CRÍTICO al cargar el pipeline: {e}")
    # Si la carga falla, asignamos None para evitar que el endpoint se ejecute
    pipeline = None

# -----------------------------
# Endpoint img2img
# -----------------------------
@app.post("/generate")
async def generate_image(
    file: UploadFile,
    prompt: str = Form(...),
    negative_prompt: str = Form("mutations, disfigured, distorted face, ugly, low quality, blurry"),
    strength: float = Form(0.35), # <--- ¡AQUÍ ESTÁ LA CLAVE! Control del Denoising
    guidance_scale: float = Form(7.5),
    num_inference_steps: int = Form(30)
):
    if pipeline is None:
        return {"error": "Pipeline no cargado. Revise los logs del Dockerfile."}, 500

    # Leer imagen enviada
    img = Image.open(BytesIO(await file.read())).convert("RGB")

    # Generar imagen
    result = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=img,
        strength=strength, # Usar el valor enviado por el usuario
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps
    ).images[0]

    # Guardar temporalmente
    output_path = "output.png"
    result.save(output_path)

    return FileResponse(output_path, media_type="image/png")
