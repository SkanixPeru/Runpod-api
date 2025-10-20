import os
import torch
from diffusers import StableDiffusionImg2ImgPipeline
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import FileResponse
from PIL import Image
from io import BytesIO
import requests

app = FastAPI()

# -----------------------------
# Configuración de modelos
# -----------------------------
MODEL_NAME = "black-forest-labs/FLUX.1-dev"
LORA_NAME = "enhanceaiteam/Flux-uncensored-v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LORA_PATH = "lora.safetensors"

# -----------------------------
# Descargar LoRA automáticamente si no existe
# -----------------------------
if not os.path.exists(LORA_PATH):
    print("📥 Descargando LoRA uncensored...")
    url = "https://huggingface.co/enhanceaiteam/Flux-uncensored-v2/resolve/main/lora.safetensors"
    r = requests.get(url)
    with open(LORA_PATH, "wb") as f:
        f.write(r.content)
    print("✅ LoRA descargado.")

# -----------------------------
# Cargar pipeline
# -----------------------------
print("⚡ Cargando modelo base y LoRA...")
pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16
).to(DEVICE)

pipeline.load_lora_weights(LORA_PATH)
pipeline.safety_checker = None
pipeline.enable_attention_slicing()

print("✅ Pipeline listo")

# -----------------------------
# Endpoint img2img
# -----------------------------
@app.post("/generate")
async def generate_image(
    file: UploadFile,
    prompt: str = Form(...),
    negative_prompt: str = Form("bad anatomy, low quality, blurry")
):
    # Leer imagen enviada
    img = Image.open(BytesIO(await file.read())).convert("RGB")

    # Generar imagen
    result = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=img,
        strength=0.5,
        guidance_scale=7.5,
        num_inference_steps=30
    ).images[0]

    # Guardar temporalmente
    output_path = "output.png"
    result.save(output_path)

    return FileResponse(output_path, media_type="image/png")
