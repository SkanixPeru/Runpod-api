from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from diffusers import StableDiffusionImg2ImgPipeline, AutoPipelineForText2Image
import torch
from PIL import Image
import io
import base64

app = FastAPI()

# ------------------------------
# Configura tu modelo NSFW
# ------------------------------
MODEL_NAME = "black-forest-labs/FLUX.1-dev"
LORA_PATH = "lora.safetensors"  # Ruta local al LoRA uncensored
DEVICE = "cuda"

# Carga pipeline img2img
pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16
).to(DEVICE)

pipeline.load_lora_weights(LORA_PATH)
pipeline.safety_checker = None  # Desactiva filtro de seguridad

# ------------------------------
# Request con prompt
# ------------------------------
class PromptRequest(BaseModel):
    prompt: str
    denoise: float = 0.5  # nivel de fuerza de la edición (0.0-1.0)

# ------------------------------
# Endpoint principal img2img
# ------------------------------
@app.post("/img2img")
async def img2img(prompt: str, image: UploadFile = File(...), denoise: float = 0.5):
    # Leer imagen
    contents = await image.read()
    init_image = Image.open(io.BytesIO(contents)).convert("RGB")
    
    # Ejecutar pipeline
    output = pipeline(prompt=prompt, image=init_image, strength=denoise, guidance_scale=7.5)
    img = output.images[0]
    
    # Convertir a Base64
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
    
    return {"image_base64": img_b64}
