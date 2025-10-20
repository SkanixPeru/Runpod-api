from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from diffusers import AutoPipelineForText2Image
import torch
from PIL import Image
import io

app = FastAPI()

# ---------- Configura el modelo ----------
MODEL_NAME = "black-forest-labs/FLUX.1-dev"
LORA_PATH = "lora.safetensors"  # o ruta remota si no lo subes
DEVICE = "cuda"

pipeline = AutoPipelineForText2Image.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16
).to(DEVICE)

pipeline.load_lora_weights(LORA_PATH)

# ---------- Endpoint para generar imagen ----------
class PromptRequest(BaseModel):
    prompt: str

@app.post("/generate")
async def generate(req: PromptRequest):
    image = pipeline(req.prompt).images[0]
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    return {"image": buf.getvalue().hex()}  # enviamos como hex para JSON
