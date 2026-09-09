
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

api = FastAPI(title="DreamAI API")

api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@api.get("/")
def home():
    return {
        "app": "DreamAI",
        "status": "online"
    }

import os
from huggingface_hub import InferenceClient

HF_TOKEN = os.environ.get("HF_TOKEN")

if not HF_TOKEN:
    print("⚠️ HF_TOKEN is not set")

client = InferenceClient(
    api_key=HF_TOKEN
)

print("🤖 Hugging Face connection configured")

from pydantic import BaseModel

class GenerateRequest(BaseModel):
    prompt: str
    style: str = "Realistic"


@api.post("/generate")
def generate_image(request: GenerateRequest):

    if not request.prompt.strip():
        return {"error": "Please enter an image description."}

    full_prompt = f"""
    Create a {request.style} image of {request.prompt}.
    Highly detailed, beautiful composition,
    professional lighting, high quality.
    """

    image = client.text_to_image(
        prompt=full_prompt,
        model="black-forest-labs/FLUX.1-schnell"
    )

    file_path = "/tmp/dreamai_generated.png"
    image.save(file_path)

    return {
        "message": "Image generated successfully",
        "image_path": file_path
    }
