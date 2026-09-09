
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from huggingface_hub import InferenceClient
import os
import io

api = FastAPI(title="DreamAI API")

api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

HF_TOKEN = os.environ.get("HF_TOKEN")

if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN is not configured")

client = InferenceClient(api_key=HF_TOKEN)


class GenerateRequest(BaseModel):
    prompt: str
    style: str = "Realistic"


@api.get("/")
def home():
    return {
        "app": "DreamAI",
        "status": "online"
    }


@api.post("/generate")
def generate_image(request: GenerateRequest):

    if not request.prompt.strip():
        raise HTTPException(
            status_code=400,
            detail="Please enter an image description."
        )

    full_prompt = f"""
Create a {request.style} image of {request.prompt}.
Highly detailed, beautiful composition,
professional lighting, high quality.
"""

    try:
        image = client.text_to_image(
            prompt=full_prompt,
            model="black-forest-labs/FLUX.1-schnell"
        )

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")

        return Response(
            content=buffer.getvalue(),
            media_type="image/png"
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail="Image generation failed."
        )
