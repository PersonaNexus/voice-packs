"""FastAPI server for voice pack inference — hot-swap adapters per request."""

import os
import subprocess
import sys
import tempfile
from typing import Optional

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
except ImportError:
    raise ImportError("Install serve extras: pip install 'voice-packs[serve]'")


app = FastAPI(
    title="PersonaNexus Voice Pack Server",
    description="Generate text with swappable personality adapters",
    version="0.1.0",
)

DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-360M"


class GenerateRequest(BaseModel):
    prompt: str
    voice_pack: str  # adapter path or HF repo path
    max_tokens: int = 200
    temperature: float = 0.7
    model: str = DEFAULT_MODEL


class BlendRequest(BaseModel):
    prompt: str
    voice_a: str
    voice_b: str
    ratio: float = 0.5
    max_tokens: int = 200
    temperature: float = 0.7
    model: str = DEFAULT_MODEL


class GenerateResponse(BaseModel):
    text: str
    voice_pack: str
    model: str


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    """Generate text with a voice pack."""
    from voice_packs.generate import generate as gen

    try:
        text = gen(
            adapter_path=req.voice_pack,
            prompt=req.prompt,
            model=req.model,
            max_tokens=req.max_tokens,
            temp=req.temperature,
        )
        return GenerateResponse(text=text, voice_pack=req.voice_pack, model=req.model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/blend", response_model=GenerateResponse)
async def blend_generate(req: BlendRequest):
    """Blend two voice packs and generate."""
    from voice_packs.blend import blend
    from voice_packs.generate import generate as gen

    blend_dir = os.path.join(tempfile.gettempdir(), "voice-blend")
    try:
        blend(req.voice_a, req.voice_b, blend_dir, req.ratio)
        text = gen(
            adapter_path=blend_dir,
            prompt=req.prompt,
            model=req.model,
            max_tokens=req.max_tokens,
            temp=req.temperature,
        )
        return GenerateResponse(
            text=text,
            voice_pack=f"blend({req.voice_a},{req.voice_b},{req.ratio})",
            model=req.model,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok"}


def run_server(host: str = "0.0.0.0", port: int = 8080):
    """Start the voice pack server."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)
