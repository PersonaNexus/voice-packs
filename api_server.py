#!/usr/bin/env python3
"""
PersonaNexus Voice Pack API Server

Adapter-aware inference server for voice packs. Loads SmolLM2-360M once,
hot-swaps LoRA adapters per request.

OpenAI-compatible /v1/completions endpoint with a `voice` parameter.

Usage:
    uv run python api_server.py
    # Server starts at http://localhost:8421

    curl http://localhost:8421/v1/completions -X POST -H "Content-Type: application/json" -d '{
        "prompt": "It was a dark and stormy night",
        "voice": "dickens",
        "max_tokens": 200,
        "temperature": 0.8
    }'
"""

import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

SCRIPT_DIR = Path(__file__).parent
BASE_MODEL = "HuggingFaceTB/SmolLM2-360M"

# Adapter directories — scan for any pack with adapters.safetensors
VOICE_PACK_DIR = SCRIPT_DIR


def discover_voices():
    """Find all voice packs with trained adapters."""
    voices = {}
    for entry in sorted(VOICE_PACK_DIR.iterdir()):
        adapter_file = entry / "adapters" / "adapters.safetensors"
        if entry.is_dir() and adapter_file.exists():
            voices[entry.name] = str(entry / "adapters")
    return voices


# Global state
_state = {
    "model": None,
    "tokenizer": None,
    "voices": {},
    "current_voice": None,
}


def load_base_model():
    """Load the base model and tokenizer once."""
    from mlx_lm import load
    print(f"Loading base model: {BASE_MODEL}")
    model, tokenizer = load(BASE_MODEL)
    _state["model"] = model
    _state["tokenizer"] = tokenizer
    _state["voices"] = discover_voices()
    _state["current_voice"] = None
    print(f"Base model loaded. {len(_state['voices'])} voice packs available:")
    for name, path in _state["voices"].items():
        print(f"  - {name}: {path}")


def swap_adapter(voice_name):
    """Swap the active LoRA adapter. Reloads base if voice_name is None."""
    if voice_name == _state["current_voice"]:
        return

    from mlx_lm import load
    from mlx_lm.utils import load_adapters

    if voice_name is None:
        # Reload base model without adapters
        print("Swapping to base model (no adapter)")
        model, tokenizer = load(BASE_MODEL)
        _state["model"] = model
        _state["tokenizer"] = tokenizer
        _state["current_voice"] = None
    else:
        adapter_path = _state["voices"].get(voice_name)
        if not adapter_path:
            raise ValueError(f"Unknown voice: {voice_name}")
        print(f"Swapping adapter to: {voice_name}")
        # Reload base + apply adapter
        model, tokenizer = load(BASE_MODEL, adapter_path=adapter_path)
        _state["model"] = model
        _state["tokenizer"] = tokenizer
        _state["current_voice"] = voice_name


def generate_text(prompt, max_tokens=200, temperature=0.8, top_p=0.95, repetition_penalty=1.1):
    """Generate text from the current model state."""
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler

    sampler = make_sampler(temp=temperature, top_p=top_p)

    text = generate(
        _state["model"],
        _state["tokenizer"],
        prompt=prompt,
        max_tokens=max_tokens,
        sampler=sampler,
        verbose=False,
    )

    return text


# --- FastAPI app ---

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


@asynccontextmanager
async def lifespan(app):
    load_base_model()
    yield


app = FastAPI(
    title="PersonaNexus Voice Pack API",
    description="Adapter-aware inference server for PersonaNexus voice packs",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class CompletionRequest(BaseModel):
    prompt: str
    voice: str | None = Field(None, description="Voice pack name (e.g. 'dickens', 'lincoln'). None for base model.")
    max_tokens: int = Field(200, ge=1, le=2048)
    temperature: float = Field(0.8, ge=0.0, le=2.0)
    top_p: float = Field(0.95, ge=0.0, le=1.0)
    repetition_penalty: float = Field(1.1, ge=1.0, le=2.0)
    model: str | None = Field(None, description="Ignored — always uses SmolLM2-360M. Present for OpenAI compat.")


class CompletionChoice(BaseModel):
    text: str
    index: int = 0
    finish_reason: str = "length"


class CompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: list[CompletionChoice]
    usage: CompletionUsage
    voice: str | None = None


class VoiceInfo(BaseModel):
    id: str
    name: str
    status: str = "loaded"


class VoiceListResponse(BaseModel):
    voices: list[VoiceInfo]
    active_voice: str | None


@app.get("/v1/voices")
async def list_voices() -> VoiceListResponse:
    """List all available voice packs."""
    voices = [
        VoiceInfo(id=name, name=name)
        for name, path in _state["voices"].items()
    ]
    return VoiceListResponse(voices=voices, active_voice=_state["current_voice"])


@app.post("/v1/completions")
async def create_completion(req: CompletionRequest) -> CompletionResponse:
    """Generate a completion with optional voice pack."""
    # Validate voice
    if req.voice and req.voice not in _state["voices"]:
        available = list(_state["voices"].keys())
        raise HTTPException(
            status_code=400,
            detail=f"Unknown voice '{req.voice}'. Available: {available}",
        )

    # Swap adapter if needed
    swap_adapter(req.voice)

    # Generate
    t0 = time.time()
    text = generate_text(
        prompt=req.prompt,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        repetition_penalty=req.repetition_penalty,
    )
    dt = time.time() - t0

    # Estimate token counts
    prompt_tokens = len(_state["tokenizer"].encode(req.prompt))
    completion_tokens = len(_state["tokenizer"].encode(text))

    return CompletionResponse(
        id=f"cmpl-{uuid.uuid4().hex[:12]}",
        created=int(time.time()),
        model=f"smollm2-360m+{req.voice or 'base'}",
        choices=[CompletionChoice(text=text)],
        usage=CompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
        voice=req.voice,
    )


@app.get("/v1/models")
async def list_models():
    """OpenAI-compatible models endpoint."""
    models = [{"id": "smollm2-360m+base", "object": "model", "owned_by": "personanexus"}]
    for name in _state["voices"]:
        models.append({"id": f"smollm2-360m+{name}", "object": "model", "owned_by": "personanexus"})
    return {"object": "list", "data": models}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": _state["model"] is not None,
        "active_voice": _state["current_voice"],
        "available_voices": list(_state["voices"].keys()),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8421)
