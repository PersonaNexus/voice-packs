"""PersonaNexus Voice Pack API Server.

Adapter-aware inference server. Loads the base model once, hot-swaps
LoRA adapters per request. OpenAI-compatible /v1/completions endpoint.

Usage via CLI:
    voice-packs serve --port 8080

Usage via curl:
    curl http://localhost:8080/v1/completions -X POST \\
      -H "Content-Type: application/json" \\
      -d '{"prompt": "The nature of truth", "voice": "aquinas"}'
"""

import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
except ImportError:
    raise ImportError("Install serve extras: pip install 'voice-packs[serve]'")


BASE_MODEL = os.environ.get("VOICE_PACKS_MODEL", "HuggingFaceTB/SmolLM2-360M")
VOICE_PACK_DIR = Path(os.environ.get(
    "VOICE_PACKS_DIR",
    os.path.join(os.path.dirname(os.path.dirname(__file__)))
))


# --- State ---

_state = {
    "model": None,
    "tokenizer": None,
    "voices": {},
    "current_voice": None,
}


def discover_voices() -> dict[str, str]:
    """Find all voice packs with trained adapters."""
    voices = {}
    if not VOICE_PACK_DIR.exists():
        return voices
    for entry in sorted(VOICE_PACK_DIR.iterdir()):
        adapter_file = entry / "adapters" / "adapters.safetensors"
        if entry.is_dir() and adapter_file.exists():
            voices[entry.name] = str(entry / "adapters")
    return voices


def load_base_model() -> None:
    """Load the base model and tokenizer once at startup."""
    from mlx_lm import load

    print(f"Loading base model: {BASE_MODEL}")
    model, tokenizer = load(BASE_MODEL)
    _state["model"] = model
    _state["tokenizer"] = tokenizer
    _state["voices"] = discover_voices()
    _state["current_voice"] = None
    print(f"Ready. {len(_state['voices'])} voice packs available:")
    for name in _state["voices"]:
        print(f"  - {name}")


def swap_adapter(voice_name: str | None) -> None:
    """Swap the active LoRA adapter. None = base model."""
    if voice_name == _state["current_voice"]:
        return

    from mlx_lm import load

    if voice_name is None:
        model, tokenizer = load(BASE_MODEL)
    else:
        adapter_path = _state["voices"].get(voice_name)
        if not adapter_path:
            raise ValueError(f"Unknown voice: {voice_name}")
        model, tokenizer = load(BASE_MODEL, adapter_path=adapter_path)

    _state["model"] = model
    _state["tokenizer"] = tokenizer
    _state["current_voice"] = voice_name


def generate_text(
    prompt: str,
    max_tokens: int = 200,
    temperature: float = 0.8,
    top_p: float = 0.95,
) -> str:
    """Generate text from the current model state."""
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler

    sampler = make_sampler(temp=temperature, top_p=top_p)
    return generate(
        _state["model"],
        _state["tokenizer"],
        prompt=prompt,
        max_tokens=max_tokens,
        sampler=sampler,
        verbose=False,
    )


# --- FastAPI app ---

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


# --- Models ---

class CompletionRequest(BaseModel):
    prompt: str
    voice: str | None = Field(None, description="Voice pack name (e.g. 'dickens'). None for base model.")
    max_tokens: int = Field(200, ge=1, le=2048)
    temperature: float = Field(0.8, ge=0.0, le=2.0)
    top_p: float = Field(0.95, ge=0.0, le=1.0)


class CompletionChoice(BaseModel):
    text: str
    index: int = 0
    finish_reason: str = "length"


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: list[CompletionChoice]
    voice: str | None = None


class VoiceInfo(BaseModel):
    id: str
    name: str
    status: str = "loaded"


# --- Endpoints ---

@app.post("/v1/completions")
async def create_completion(req: CompletionRequest) -> CompletionResponse:
    """Generate a completion with optional voice pack."""
    if req.voice and req.voice not in _state["voices"]:
        available = list(_state["voices"].keys())
        raise HTTPException(status_code=400, detail=f"Unknown voice '{req.voice}'. Available: {available}")

    swap_adapter(req.voice)
    text = generate_text(prompt=req.prompt, max_tokens=req.max_tokens, temperature=req.temperature, top_p=req.top_p)

    return CompletionResponse(
        id=f"cmpl-{uuid.uuid4().hex[:12]}",
        created=int(time.time()),
        model=f"smollm2-360m+{req.voice or 'base'}",
        choices=[CompletionChoice(text=text)],
        voice=req.voice,
    )


@app.get("/v1/voices")
async def list_voices():
    """List all available voice packs."""
    voices = [VoiceInfo(id=name, name=name) for name in _state["voices"]]
    return {"voices": voices, "active_voice": _state["current_voice"]}


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


def run_server(host: str = "0.0.0.0", port: int = 8080) -> None:
    """Start the voice pack server."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)
