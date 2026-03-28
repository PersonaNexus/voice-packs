"""Generate text using a voice pack adapter."""

import subprocess
import sys


DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-360M"


def generate(
    adapter_path: str,
    prompt: str,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 200,
    temp: float = 0.7,
) -> str:
    """Generate text with a LoRA adapter."""
    cmd = [
        sys.executable, "-m", "mlx_lm.generate",
        "--model", model,
        "--adapter-path", adapter_path,
        "--max-tokens", str(max_tokens),
        "--temp", str(temp),
        "--prompt", prompt,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    parts = result.stdout.split("==========")
    if len(parts) >= 3:
        return parts[1].strip()
    return result.stdout.strip()
