"""Train a LoRA voice pack adapter."""

import os
import subprocess
import sys


DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-360M"


def train(
    data_dir: str,
    adapter_path: str,
    model: str = DEFAULT_MODEL,
    iters: int = 1000,
    batch_size: int = 4,
    num_layers: int = 12,
    learning_rate: float = 5e-5,
    save_every: int = 500,
) -> bool:
    """Train a LoRA adapter using mlx-lm."""
    os.makedirs(adapter_path, exist_ok=True)

    cmd = [
        sys.executable, "-m", "mlx_lm.lora",
        "--model", model,
        "--train",
        "--data", data_dir,
        "--batch-size", str(batch_size),
        "--num-layers", str(num_layers),
        "--iters", str(iters),
        "--learning-rate", str(learning_rate),
        "--adapter-path", adapter_path,
        "--save-every", str(save_every),
        "--val-batches", "5",
    ]

    print(f"Training voice pack...")
    print(f"  Model: {model}")
    print(f"  Data: {data_dir}")
    print(f"  Iterations: {iters}")
    print(f"  Output: {adapter_path}")
    print()

    result = subprocess.run(cmd)
    return result.returncode == 0
