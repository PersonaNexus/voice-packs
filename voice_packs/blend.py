"""Blend two voice pack adapters by interpolating weights."""

import os
import shutil

import numpy as np
from safetensors.numpy import load_file, save_file


def blend(
    adapter_a: str,
    adapter_b: str,
    output_path: str,
    ratio: float = 0.5,
) -> str:
    """Linearly interpolate two adapter weight files.

    Args:
        adapter_a: Path to first adapter directory
        adapter_b: Path to second adapter directory
        output_path: Where to save blended adapter
        ratio: Blend ratio (1.0 = 100% A, 0.0 = 100% B)

    Returns:
        Path to blended adapter
    """
    os.makedirs(output_path, exist_ok=True)

    weights_a = load_file(os.path.join(adapter_a, "adapters.safetensors"))
    weights_b = load_file(os.path.join(adapter_b, "adapters.safetensors"))

    blended = {}
    for key in weights_a:
        if key in weights_b:
            blended[key] = (ratio * weights_a[key] + (1 - ratio) * weights_b[key]).astype(
                weights_a[key].dtype
            )
        else:
            blended[key] = weights_a[key]

    for key in weights_b:
        if key not in blended:
            blended[key] = weights_b[key]

    save_file(blended, os.path.join(output_path, "adapters.safetensors"))

    # Copy adapter config from A
    config_src = os.path.join(adapter_a, "adapter_config.json")
    if os.path.exists(config_src):
        shutil.copy(config_src, os.path.join(output_path, "adapter_config.json"))

    return output_path
