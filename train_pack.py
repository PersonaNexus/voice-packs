#!/usr/bin/env python3
"""
Train, inspect, and report on PersonaNexus voice packs.

Usage:
    python train_pack.py aquinas                   # train from existing data
    python train_pack.py newman --prepare          # prepare data + train
    python train_pack.py --list                    # list available/planned packs
    python train_pack.py --report                  # human-readable report
    python train_pack.py --report --json           # machine-readable report
    python train_pack.py --report --strict         # exit non-zero on mismatches/issues
"""

import argparse
import json
import os
import subprocess
import sys
from typing import Any

from voice_packs.audit import (
    generate_report,
    load_registry,
    load_yaml,
    print_human_report,
    safe_get,
)

REGISTRY_PATH = os.path.join(os.path.dirname(__file__), "registry.yaml")
REPO_ROOT = os.path.dirname(__file__)
BASE_MODEL = "HuggingFaceTB/SmolLM2-360M"
DATA_BASE = os.path.expanduser("~/personal-ai-org/catholic-finetune/data")


def list_packs() -> None:
    reg = load_registry()
    print("PersonaNexus Voice Packs:")
    print(f"{'Name':<20} {'Status':<10} {'Category':<25} {'Style'}")
    print("-" * 100)
    for name, pack in reg["packs"].items():
        print(
            f"{name:<20} {pack['status']:<10} {pack.get('category', ''):<25} {pack.get('style', '')[:50]}"
        )


def train_pack(pack_name: str, prepare: bool = False, iters: int = 1000, lr: float = 5e-5, layers: int = 12, batch: int = 4) -> None:
    reg = load_registry()
    if pack_name not in reg["packs"]:
        print(f"Error: Unknown pack '{pack_name}'. Use --list to see available packs.")
        sys.exit(1)

    pack = reg["packs"][pack_name]
    data_dir = os.path.join(DATA_BASE, f"personality-{pack_name}")
    adapter_dir = os.path.join(REPO_ROOT, pack_name, "adapters")

    if not os.path.exists(data_dir):
        if prepare:
            print(f"Preparing data for {pack_name}...")
            print("Data preparation not yet implemented for this pack. Prepare manually.")
            sys.exit(1)
        else:
            print(f"Error: No training data at {data_dir}")
            print("Run with --prepare to generate data, or prepare manually.")
            sys.exit(1)

    os.makedirs(adapter_dir, exist_ok=True)

    print(f"Training voice pack: {pack_name}")
    print(f"  Base model: {BASE_MODEL}")
    print(f"  Data: {data_dir}")
    print(f"  Iterations: {iters}")
    print(f"  Learning rate: {lr}")
    print(f"  LoRA layers: {layers}")

    cmd = [
        "uv", "run", "mlx_lm.lora",
        "--model", BASE_MODEL,
        "--train",
        "--data", data_dir,
        "--batch-size", str(batch),
        "--num-layers", str(layers),
        "--iters", str(iters),
        "--learning-rate", str(lr),
        "--adapter-path", adapter_dir,
        "--save-every", str(max(iters // 2, 1)),
    ]

    print(f"\nRunning: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=os.path.expanduser("~/personal-ai-org/catholic-finetune"))
    if result.returncode == 0:
        print(f"\nVoice pack '{pack_name}' trained successfully!")
        print(f"Adapters saved to: {adapter_dir}")
    else:
        print(f"\nTraining failed with exit code {result.returncode}")


def generate_sample(pack_name: str, prompt: str = "The nature of truth is", tokens: int = 200, temp: float = 0.7) -> None:
    adapter_dir = os.path.join(REPO_ROOT, pack_name, "adapters")
    if not os.path.exists(os.path.join(adapter_dir, "adapters.safetensors")):
        print(f"Error: No trained adapter for '{pack_name}'")
        sys.exit(1)

    cmd = [
        "uv", "run", "mlx_lm.generate",
        "--model", BASE_MODEL,
        "--adapter-path", adapter_dir,
        "--max-tokens", str(tokens),
        "--temp", str(temp),
        "--prompt", prompt,
    ]

    subprocess.run(cmd, cwd=os.path.expanduser("~/personal-ai-org/catholic-finetune"))



# Report functions imported from voice_packs.audit:
# build_pack_report, generate_report, print_human_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PersonaNexus voice packs")
    parser.add_argument("pack", nargs="?", help="Voice pack name to train")
    parser.add_argument("--list", action="store_true", help="List all packs")
    parser.add_argument("--prepare", action="store_true", help="Prepare training data")
    parser.add_argument("--generate", type=str, help="Generate sample with prompt")
    parser.add_argument("--iters", type=int, default=1000, help="Training iterations")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--report", action="store_true", help="Generate a voice pack audit report")
    parser.add_argument("--json", action="store_true", help="Emit report as JSON (use with --report)")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero if report finds issues or mismatches")
    args = parser.parse_args()

    if args.list:
        list_packs()
    elif args.report:
        report = generate_report()
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print_human_report(report)
        if args.strict:
            summary = report["summary"]
            if summary["total_issues"] > 0 or summary["total_mismatches"] > 0:
                sys.exit(1)
    elif args.pack and args.generate:
        generate_sample(args.pack, prompt=args.generate)
    elif args.pack:
        train_pack(args.pack, prepare=args.prepare, iters=args.iters, lr=args.lr)
    else:
        parser.print_help()
