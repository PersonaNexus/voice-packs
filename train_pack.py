#!/usr/bin/env python3
"""
Train a new voice pack from a PersonaNexus corpus.

Usage:
    python train_pack.py aquinas          # train from existing data
    python train_pack.py newman --prepare # prepare data + train
    python train_pack.py --list           # list available/planned packs
"""

import argparse
import json
import os
import subprocess
import sys

import yaml

REGISTRY_PATH = os.path.join(os.path.dirname(__file__), "registry.yaml")
BASE_MODEL = "HuggingFaceTB/SmolLM2-360M"
DATA_BASE = os.path.expanduser("~/personal-ai-org/catholic-finetune/data")


def load_registry():
    with open(REGISTRY_PATH) as f:
        return yaml.safe_load(f)


def list_packs():
    reg = load_registry()
    print("PersonaNexus Voice Packs:")
    print(f"{'Name':<20} {'Status':<10} {'Category':<25} {'Style'}")
    print("-" * 80)
    for name, pack in reg["packs"].items():
        print(f"{name:<20} {pack['status']:<10} {pack.get('category', ''):<25} {pack.get('style', '')[:40]}")


def train_pack(pack_name, prepare=False, iters=1000, lr=5e-5, layers=12, batch=4):
    reg = load_registry()
    if pack_name not in reg["packs"]:
        print(f"Error: Unknown pack '{pack_name}'. Use --list to see available packs.")
        sys.exit(1)

    pack = reg["packs"][pack_name]
    data_dir = os.path.join(DATA_BASE, f"personality-{pack_name}")
    adapter_dir = os.path.join(os.path.dirname(__file__), pack_name, "adapters")

    if not os.path.exists(data_dir):
        if prepare:
            print(f"Preparing data for {pack_name}...")
            # Would call prepare_personalities.py with appropriate args
            print("Data preparation not yet implemented for this pack. Prepare manually.")
            sys.exit(1)
        else:
            print(f"Error: No training data at {data_dir}")
            print(f"Run with --prepare to generate data, or prepare manually.")
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
        "--save-every", str(iters // 2),
    ]

    print(f"\nRunning: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=os.path.expanduser("~/personal-ai-org/catholic-finetune"))
    if result.returncode == 0:
        print(f"\nVoice pack '{pack_name}' trained successfully!")
        print(f"Adapters saved to: {adapter_dir}")
    else:
        print(f"\nTraining failed with exit code {result.returncode}")


def generate_sample(pack_name, prompt="The nature of truth is", tokens=200, temp=0.7):
    adapter_dir = os.path.join(os.path.dirname(__file__), pack_name, "adapters")
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PersonaNexus voice packs")
    parser.add_argument("pack", nargs="?", help="Voice pack name to train")
    parser.add_argument("--list", action="store_true", help="List all packs")
    parser.add_argument("--prepare", action="store_true", help="Prepare training data")
    parser.add_argument("--generate", type=str, help="Generate sample with prompt")
    parser.add_argument("--iters", type=int, default=1000, help="Training iterations")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    args = parser.parse_args()

    if args.list:
        list_packs()
    elif args.pack and args.generate:
        generate_sample(args.pack, prompt=args.generate)
    elif args.pack:
        train_pack(args.pack, prepare=args.prepare, iters=args.iters, lr=args.lr)
    else:
        parser.print_help()
