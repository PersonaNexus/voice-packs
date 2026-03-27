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

try:
    import yaml  # type: ignore
except ModuleNotFoundError:
    yaml = None

REGISTRY_PATH = os.path.join(os.path.dirname(__file__), "registry.yaml")
REPO_ROOT = os.path.dirname(__file__)
BASE_MODEL = "HuggingFaceTB/SmolLM2-360M"
DATA_BASE = os.path.expanduser("~/personal-ai-org/catholic-finetune/data")


def load_yaml(path: str) -> dict[str, Any]:
    if yaml is not None:
        with open(path) as f:
            return yaml.safe_load(f) or {}

    ruby = subprocess.run(
        [
            "/usr/bin/ruby",
            "-rjson",
            "-ryaml",
            "-e",
            "print JSON.dump(YAML.load_file(ARGV[0]) || {})",
            path,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(ruby.stdout or "{}")


def load_registry() -> dict[str, Any]:
    return load_yaml(REGISTRY_PATH)


def safe_get(dct: dict[str, Any] | None, *keys: str, default=None):
    current = dct or {}
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


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


def build_pack_report(pack_name: str, registry_pack: dict[str, Any]) -> dict[str, Any]:
    pack_dir = os.path.join(REPO_ROOT, pack_name)
    meta_path = os.path.join(pack_dir, "voice-pack.yaml")
    meta = load_yaml(meta_path) if os.path.exists(meta_path) else None

    assets = {
        "voice_pack_yaml": os.path.exists(meta_path),
        "adapters_dir": os.path.isdir(os.path.join(pack_dir, "adapters")),
        "adapter_weights": os.path.exists(os.path.join(pack_dir, "adapters", "adapters.safetensors")),
        "eval_dir": os.path.isdir(os.path.join(pack_dir, "eval")),
        "eval_metrics": os.path.exists(os.path.join(pack_dir, "eval", "metrics.json")),
        "samples_dir": os.path.isdir(os.path.join(pack_dir, "samples")),
        "samples_markdown": os.path.exists(os.path.join(pack_dir, "samples", "samples.md")),
    }

    mismatches = []
    if meta:
        comparisons = [
            ("name", registry_pack.get("name"), meta.get("display_name") or meta.get("name")),
            ("status", registry_pack.get("status"), meta.get("status")),
            ("category", registry_pack.get("category"), meta.get("category")),
            ("corpus_words", registry_pack.get("corpus_words"), safe_get(meta, "corpus", "total_words")),
            ("eval_repetition", registry_pack.get("eval_repetition"), safe_get(meta, "evaluation", "repetition")),
            ("eval_vocab_richness", registry_pack.get("eval_vocab_richness"), safe_get(meta, "evaluation", "vocab_richness")),
            ("eval_theological_density", registry_pack.get("eval_theological_density"), safe_get(meta, "evaluation", "theological_density")),
        ]
        for field, registry_value, meta_value in comparisons:
            if registry_value is None or meta_value is None:
                continue
            if registry_value != meta_value:
                mismatches.append(
                    {
                        "field": field,
                        "registry": registry_value,
                        "voice_pack_yaml": meta_value,
                    }
                )

    issues = []
    status = registry_pack.get("status", "unknown")
    if status == "trained":
        for key in ("voice_pack_yaml", "adapter_weights", "eval_metrics", "samples_markdown"):
            if not assets[key]:
                issues.append(f"missing required trained-pack asset: {key}")
    if meta is None and status == "trained":
        issues.append("trained pack has no voice-pack.yaml metadata")
    if mismatches:
        issues.append(f"metadata mismatches: {len(mismatches)}")

    return {
        "id": pack_name,
        "display_name": registry_pack.get("name") or safe_get(meta, "display_name") or pack_name,
        "status": status,
        "category": registry_pack.get("category") or safe_get(meta, "category"),
        "style": registry_pack.get("style") or safe_get(meta, "style", "tone"),
        "corpus_words": registry_pack.get("corpus_words") or safe_get(meta, "corpus", "total_words"),
        "metrics": {
            "repetition": registry_pack.get("eval_repetition", safe_get(meta, "evaluation", "repetition")),
            "vocab_richness": registry_pack.get("eval_vocab_richness", safe_get(meta, "evaluation", "vocab_richness")),
            "theological_density": registry_pack.get("eval_theological_density", safe_get(meta, "evaluation", "theological_density")),
        },
        "assets": assets,
        "mismatches": mismatches,
        "issues": issues,
    }


def generate_report() -> dict[str, Any]:
    registry = load_registry()
    packs = [build_pack_report(pack_name, pack) for pack_name, pack in registry.get("packs", {}).items()]
    trained = [p for p in packs if p["status"] == "trained"]
    planned = [p for p in packs if p["status"] != "trained"]
    issue_count = sum(len(p["issues"]) for p in packs)
    mismatch_count = sum(len(p["mismatches"]) for p in packs)

    return {
        "repo": "PersonaNexus Voice Packs",
        "base_model": registry.get("base_model"),
        "framework": registry.get("framework"),
        "adapter_type": registry.get("adapter_type"),
        "summary": {
            "total_packs": len(packs),
            "trained_packs": len(trained),
            "planned_packs": len(planned),
            "packs_with_issues": len([p for p in packs if p["issues"]]),
            "total_issues": issue_count,
            "total_mismatches": mismatch_count,
        },
        "packs": packs,
    }


def print_human_report(report: dict[str, Any]) -> None:
    summary = report["summary"]
    print("PersonaNexus Voice Pack Audit")
    print("=" * 80)
    print(f"Base model:   {report.get('base_model')}")
    print(f"Framework:    {report.get('framework')}")
    print(f"Adapter type: {report.get('adapter_type')}")
    print(
        f"Packs:        {summary['total_packs']} total | "
        f"{summary['trained_packs']} trained | {summary['planned_packs']} planned"
    )
    print(
        f"Health:       {summary['packs_with_issues']} packs with issues | "
        f"{summary['total_mismatches']} metadata mismatches"
    )

    for pack in report["packs"]:
        print("\n" + "-" * 80)
        print(f"{pack['id']} — {pack['display_name']}")
        print(f"  status:   {pack['status']}")
        print(f"  category: {pack['category'] or '-'}")
        print(f"  corpus:   {pack['corpus_words'] or '-'} words")
        metrics = pack["metrics"]
        metric_text = ", ".join(
            f"{label}={value}"
            for label, value in metrics.items()
            if value is not None
        ) or "-"
        print(f"  metrics:  {metric_text}")
        assets = ", ".join(
            f"{name}={'yes' if exists else 'no'}" for name, exists in pack["assets"].items()
        )
        print(f"  assets:   {assets}")
        if pack["mismatches"]:
            print("  mismatches:")
            for mismatch in pack["mismatches"]:
                print(
                    f"    - {mismatch['field']}: registry={mismatch['registry']!r} "
                    f"voice-pack.yaml={mismatch['voice_pack_yaml']!r}"
                )
        if pack["issues"]:
            print("  issues:")
            for issue in pack["issues"]:
                print(f"    - {issue}")
        else:
            print("  issues:   none")


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
