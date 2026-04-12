"""Audit and reporting for PersonaNexus voice packs.

Inspects the registry, per-pack metadata, and on-disk assets to produce
a structured health report.  Used by both the CLI (`voice-packs audit`)
and the legacy `train_pack.py --report` entry-point.
"""

from __future__ import annotations

import json
import os
from typing import Any

try:
    import yaml  # type: ignore
except ModuleNotFoundError:
    yaml = None

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DEFAULT_REGISTRY = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "registry.yaml"
)
_DEFAULT_REPO_ROOT = os.path.dirname(os.path.dirname(__file__))


def load_yaml(path: str) -> dict[str, Any]:
    """Load a YAML file safely."""
    if yaml is None:
        raise ImportError("PyYAML is required: pip install pyyaml")
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_registry(registry_path: str = _DEFAULT_REGISTRY) -> dict[str, Any]:
    """Load the voice-pack registry YAML."""
    return load_yaml(registry_path)


def safe_get(dct: dict[str, Any] | None, *keys: str, default=None):
    """Drill into nested dicts, returning *default* on any missing key."""
    current = dct or {}
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


# ---------------------------------------------------------------------------
# Report building
# ---------------------------------------------------------------------------


def build_pack_report(
    pack_name: str,
    registry_pack: dict[str, Any],
    repo_root: str = _DEFAULT_REPO_ROOT,
) -> dict[str, Any]:
    """Build a structured audit report for a single voice pack."""
    pack_dir = os.path.join(repo_root, pack_name)
    meta_path = os.path.join(pack_dir, "voice-pack.yaml")
    meta = load_yaml(meta_path) if os.path.exists(meta_path) else None

    assets = {
        "voice_pack_yaml": os.path.exists(meta_path),
        "adapters_dir": os.path.isdir(os.path.join(pack_dir, "adapters")),
        "adapter_weights": os.path.exists(
            os.path.join(pack_dir, "adapters", "adapters.safetensors")
        ),
        "eval_dir": os.path.isdir(os.path.join(pack_dir, "eval")),
        "eval_metrics": os.path.exists(
            os.path.join(pack_dir, "eval", "metrics.json")
        ),
        "samples_dir": os.path.isdir(os.path.join(pack_dir, "samples")),
        "samples_markdown": os.path.exists(
            os.path.join(pack_dir, "samples", "samples.md")
        ),
    }

    mismatches: list[dict[str, Any]] = []
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

    issues: list[str] = []
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


def generate_report(
    registry_path: str = _DEFAULT_REGISTRY,
    repo_root: str = _DEFAULT_REPO_ROOT,
    statuses: list[str] | None = None,
    issues_only: bool = False,
) -> dict[str, Any]:
    """Generate a full audit report across all registered voice packs."""
    registry = load_registry(registry_path)
    packs = [
        build_pack_report(name, pack, repo_root=repo_root)
        for name, pack in registry.get("packs", {}).items()
    ]
    if statuses:
        allowed = set(statuses)
        packs = [p for p in packs if p["status"] in allowed]
    if issues_only:
        packs = [p for p in packs if p["issues"]]

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
        "status_summary": _build_status_summary(packs),
        "packs": packs,
    }


# ---------------------------------------------------------------------------
# Status summary / leaderboard layer
# ---------------------------------------------------------------------------


def _build_status_summary(packs: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute leaderboard rankings, issue buckets, and category splits.

    Returned as a new top-level field so existing consumers are unaffected.
    """
    trained = [p for p in packs if p["status"] == "trained"]
    planned = [p for p in packs if p["status"] != "trained"]

    # --- Leaderboards (trained packs only, where metrics exist) ---
    def _ranked(key: str, *, ascending: bool = True) -> list[dict[str, Any]]:
        eligible = [
            p for p in trained if p["metrics"].get(key) is not None
        ]
        eligible.sort(key=lambda p: p["metrics"][key], reverse=not ascending)
        return [
            {"id": p["id"], "display_name": p["display_name"], "value": p["metrics"][key]}
            for p in eligible
        ]

    leaderboards = {
        "vocab_richness_top": _ranked("vocab_richness", ascending=False),
        "repetition_lowest": _ranked("repetition", ascending=True),
    }

    # --- Issue buckets ---
    issue_buckets: dict[str, list[str]] = {}
    for p in packs:
        for issue in p["issues"]:
            # Normalise to a bucket key (first two meaningful words)
            if "missing required" in issue:
                bucket = "missing_assets"
            elif "mismatches" in issue:
                bucket = "metadata_mismatch"
            elif "no voice-pack.yaml" in issue:
                bucket = "missing_metadata"
            else:
                bucket = "other"
            issue_buckets.setdefault(bucket, []).append(p["id"])

    # --- Category splits ---
    def _category_split(pack_list: list[dict[str, Any]]) -> dict[str, list[str]]:
        by_cat: dict[str, list[str]] = {}
        for p in pack_list:
            cat = p.get("category") or "uncategorized"
            by_cat.setdefault(cat, []).append(p["id"])
        return by_cat

    return {
        "leaderboards": leaderboards,
        "issue_buckets": issue_buckets,
        "trained_by_category": _category_split(trained),
        "planned_by_category": _category_split(planned),
    }


# ---------------------------------------------------------------------------
# Human-readable output
# ---------------------------------------------------------------------------


def format_human_report(report: dict[str, Any]) -> str:
    """Return the audit report as a human-readable string."""
    lines: list[str] = []
    summary = report["summary"]
    lines.append("PersonaNexus Voice Pack Audit")
    lines.append("=" * 80)
    lines.append(f"Base model:   {report.get('base_model')}")
    lines.append(f"Framework:    {report.get('framework')}")
    lines.append(f"Adapter type: {report.get('adapter_type')}")
    lines.append(
        f"Packs:        {summary['total_packs']} total | "
        f"{summary['trained_packs']} trained | {summary['planned_packs']} planned"
    )
    lines.append(
        f"Health:       {summary['packs_with_issues']} packs with issues | "
        f"{summary['total_mismatches']} metadata mismatches"
    )

    for pack in report["packs"]:
        lines.append("")
        lines.append("-" * 80)
        lines.append(f"{pack['id']} — {pack['display_name']}")
        lines.append(f"  status:   {pack['status']}")
        lines.append(f"  category: {pack['category'] or '-'}")
        lines.append(f"  corpus:   {pack['corpus_words'] or '-'} words")
        metrics = pack["metrics"]
        metric_text = ", ".join(
            f"{label}={value}"
            for label, value in metrics.items()
            if value is not None
        ) or "-"
        lines.append(f"  metrics:  {metric_text}")
        assets = ", ".join(
            f"{name}={'yes' if exists else 'no'}"
            for name, exists in pack["assets"].items()
        )
        lines.append(f"  assets:   {assets}")
        if pack["mismatches"]:
            lines.append("  mismatches:")
            for mismatch in pack["mismatches"]:
                lines.append(
                    f"    - {mismatch['field']}: registry={mismatch['registry']!r} "
                    f"voice-pack.yaml={mismatch['voice_pack_yaml']!r}"
                )
        if pack["issues"]:
            lines.append("  issues:")
            for issue in pack["issues"]:
                lines.append(f"    - {issue}")
        else:
            lines.append("  issues:   none")

    return "\n".join(lines)


def print_human_report(report: dict[str, Any]) -> None:
    """Print the human-readable report to stdout."""
    print(format_human_report(report))


# ---------------------------------------------------------------------------
# Markdown summary output
# ---------------------------------------------------------------------------


def format_markdown_summary(report: dict[str, Any]) -> str:
    """Return a concise Markdown summary suitable for dashboards / morning review."""
    lines: list[str] = []
    summary = report["summary"]
    ss = report.get("status_summary", {})

    lines.append("# Voice Pack Audit Summary")
    lines.append("")
    lines.append(
        f"**{summary['trained_packs']}** trained / "
        f"**{summary['planned_packs']}** planned / "
        f"**{summary['total_issues']}** issues"
    )

    # Leaderboards
    leaderboards = ss.get("leaderboards", {})
    vocab_top = leaderboards.get("vocab_richness_top", [])
    rep_low = leaderboards.get("repetition_lowest", [])

    if vocab_top:
        lines.append("")
        lines.append("## Vocab Richness (top)")
        lines.append("")
        lines.append("| Rank | Pack | Score |")
        lines.append("|------|------|-------|")
        for i, entry in enumerate(vocab_top[:5], 1):
            lines.append(f"| {i} | {entry['display_name']} (`{entry['id']}`) | {entry['value']} |")

    if rep_low:
        lines.append("")
        lines.append("## Lowest Repetition")
        lines.append("")
        lines.append("| Rank | Pack | Score |")
        lines.append("|------|------|-------|")
        for i, entry in enumerate(rep_low[:5], 1):
            lines.append(f"| {i} | {entry['display_name']} (`{entry['id']}`) | {entry['value']} |")

    # Issue buckets
    issue_buckets = ss.get("issue_buckets", {})
    if issue_buckets:
        lines.append("")
        lines.append("## Issue Buckets")
        lines.append("")
        for bucket, pack_ids in sorted(issue_buckets.items()):
            lines.append(f"- **{bucket}**: {', '.join(sorted(set(pack_ids)))}")

    # Category splits
    for label, key in [("Trained", "trained_by_category"), ("Planned", "planned_by_category")]:
        by_cat = ss.get(key, {})
        if by_cat:
            lines.append("")
            lines.append(f"## {label} by Category")
            lines.append("")
            for cat, ids in sorted(by_cat.items()):
                lines.append(f"- **{cat}**: {', '.join(ids)}")

    lines.append("")
    return "\n".join(lines)
