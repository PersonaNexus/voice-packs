"""Tests for voice_packs.audit module and CLI integration."""

from __future__ import annotations

import json
import os
import textwrap

import pytest
import yaml

from voice_packs.audit import (
    build_pack_report,
    format_human_report,
    format_status_summary,
    generate_report,
    load_yaml,
    safe_get,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_repo(tmp_path):
    """Create a minimal voice-pack repo structure for testing."""
    # Write a small registry
    registry = {
        "version": "0.1",
        "base_model": "test-model",
        "framework": "mlx-lm",
        "adapter_type": "lora",
        "packs": {
            "alpha": {
                "name": "Alpha Author",
                "status": "trained",
                "category": "philosophy",
                "style": "Systematic",
                "corpus_words": 100000,
                "eval_repetition": 0.15,
                "eval_vocab_richness": 0.55,
            },
            "beta": {
                "name": "Beta Author",
                "status": "planned",
                "category": "fiction",
                "style": "Sparse",
            },
        },
    }
    reg_path = tmp_path / "registry.yaml"
    reg_path.write_text(yaml.dump(registry))

    # Create a trained pack with full assets
    alpha_dir = tmp_path / "alpha"
    (alpha_dir / "adapters").mkdir(parents=True)
    (alpha_dir / "adapters" / "adapters.safetensors").write_bytes(b"\x00")
    (alpha_dir / "eval").mkdir()
    (alpha_dir / "eval" / "metrics.json").write_text("{}")
    (alpha_dir / "samples").mkdir()
    (alpha_dir / "samples" / "samples.md").write_text("# Samples")

    # voice-pack.yaml with matching metadata
    meta = {
        "display_name": "Alpha Author",
        "status": "trained",
        "category": "philosophy",
        "corpus": {"total_words": 100000},
        "evaluation": {"repetition": 0.15, "vocab_richness": 0.55},
    }
    (alpha_dir / "voice-pack.yaml").write_text(yaml.dump(meta))

    # beta has no on-disk directory (planned)
    return tmp_path, str(reg_path)


@pytest.fixture
def tmp_repo_mismatch(tmp_path):
    """Repo where registry and voice-pack.yaml disagree on a field."""
    registry = {
        "version": "0.1",
        "base_model": "test-model",
        "framework": "mlx-lm",
        "adapter_type": "lora",
        "packs": {
            "gamma": {
                "name": "Gamma Author",
                "status": "trained",
                "category": "theology",
                "corpus_words": 50000,
            },
        },
    }
    reg_path = tmp_path / "registry.yaml"
    reg_path.write_text(yaml.dump(registry))

    gamma_dir = tmp_path / "gamma"
    (gamma_dir / "adapters").mkdir(parents=True)
    (gamma_dir / "adapters" / "adapters.safetensors").write_bytes(b"\x00")
    (gamma_dir / "eval").mkdir()
    (gamma_dir / "eval" / "metrics.json").write_text("{}")
    (gamma_dir / "samples").mkdir()
    (gamma_dir / "samples" / "samples.md").write_text("# Samples")

    # Mismatch: corpus_words differs
    meta = {
        "display_name": "Gamma Author",
        "status": "trained",
        "category": "theology",
        "corpus": {"total_words": 99999},
    }
    (gamma_dir / "voice-pack.yaml").write_text(yaml.dump(meta))

    return tmp_path, str(reg_path)


# ---------------------------------------------------------------------------
# Unit tests — safe_get
# ---------------------------------------------------------------------------

class TestSafeGet:
    def test_flat(self):
        assert safe_get({"a": 1}, "a") == 1

    def test_nested(self):
        assert safe_get({"a": {"b": 2}}, "a", "b") == 2

    def test_missing_key(self):
        assert safe_get({"a": 1}, "x") is None

    def test_missing_nested(self):
        assert safe_get({"a": 1}, "a", "b") is None

    def test_default(self):
        assert safe_get({}, "x", default=42) == 42

    def test_none_dict(self):
        assert safe_get(None, "a") is None


# ---------------------------------------------------------------------------
# Unit tests — load_yaml
# ---------------------------------------------------------------------------

def test_load_yaml_roundtrip(tmp_path):
    data = {"key": "value", "nested": {"a": 1}}
    p = tmp_path / "test.yaml"
    p.write_text(yaml.dump(data))
    assert load_yaml(str(p)) == data


# ---------------------------------------------------------------------------
# Unit tests — build_pack_report
# ---------------------------------------------------------------------------

class TestBuildPackReport:
    def test_trained_pack_clean(self, tmp_repo):
        repo_root, reg_path = tmp_repo
        registry = load_yaml(reg_path)
        report = build_pack_report("alpha", registry["packs"]["alpha"], repo_root=str(repo_root))

        assert report["id"] == "alpha"
        assert report["display_name"] == "Alpha Author"
        assert report["status"] == "trained"
        assert report["issues"] == []
        assert report["mismatches"] == []
        assert report["assets"]["adapter_weights"] is True
        assert report["assets"]["voice_pack_yaml"] is True

    def test_planned_pack_no_issues(self, tmp_repo):
        repo_root, reg_path = tmp_repo
        registry = load_yaml(reg_path)
        report = build_pack_report("beta", registry["packs"]["beta"], repo_root=str(repo_root))

        assert report["status"] == "planned"
        # Planned packs don't require assets
        assert report["issues"] == []

    def test_trained_missing_assets(self, tmp_path):
        """A trained pack with no on-disk assets should report issues."""
        registry = {
            "packs": {
                "empty": {
                    "name": "Empty Pack",
                    "status": "trained",
                    "category": "test",
                }
            }
        }
        report = build_pack_report("empty", registry["packs"]["empty"], repo_root=str(tmp_path))
        assert report["status"] == "trained"
        assert len(report["issues"]) > 0
        assert any("missing required" in i for i in report["issues"])
        assert any("no voice-pack.yaml" in i for i in report["issues"])

    def test_mismatch_detected(self, tmp_repo_mismatch):
        repo_root, reg_path = tmp_repo_mismatch
        registry = load_yaml(reg_path)
        report = build_pack_report("gamma", registry["packs"]["gamma"], repo_root=str(repo_root))

        assert len(report["mismatches"]) == 1
        assert report["mismatches"][0]["field"] == "corpus_words"
        assert any("mismatches" in i for i in report["issues"])


# ---------------------------------------------------------------------------
# Integration tests — generate_report
# ---------------------------------------------------------------------------

class TestGenerateReport:
    def test_full_report_structure(self, tmp_repo):
        repo_root, reg_path = tmp_repo
        report = generate_report(registry_path=reg_path, repo_root=str(repo_root))

        assert report["base_model"] == "test-model"
        assert report["summary"]["total_packs"] == 2
        assert report["summary"]["trained_packs"] == 1
        assert report["summary"]["planned_packs"] == 1
        assert len(report["packs"]) == 2

    def test_clean_repo_no_issues(self, tmp_repo):
        repo_root, reg_path = tmp_repo
        report = generate_report(registry_path=reg_path, repo_root=str(repo_root))
        assert report["summary"]["total_issues"] == 0
        assert report["summary"]["total_mismatches"] == 0

    def test_mismatch_counted(self, tmp_repo_mismatch):
        repo_root, reg_path = tmp_repo_mismatch
        report = generate_report(registry_path=reg_path, repo_root=str(repo_root))
        assert report["summary"]["total_mismatches"] == 1
        assert report["summary"]["total_issues"] > 0

    def test_json_serializable(self, tmp_repo):
        repo_root, reg_path = tmp_repo
        report = generate_report(registry_path=reg_path, repo_root=str(repo_root))
        # Should not raise
        output = json.dumps(report)
        assert isinstance(json.loads(output), dict)


# ---------------------------------------------------------------------------
# Integration tests — format_human_report
# ---------------------------------------------------------------------------

class TestFormatHumanReport:
    def test_contains_header(self, tmp_repo):
        repo_root, reg_path = tmp_repo
        report = generate_report(registry_path=reg_path, repo_root=str(repo_root))
        text = format_human_report(report)
        assert "PersonaNexus Voice Pack Audit" in text
        assert "test-model" in text
        assert "alpha" in text
        assert "beta" in text

    def test_no_issues_shown(self, tmp_repo):
        repo_root, reg_path = tmp_repo
        report = generate_report(registry_path=reg_path, repo_root=str(repo_root))
        text = format_human_report(report)
        assert "issues:   none" in text


class TestFormatStatusSummary:
    def test_pass_for_clean_repo(self, tmp_repo):
        repo_root, reg_path = tmp_repo
        report = generate_report(registry_path=reg_path, repo_root=str(repo_root))
        text = format_status_summary(report)
        assert text.startswith("PASS |")
        assert "issues=0" in text
        assert "mismatches=0" in text

    def test_fail_when_issues_exist(self, tmp_repo_mismatch):
        repo_root, reg_path = tmp_repo_mismatch
        report = generate_report(registry_path=reg_path, repo_root=str(repo_root))
        text = format_status_summary(report)
        assert text.startswith("FAIL |")
        assert "mismatches=1" in text


# ---------------------------------------------------------------------------
# CLI integration tests
# ---------------------------------------------------------------------------

class TestCLI:
    def test_audit_human(self, tmp_repo, capsys):
        from voice_packs.cli import main
        import sys

        repo_root, reg_path = tmp_repo
        sys.argv = ["voice-packs", "audit", "--registry", reg_path, "--repo-root", str(repo_root)]
        main()
        captured = capsys.readouterr()
        assert "PersonaNexus Voice Pack Audit" in captured.out

    def test_audit_json(self, tmp_repo, capsys):
        from voice_packs.cli import main
        import sys

        repo_root, reg_path = tmp_repo
        sys.argv = ["voice-packs", "audit", "--json", "--registry", reg_path, "--repo-root", str(repo_root)]
        main()
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["summary"]["total_packs"] == 2

    def test_audit_summary(self, tmp_repo, capsys):
        from voice_packs.cli import main
        import sys

        repo_root, reg_path = tmp_repo
        sys.argv = ["voice-packs", "audit", "--summary", "--registry", reg_path, "--repo-root", str(repo_root)]
        main()
        captured = capsys.readouterr()
        assert captured.out.strip().startswith("PASS |")

    def test_audit_strict_clean(self, tmp_repo):
        from voice_packs.cli import main
        import sys

        repo_root, reg_path = tmp_repo
        sys.argv = ["voice-packs", "audit", "--strict", "--registry", reg_path, "--repo-root", str(repo_root)]
        # Clean repo — should not raise
        main()

    def test_audit_strict_fails_on_mismatch(self, tmp_repo_mismatch):
        from voice_packs.cli import main
        import sys

        repo_root, reg_path = tmp_repo_mismatch
        sys.argv = ["voice-packs", "audit", "--strict", "--registry", reg_path, "--repo-root", str(repo_root)]
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1
