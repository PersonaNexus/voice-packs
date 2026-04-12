"""Tests for voice_packs.audit module and CLI integration."""

from __future__ import annotations

import json

import pytest
import yaml

from voice_packs.audit import (
    build_pack_report,
    format_human_report,
    format_markdown_summary,
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

    def test_status_filter(self, tmp_repo):
        repo_root, reg_path = tmp_repo
        report = generate_report(
            registry_path=reg_path,
            repo_root=str(repo_root),
            statuses=["trained"],
        )
        assert report["summary"]["total_packs"] == 1
        assert [pack["id"] for pack in report["packs"]] == ["alpha"]

    def test_issues_only_filter(self, tmp_repo_mismatch):
        repo_root, reg_path = tmp_repo_mismatch
        report = generate_report(
            registry_path=reg_path,
            repo_root=str(repo_root),
            issues_only=True,
        )
        assert report["summary"]["total_packs"] == 1
        assert [pack["id"] for pack in report["packs"]] == ["gamma"]


# ---------------------------------------------------------------------------
# Integration tests — status_summary (leaderboards, buckets, category splits)
# ---------------------------------------------------------------------------


class TestStatusSummary:
    def test_present_in_report(self, tmp_repo):
        repo_root, reg_path = tmp_repo
        report = generate_report(registry_path=reg_path, repo_root=str(repo_root))
        assert "status_summary" in report
        ss = report["status_summary"]
        assert "leaderboards" in ss
        assert "issue_buckets" in ss
        assert "trained_by_category" in ss
        assert "planned_by_category" in ss

    def test_leaderboard_ordering(self, tmp_repo):
        repo_root, reg_path = tmp_repo
        report = generate_report(registry_path=reg_path, repo_root=str(repo_root))
        ss = report["status_summary"]
        vocab_top = ss["leaderboards"]["vocab_richness_top"]
        # Only alpha has metrics in this fixture
        assert len(vocab_top) == 1
        assert vocab_top[0]["id"] == "alpha"
        assert vocab_top[0]["value"] == 0.55

    def test_category_splits(self, tmp_repo):
        repo_root, reg_path = tmp_repo
        report = generate_report(registry_path=reg_path, repo_root=str(repo_root))
        ss = report["status_summary"]
        assert "philosophy" in ss["trained_by_category"]
        assert "alpha" in ss["trained_by_category"]["philosophy"]
        assert "fiction" in ss["planned_by_category"]
        assert "beta" in ss["planned_by_category"]["fiction"]

    def test_issue_buckets_on_mismatch(self, tmp_repo_mismatch):
        repo_root, reg_path = tmp_repo_mismatch
        report = generate_report(registry_path=reg_path, repo_root=str(repo_root))
        ss = report["status_summary"]
        assert "metadata_mismatch" in ss["issue_buckets"]
        assert "gamma" in ss["issue_buckets"]["metadata_mismatch"]

    def test_issue_buckets_missing_assets(self, tmp_path):
        """Trained pack with no assets produces missing_assets + missing_metadata buckets."""
        registry = {
            "version": "0.1",
            "base_model": "m",
            "framework": "f",
            "adapter_type": "a",
            "packs": {
                "empty": {"name": "Empty", "status": "trained", "category": "test"},
            },
        }
        reg_path = tmp_path / "registry.yaml"
        reg_path.write_text(yaml.dump(registry))
        report = generate_report(registry_path=str(reg_path), repo_root=str(tmp_path))
        buckets = report["status_summary"]["issue_buckets"]
        assert "missing_assets" in buckets
        assert "missing_metadata" in buckets

    def test_clean_repo_empty_buckets(self, tmp_repo):
        repo_root, reg_path = tmp_repo
        report = generate_report(registry_path=reg_path, repo_root=str(repo_root))
        assert report["status_summary"]["issue_buckets"] == {}

    def test_leaderboard_multi_packs(self, tmp_path):
        """With two trained packs, leaderboards sort correctly."""
        registry = {
            "version": "0.1",
            "base_model": "m",
            "framework": "f",
            "adapter_type": "a",
            "packs": {
                "low": {
                    "name": "Low Rep",
                    "status": "trained",
                    "category": "a",
                    "eval_repetition": 0.10,
                    "eval_vocab_richness": 0.40,
                },
                "high": {
                    "name": "High Vocab",
                    "status": "trained",
                    "category": "b",
                    "eval_repetition": 0.30,
                    "eval_vocab_richness": 0.70,
                },
            },
        }
        reg_path = tmp_path / "registry.yaml"
        reg_path.write_text(yaml.dump(registry))
        # Create minimal dirs so no missing-asset noise
        for name in ("low", "high"):
            d = tmp_path / name
            (d / "adapters").mkdir(parents=True)
            (d / "adapters" / "adapters.safetensors").write_bytes(b"\x00")
            (d / "eval").mkdir()
            (d / "eval" / "metrics.json").write_text("{}")
            (d / "samples").mkdir()
            (d / "samples" / "samples.md").write_text("")
            (d / "voice-pack.yaml").write_text(yaml.dump({
                "display_name": registry["packs"][name]["name"],
                "status": "trained",
                "category": registry["packs"][name]["category"],
            }))

        report = generate_report(registry_path=str(reg_path), repo_root=str(tmp_path))
        lb = report["status_summary"]["leaderboards"]
        # vocab richness: high first
        assert lb["vocab_richness_top"][0]["id"] == "high"
        assert lb["vocab_richness_top"][1]["id"] == "low"
        # repetition: low first
        assert lb["repetition_lowest"][0]["id"] == "low"
        assert lb["repetition_lowest"][1]["id"] == "high"


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


# ---------------------------------------------------------------------------
# Integration tests — format_markdown_summary
# ---------------------------------------------------------------------------


class TestFormatMarkdownSummary:
    def test_contains_header_and_counts(self, tmp_repo):
        repo_root, reg_path = tmp_repo
        report = generate_report(registry_path=reg_path, repo_root=str(repo_root))
        md = format_markdown_summary(report)
        assert "# Voice Pack Audit Summary" in md
        assert "**1** trained" in md
        assert "**1** planned" in md
        assert "**0** issues" in md

    def test_leaderboard_table(self, tmp_repo):
        repo_root, reg_path = tmp_repo
        report = generate_report(registry_path=reg_path, repo_root=str(repo_root))
        md = format_markdown_summary(report)
        assert "## Vocab Richness (top)" in md
        assert "`alpha`" in md

    def test_category_sections(self, tmp_repo):
        repo_root, reg_path = tmp_repo
        report = generate_report(registry_path=reg_path, repo_root=str(repo_root))
        md = format_markdown_summary(report)
        assert "## Trained by Category" in md
        assert "**philosophy**" in md
        assert "## Planned by Category" in md
        assert "**fiction**" in md

    def test_issue_buckets_shown(self, tmp_repo_mismatch):
        repo_root, reg_path = tmp_repo_mismatch
        report = generate_report(registry_path=reg_path, repo_root=str(repo_root))
        md = format_markdown_summary(report)
        assert "## Issue Buckets" in md
        assert "metadata_mismatch" in md


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

    def test_audit_markdown(self, tmp_repo, capsys):
        from voice_packs.cli import main
        import sys

        repo_root, reg_path = tmp_repo
        sys.argv = ["voice-packs", "audit", "--markdown", "--registry", reg_path, "--repo-root", str(repo_root)]
        main()
        captured = capsys.readouterr()
        assert "# Voice Pack Audit Summary" in captured.out
        assert "Vocab Richness" in captured.out

    def test_audit_json_includes_status_summary(self, tmp_repo, capsys):
        from voice_packs.cli import main
        import sys

        repo_root, reg_path = tmp_repo
        sys.argv = ["voice-packs", "audit", "--json", "--registry", reg_path, "--repo-root", str(repo_root)]
        main()
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "status_summary" in data
        assert "leaderboards" in data["status_summary"]

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

    def test_audit_status_filter(self, tmp_repo, capsys):
        from voice_packs.cli import main
        import sys

        repo_root, reg_path = tmp_repo
        sys.argv = [
            "voice-packs", "audit", "--json", "--status", "trained",
            "--registry", reg_path, "--repo-root", str(repo_root),
        ]
        main()
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["summary"]["total_packs"] == 1
        assert [pack["id"] for pack in data["packs"]] == ["alpha"]

    def test_audit_issues_only_filter(self, tmp_repo_mismatch, capsys):
        from voice_packs.cli import main
        import sys

        repo_root, reg_path = tmp_repo_mismatch
        sys.argv = [
            "voice-packs", "audit", "--json", "--issues-only",
            "--registry", reg_path, "--repo-root", str(repo_root),
        ]
        main()
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["summary"]["total_packs"] == 1
        assert [pack["id"] for pack in data["packs"]] == ["gamma"]
