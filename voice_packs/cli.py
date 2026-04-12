"""voice-packs CLI — train, generate, blend, and serve voice pack adapters."""

import argparse
import os
import sys

from voice_packs.paths import validate_path, validate_output_path


def cmd_train(args):
    """Train a voice pack from a corpus directory."""
    from voice_packs.prepare import prepare_corpus
    from voice_packs.train import train

    output_path = validate_output_path(args.output)
    data_dir = os.path.join(output_path, "data")

    if args.corpus.startswith("http://") or args.corpus.startswith("https://"):
        # URL mode: download from one or more URLs
        from voice_packs.prepare import ingest_urls, clean_text, chunk_text, write_splits
        print(f"=== Downloading corpus from URL ===\n")
        urls = [u.strip() for u in args.corpus.split(",")]
        raw = ingest_urls(urls)
        cleaned = clean_text(raw)
        chunks = chunk_text(cleaned, args.chunk_size)
        counts = write_splits(chunks, data_dir)
        stats = {"word_count": len(cleaned.split()), "chunk_count": len(chunks), "splits": counts}
        print(f"Corpus: {stats['word_count']:,} words, {stats['chunk_count']} chunks")
    else:
        corpus_path = validate_path(args.corpus, must_exist=True)
        print(f"=== Preparing corpus from {corpus_path} ===\n")
        stats = prepare_corpus(corpus_path, data_dir, chunk_size=args.chunk_size)

    # Train
    adapter_dir = os.path.join(output_path, "adapters")
    print(f"\n=== Training voice pack '{args.name}' ===\n")
    success = train(
        data_dir=data_dir,
        adapter_path=adapter_dir,
        model=args.model,
        iters=args.iters,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )

    if success:
        print(f"\n✅ Voice pack '{args.name}' trained successfully!")
        print(f"   Adapter: {adapter_dir}")
        print(f"   Corpus: {stats['word_count']:,} words, {stats['chunk_count']} chunks")
        print(f"\n   Generate: voice-packs generate --pack {adapter_dir} --prompt 'Your text here'")
        print(f"   Serve:    voice-packs serve --port 8080")
    else:
        print(f"\n❌ Training failed.")
        sys.exit(1)


def cmd_generate(args):
    """Generate text with a voice pack."""
    from voice_packs.generate import generate

    pack_path = validate_path(args.pack, must_exist=True)
    text = generate(
        adapter_path=pack_path,
        prompt=args.prompt,
        model=args.model,
        max_tokens=args.max_tokens,
        temp=args.temp,
    )
    print(text)


def cmd_blend(args):
    """Blend two voice packs and generate."""
    from voice_packs.blend import blend
    from voice_packs.generate import generate

    pack_a = validate_path(args.pack_a, must_exist=True)
    pack_b = validate_path(args.pack_b, must_exist=True)
    blend_dir = validate_output_path(args.output) if args.output else os.path.join(os.getcwd(), "blended-adapter")
    print(f"Blending {pack_a} ({args.ratio:.0%}) + {pack_b} ({1-args.ratio:.0%})...")
    blend(pack_a, pack_b, blend_dir, args.ratio)
    print(f"Blended adapter saved to {blend_dir}")

    if args.prompt:
        print(f"\nGenerating from blend:\n")
        text = generate(
            adapter_path=blend_dir,
            prompt=args.prompt,
            model=args.model,
            max_tokens=args.max_tokens,
            temp=args.temp,
        )
        print(text)


def cmd_serve(args):
    """Start the voice pack API server."""
    from voice_packs.serve import run_server
    print(f"Starting voice pack server on {args.host}:{args.port}...")
    run_server(host=args.host, port=args.port)


def cmd_audit(args):
    """Audit voice packs: check assets, metadata, and registry consistency."""
    import json as _json

    from voice_packs.audit import generate_report, print_human_report, format_markdown_summary

    report = generate_report(
        registry_path=args.registry,
        repo_root=args.repo_root,
        statuses=args.status,
        issues_only=args.issues_only,
    )

    if args.json:
        print(_json.dumps(report, indent=2))
    elif args.markdown:
        print(format_markdown_summary(report))
    else:
        print_human_report(report)

    if args.strict:
        summary = report["summary"]
        if summary["total_issues"] > 0 or summary["total_mismatches"] > 0:
            raise SystemExit(1)


def cmd_list(args):
    """List available pre-trained voice packs."""
    print("Pre-trained voice packs (download from HuggingFace):\n")
    print("  huggingface.co/jcrowan3/voice-pack-adapters\n")
    packs = [
        ("aquinas", "St. Thomas Aquinas", "Systematic, scholastic"),
        ("augustine", "St. Augustine", "Introspective, rhetorical"),
        ("chesterton", "G.K. Chesterton", "Witty, paradoxical"),
        ("newman", "John Henry Newman", "Careful, apologetic"),
        ("plato", "Plato", "Dialogic, Socratic"),
        ("marcus-aurelius", "Marcus Aurelius", "Stoic, meditative"),
        ("seneca", "Seneca", "Practical, epistolary"),
        ("hemingway", "Ernest Hemingway", "Sparse, declarative"),
        ("austen", "Jane Austen", "Regency social"),
        ("tolkien-adjacent", "Dunsany/Morris", "Archaic fantasy"),
        ("lincoln", "Abraham Lincoln", "Eloquent, principled"),
        ("shakespeare", "William Shakespeare", "Poetic, dramatic"),
        ("dickens", "Charles Dickens", "Vivid, satirical"),
    ]
    print(f"  {'Pack':<20} {'Author':<25} {'Style'}")
    print(f"  {'-'*20} {'-'*25} {'-'*30}")
    for name, author, style in packs:
        print(f"  {name:<20} {author:<25} {style}")
    print(f"\n  Usage: voice-packs generate --pack jcrowan3/voice-pack-adapters/aquinas/360m --prompt '...'")


def main():
    parser = argparse.ArgumentParser(
        prog="voice-packs",
        description="PersonaNexus Voice Packs — weight-level personality for language models",
    )
    sub = parser.add_subparsers(dest="command")

    # train
    p_train = sub.add_parser("train", help="Train a voice pack from a corpus")
    p_train.add_argument("--corpus", required=True, help="Directory containing text files")
    p_train.add_argument("--name", required=True, help="Name for the voice pack")
    p_train.add_argument("--output", default="./voice-pack-output", help="Output directory")
    p_train.add_argument("--model", default="HuggingFaceTB/SmolLM2-360M", help="Base model")
    p_train.add_argument("--iters", type=int, default=1000, help="Training iterations")
    p_train.add_argument("--batch-size", type=int, default=4, help="Batch size")
    p_train.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    p_train.add_argument("--chunk-size", type=int, default=400, help="Words per training chunk")

    # generate
    p_gen = sub.add_parser("generate", help="Generate text with a voice pack")
    p_gen.add_argument("--pack", required=True, help="Path to adapter directory or HF repo")
    p_gen.add_argument("--prompt", required=True, help="Text prompt")
    p_gen.add_argument("--model", default="HuggingFaceTB/SmolLM2-360M", help="Base model")
    p_gen.add_argument("--max-tokens", type=int, default=200, help="Max tokens to generate")
    p_gen.add_argument("--temp", type=float, default=0.7, help="Sampling temperature")

    # blend
    p_blend = sub.add_parser("blend", help="Blend two voice packs")
    p_blend.add_argument("--pack-a", required=True, help="First adapter path")
    p_blend.add_argument("--pack-b", required=True, help="Second adapter path")
    p_blend.add_argument("--ratio", type=float, default=0.5, help="Blend ratio (1.0=100%% A)")
    p_blend.add_argument("--output", help="Output path for blended adapter")
    p_blend.add_argument("--prompt", help="Optional: generate from the blend")
    p_blend.add_argument("--model", default="HuggingFaceTB/SmolLM2-360M", help="Base model")
    p_blend.add_argument("--max-tokens", type=int, default=200)
    p_blend.add_argument("--temp", type=float, default=0.7)

    # serve
    p_serve = sub.add_parser("serve", help="Start the voice pack API server")
    p_serve.add_argument("--host", default="0.0.0.0", help="Server host")
    p_serve.add_argument("--port", type=int, default=8080, help="Server port")

    # audit
    _default_registry = os.path.join(os.path.dirname(os.path.dirname(__file__)), "registry.yaml")
    _default_repo_root = os.path.dirname(os.path.dirname(__file__))
    p_audit = sub.add_parser("audit", help="Audit voice packs (assets, metadata, registry)")
    p_audit.add_argument("--json", action="store_true", help="Emit report as JSON")
    p_audit.add_argument("--markdown", action="store_true", help="Emit concise Markdown summary (leaderboards, issue buckets)")
    p_audit.add_argument("--strict", action="store_true", help="Exit non-zero on issues or mismatches")
    p_audit.add_argument("--status", action="append", help="Filter to one or more exact pack statuses (repeatable)")
    p_audit.add_argument("--issues-only", action="store_true", help="Show only packs with issues")
    p_audit.add_argument("--registry", default=_default_registry, help="Path to registry.yaml")
    p_audit.add_argument("--repo-root", default=_default_repo_root, help="Voice-packs repo root")

    # list
    sub.add_parser("list", help="List pre-trained voice packs")

    args = parser.parse_args()

    if args.command == "train":
        cmd_train(args)
    elif args.command == "generate":
        cmd_generate(args)
    elif args.command == "blend":
        cmd_blend(args)
    elif args.command == "serve":
        cmd_serve(args)
    elif args.command == "audit":
        cmd_audit(args)
    elif args.command == "list":
        cmd_list(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
