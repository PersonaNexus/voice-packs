"""Corpus preparation — ingest, clean, chunk, and split text for LoRA training."""

import json
import os
import random
import re
from pathlib import Path


def ingest_directory(corpus_path: str) -> str:
    """Read all text files from a directory and concatenate."""
    corpus_path = Path(corpus_path)
    texts = []

    for ext in ["*.txt", "*.md", "*.rst", "*.html"]:
        for f in sorted(corpus_path.rglob(ext)):
            content = f.read_text(errors="replace")
            if ext == "*.html":
                content = _strip_html(content)
            if len(content.strip()) > 100:
                texts.append(content.strip())

    if not texts:
        raise ValueError(f"No text files found in {corpus_path}")

    return "\n\n".join(texts)


def _strip_html(html: str) -> str:
    """Basic HTML stripping without external dependencies."""
    text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)
    text = re.sub(r"  +", " ", text)
    return text


def clean_text(text: str) -> str:
    """Clean and normalize text for training."""
    # Collapse single-line artifacts (from HTML bold/italic)
    text = text.replace("\n\n", "<<PARA>>")
    text = text.replace("\n", " ")
    text = text.replace("<<PARA>>", "\n\n")

    # Fix spacing
    text = re.sub(r"  +", " ", text)
    text = re.sub(r" ([.,;:?!])", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def chunk_text(text: str, chunk_size: int = 400) -> list[str]:
    """Split text into chunks of approximately chunk_size words."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 20]

    chunks = []
    current = []
    words = 0

    for s in sentences:
        w = len(s.split())
        if words + w > chunk_size and current:
            chunks.append(" ".join(current))
            current = [s]
            words = w
        else:
            current.append(s)
            words += w

    if current:
        chunks.append(" ".join(current))

    return chunks


def write_splits(
    chunks: list[str],
    output_dir: str,
    train_ratio: float = 0.95,
    seed: int = 42,
) -> dict[str, int]:
    """Write train/valid/test JSONL splits."""
    os.makedirs(output_dir, exist_ok=True)
    random.seed(seed)
    random.shuffle(chunks)

    split_idx = max(int(len(chunks) * train_ratio), len(chunks) - max(5, len(chunks) // 20))

    splits = {
        "train": chunks[:split_idx],
        "valid": chunks[split_idx:],
        "test": chunks[split_idx:][:20],
    }

    counts = {}
    for name, data in splits.items():
        path = os.path.join(output_dir, f"{name}.jsonl")
        with open(path, "w") as f:
            for chunk in data:
                f.write(json.dumps({"text": chunk}) + "\n")
        counts[name] = len(data)

    return counts


def prepare_corpus(
    corpus_path: str,
    output_dir: str,
    chunk_size: int = 400,
) -> dict:
    """Full pipeline: ingest → clean → chunk → split."""
    print(f"Ingesting from {corpus_path}...")
    raw = ingest_directory(corpus_path)

    print("Cleaning...")
    cleaned = clean_text(raw)

    word_count = len(cleaned.split())
    print(f"Corpus: {word_count:,} words")

    if word_count < 50_000:
        print(f"WARNING: {word_count:,} words is below the 100K minimum for good results.")
    elif word_count < 100_000:
        print(f"Note: {word_count:,} words is marginal. 100K+ recommended for production quality.")

    print("Chunking...")
    chunks = chunk_text(cleaned, chunk_size)
    print(f"Created {len(chunks)} chunks")

    print("Writing splits...")
    counts = write_splits(chunks, output_dir)
    print(f"  train: {counts['train']}, valid: {counts['valid']}, test: {counts['test']}")

    return {
        "word_count": word_count,
        "chunk_count": len(chunks),
        "splits": counts,
    }
