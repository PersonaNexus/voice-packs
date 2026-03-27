#!/usr/bin/env python3
"""
Download and prepare corpus data for secular voice packs: Lincoln, Shakespeare, Dickens.

Downloads plain text from Project Gutenberg, cleans it, and produces
train/valid/test JSONL files compatible with mlx-lm fine-tuning.

Usage:
    python prepare_secular.py              # prepare all three
    python prepare_secular.py lincoln      # prepare just one
"""

import json
import os
import random
import re
import sys
import urllib.request

OUT_BASE = os.path.expanduser("~/personal-ai-org/catholic-finetune/data")

# Gutenberg plain-text URLs (UTF-8 versions)
SOURCES = {
    "lincoln": {
        "display_name": "Abraham Lincoln",
        "style": "Eloquent, principled, frontier plainspoken",
        "category": "political-rhetoric",
        "urls": [
            # Lincoln's speeches and writings collections
            ("https://www.gutenberg.org/cache/epub/2653/pg2653.txt", "Lincoln's Inaugurals and Letters"),
            ("https://www.gutenberg.org/cache/epub/3253/pg3253.txt", "Lincoln's Gettysburg Address etc"),
            ("https://www.gutenberg.org/cache/epub/47136/pg47136.txt", "Writings of Abraham Lincoln Vol 1"),
            ("https://www.gutenberg.org/cache/epub/47137/pg47137.txt", "Writings of Abraham Lincoln Vol 2"),
            ("https://www.gutenberg.org/cache/epub/47138/pg47138.txt", "Writings of Abraham Lincoln Vol 3"),
            ("https://www.gutenberg.org/cache/epub/47139/pg47139.txt", "Writings of Abraham Lincoln Vol 4"),
            ("https://www.gutenberg.org/cache/epub/47140/pg47140.txt", "Writings of Abraham Lincoln Vol 5"),
            ("https://www.gutenberg.org/cache/epub/47141/pg47141.txt", "Writings of Abraham Lincoln Vol 6"),
            ("https://www.gutenberg.org/cache/epub/47142/pg47142.txt", "Writings of Abraham Lincoln Vol 7"),
        ],
    },
    "shakespeare": {
        "display_name": "William Shakespeare",
        "style": "Poetic, dramatic, iambic, richly metaphorical",
        "category": "literary-drama",
        "urls": [
            ("https://www.gutenberg.org/cache/epub/100/pg100.txt", "Complete Works of Shakespeare"),
        ],
    },
    "dickens": {
        "display_name": "Charles Dickens",
        "style": "Vivid, social, satirical, ornate prose",
        "category": "literary-fiction",
        "urls": [
            ("https://www.gutenberg.org/cache/epub/98/pg98.txt", "A Tale of Two Cities"),
            ("https://www.gutenberg.org/cache/epub/1400/pg1400.txt", "Great Expectations"),
            ("https://www.gutenberg.org/cache/epub/766/pg766.txt", "David Copperfield"),
            ("https://www.gutenberg.org/cache/epub/580/pg580.txt", "The Pickwick Papers"),
            ("https://www.gutenberg.org/cache/epub/730/pg730.txt", "Oliver Twist"),
            ("https://www.gutenberg.org/cache/epub/46/pg46.txt", "A Christmas Carol"),
            ("https://www.gutenberg.org/cache/epub/1023/pg1023.txt", "Bleak House"),
        ],
    },
}


def download_text(url, label):
    """Download a Gutenberg text file and return the cleaned body."""
    print(f"  Downloading: {label}...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        print(f"    FAILED: {e}")
        return ""

    # Strip Gutenberg header/footer
    body = strip_gutenberg_boilerplate(raw)
    return body


def strip_gutenberg_boilerplate(text):
    """Remove Project Gutenberg header and footer."""
    # Find start of actual text
    start_markers = [
        "*** START OF THIS PROJECT GUTENBERG",
        "*** START OF THE PROJECT GUTENBERG",
        "***START OF THIS PROJECT GUTENBERG",
        "***START OF THE PROJECT GUTENBERG",
    ]
    end_markers = [
        "*** END OF THIS PROJECT GUTENBERG",
        "*** END OF THE PROJECT GUTENBERG",
        "***END OF THIS PROJECT GUTENBERG",
        "***END OF THE PROJECT GUTENBERG",
        "End of the Project Gutenberg",
        "End of Project Gutenberg",
    ]

    start_idx = 0
    for marker in start_markers:
        idx = text.find(marker)
        if idx != -1:
            # Move past the marker line
            nl = text.find("\n", idx)
            start_idx = nl + 1 if nl != -1 else idx + len(marker)
            break

    end_idx = len(text)
    for marker in end_markers:
        idx = text.find(marker)
        if idx != -1 and idx > start_idx:
            end_idx = idx
            break

    return text[start_idx:end_idx].strip()


def clean_text(text):
    """Basic text cleanup."""
    # Normalize whitespace
    text = re.sub(r"\r\n", "\n", text)
    # Remove excessive blank lines (keep max 2)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    # Remove lines that are just underscores or dashes (decorative)
    text = re.sub(r"\n[_\-=]{5,}\n", "\n\n", text)
    return text.strip()


def chunk_text(text, chunk_size=400):
    """Split text into chunks of ~chunk_size words."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip() and len(p.strip()) > 30]
    chunks = []
    current = []
    current_words = 0

    for para in paragraphs:
        words = len(para.split())
        if current_words + words > chunk_size and current:
            chunks.append("\n\n".join(current))
            current = [para]
            current_words = words
        else:
            current.append(para)
            current_words += words

    if current:
        chunks.append("\n\n".join(current))

    # Filter out very short chunks
    chunks = [c for c in chunks if len(c.split()) >= 50]
    return chunks


def write_jsonl(chunks, outdir, name):
    """Write train/valid/test JSONL splits."""
    os.makedirs(outdir, exist_ok=True)
    random.seed(42)
    random.shuffle(chunks)

    split_idx = int(len(chunks) * 0.95)
    train = chunks[:split_idx]
    val = chunks[split_idx:]

    for split_name, data in [("train", train), ("valid", val), ("test", val[:20])]:
        path = os.path.join(outdir, f"{split_name}.jsonl")
        with open(path, "w") as f:
            for chunk in data:
                f.write(json.dumps({"text": chunk}) + "\n")

    total_words = sum(len(c.split()) for c in chunks)
    print(f"  {name}: {len(chunks)} chunks ({total_words:,} words) -> {len(train)} train / {len(val)} val")
    return total_words


def prepare_author(name):
    """Download and prepare data for one author."""
    source = SOURCES[name]
    print(f"\n{'='*60}")
    print(f"Preparing: {source['display_name']}")
    print(f"{'='*60}")

    all_text = []
    for url, label in source["urls"]:
        text = download_text(url, label)
        if text:
            text = clean_text(text)
            all_text.append(text)
            print(f"    -> {len(text.split()):,} words")

    if not all_text:
        print(f"  ERROR: No text downloaded for {name}")
        return 0

    combined = "\n\n\n".join(all_text)
    total_words = len(combined.split())
    print(f"  Total raw corpus: {total_words:,} words")

    chunks = chunk_text(combined, chunk_size=400)
    outdir = os.path.join(OUT_BASE, f"personality-{name}")
    corpus_words = write_jsonl(chunks, outdir, name)
    return corpus_words


def main():
    targets = sys.argv[1:] if len(sys.argv) > 1 else list(SOURCES.keys())

    for name in targets:
        if name not in SOURCES:
            print(f"Unknown author: {name}. Available: {', '.join(SOURCES.keys())}")
            sys.exit(1)

    results = {}
    for name in targets:
        words = prepare_author(name)
        results[name] = words

    print(f"\n{'='*60}")
    print("Summary:")
    for name, words in results.items():
        print(f"  {name}: {words:,} words")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
