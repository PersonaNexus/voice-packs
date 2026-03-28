#!/usr/bin/env python3
"""Download, prepare, and train humor voice packs: Twain, Wilde, Wodehouse."""

import os
from voice_packs.prepare import ingest_urls, clean_text, chunk_text, write_splits
from voice_packs.train import train

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_BASE = os.path.expanduser("~/personal-ai-org/catholic-finetune/data")

HUMOR_PACKS = {
    "twain": {
        "display_name": "Mark Twain",
        "style": "Dry wit, frontier humor, social satire",
        "urls": [
            "https://www.gutenberg.org/cache/epub/76/pg76.txt",      # Adventures of Tom Sawyer
            "https://www.gutenberg.org/cache/epub/74/pg74.txt",      # Adventures of Huckleberry Finn
            "https://www.gutenberg.org/cache/epub/86/pg86.txt",      # A Connecticut Yankee
            "https://www.gutenberg.org/cache/epub/1044/pg1044.txt",  # The Prince and the Pauper
            "https://www.gutenberg.org/cache/epub/3176/pg3176.txt",  # Following the Equator
            "https://www.gutenberg.org/cache/epub/3193/pg3193.txt",  # Roughing It
            "https://www.gutenberg.org/cache/epub/245/pg245.txt",    # The Mysterious Stranger
            "https://www.gutenberg.org/cache/epub/119/pg119.txt",    # A Tramp Abroad
            "https://www.gutenberg.org/cache/epub/3175/pg3175.txt",  # Innocents Abroad
        ],
    },
    "wilde": {
        "display_name": "Oscar Wilde",
        "style": "Razor-sharp epigrams, witty social commentary",
        "urls": [
            "https://www.gutenberg.org/cache/epub/174/pg174.txt",    # Picture of Dorian Gray
            "https://www.gutenberg.org/cache/epub/844/pg844.txt",    # The Importance of Being Earnest
            "https://www.gutenberg.org/cache/epub/854/pg854.txt",    # An Ideal Husband
            "https://www.gutenberg.org/cache/epub/790/pg790.txt",    # Lady Windermere's Fan
            "https://www.gutenberg.org/cache/epub/921/pg921.txt",    # A Woman of No Importance
            "https://www.gutenberg.org/cache/epub/14522/pg14522.txt", # De Profundis
            "https://www.gutenberg.org/cache/epub/1338/pg1338.txt",  # Essays and Lectures
            "https://www.gutenberg.org/cache/epub/885/pg885.txt",    # An Ideal Husband (alt)
            "https://www.gutenberg.org/cache/epub/773/pg773.txt",    # Lord Arthur Savile's Crime
        ],
    },
    "wodehouse": {
        "display_name": "P.G. Wodehouse",
        "style": "Absurdist British comedy, elaborate metaphors",
        "urls": [
            "https://www.gutenberg.org/cache/epub/8164/pg8164.txt",  # Right Ho, Jeeves
            "https://www.gutenberg.org/cache/epub/10554/pg10554.txt", # My Man Jeeves
            "https://www.gutenberg.org/cache/epub/20717/pg20717.txt", # The Man with Two Left Feet
            "https://www.gutenberg.org/cache/epub/6877/pg6877.txt",  # Love Among the Chickens
            "https://www.gutenberg.org/cache/epub/6837/pg6837.txt",  # Mike
            "https://www.gutenberg.org/cache/epub/7471/pg7471.txt",  # Psmith in the City
            "https://www.gutenberg.org/cache/epub/3756/pg3756.txt",  # A Damsel in Distress
            "https://www.gutenberg.org/cache/epub/20533/pg20533.txt", # The Adventures of Sally
            "https://www.gutenberg.org/cache/epub/2005/pg2005.txt",  # Something New
        ],
    },
}


def prepare_and_train(name, config):
    print(f"\n{'='*60}")
    print(f"Preparing: {config['display_name']}")
    print(f"{'='*60}")

    # Download and prepare corpus
    raw = ingest_urls(config["urls"])
    cleaned = clean_text(raw)
    word_count = len(cleaned.split())
    print(f"Corpus: {word_count:,} words")

    chunks = chunk_text(cleaned, chunk_size=400)
    print(f"Created {len(chunks)} chunks")

    data_dir = os.path.join(DATA_BASE, f"personality-{name}")
    counts = write_splits(chunks, data_dir)
    print(f"Splits: train={counts['train']}, valid={counts['valid']}, test={counts['test']}")

    # Train
    adapter_path = os.path.join(REPO_ROOT, name, "adapters")
    success = train(data_dir, adapter_path)

    if success:
        print(f"\n{config['display_name']} voice pack trained successfully!")
    else:
        print(f"\nWARNING: {config['display_name']} training failed!")

    return word_count, success


if __name__ == "__main__":
    results = {}
    for name, config in HUMOR_PACKS.items():
        word_count, success = prepare_and_train(name, config)
        results[name] = {"words": word_count, "success": success}

    print(f"\n{'='*60}")
    print("Summary:")
    for name, r in results.items():
        status = "OK" if r["success"] else "FAILED"
        print(f"  {name}: {r['words']:,} words — {status}")
    print(f"{'='*60}")
