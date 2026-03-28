# PersonaNexus Voice Packs

Weight-level personality modules for language models. Each voice pack is a LoRA adapter trained on a specific author's corpus, producing text in that author's distinctive style.

Unlike system prompts, voice packs modify the model's weights — producing deeper personality transfer that resists drift over long conversations.

**Part of the PersonaNexus ecosystem:**
- [PersonaNexus](https://github.com/PersonaNexus/personanexus) — declarative agent identity framework
- [AgentSkillFactory](https://github.com/PersonaNexus/AgentSkillFactory) — auto-generate agent identities from job descriptions
- **Voice Packs** (this repo) — weight-level personality adapters
- [Adapter Weights on HuggingFace](https://huggingface.co/jcrowan3/voice-pack-adapters) — pre-trained adapters for 13 voices

## How It Works

1. A **base model** (e.g., SmolLM2-360M) provides general language ability
2. A **voice pack** (LoRA adapter) is loaded on top, shifting the model's style
3. A **PersonaNexus identity YAML** defines the personality traits and behavioral rules
4. Together: the identity says *what* the agent does, the voice pack says *how* it sounds

## Available Voice Packs

### Philosophy & Theology

| Pack | Author | Style | Corpus | Status |
|------|--------|-------|--------|--------|
| `aquinas` | St. Thomas Aquinas | Systematic, scholastic, Q&A articles | Summa Theologica (547K words) | **Trained** |
| `augustine` | St. Augustine | Introspective, rhetorical, narrative | Confessions, City of God, On the Trinity (554K words) | **Trained** |
| `chesterton` | G.K. Chesterton | Witty, paradoxical, accessible | Orthodoxy, Heretics, Everlasting Man (332K words) | **Trained** |
| `newman` | John Henry Newman | Careful, apologetic, developmental | Apologia, Development of Doctrine (198K words) | **Trained** |
| `john-of-the-cross` | St. John of the Cross | Mystical, poetic, contemplative | Dark Night, Ascent of Mount Carmel (66K words) | Experimental |
| `teresa-avila` | St. Teresa of Avila | Practical mysticism, direct, warm | Interior Castle, Way of Perfection (76K words) | Experimental |
| `francis-de-sales` | St. Francis de Sales | Gentle, pastoral, devotional | Introduction to the Devout Life, Treatise on Love of God (62K words) | Experimental |

### Classical Philosophy

| Pack | Author | Style | Corpus | Status |
|------|--------|-------|--------|--------|
| `plato` | Plato (Jowett trans.) | Dialogic, Socratic questioning | Republic, Symposium, Apology, Phaedo (304K words) | **Trained** |
| `marcus-aurelius` | Marcus Aurelius | Stoic, aphoristic, meditative | Meditations (112K words) | **Trained** |
| `seneca` | Seneca | Practical philosophy, epistolary | Letters to Lucilius, On the Shortness of Life (114K words) | **Trained** |

### Literary Fiction

| Pack | Author | Style | Corpus | Status |
|------|--------|-------|--------|--------|
| `hemingway` | Ernest Hemingway | Sparse, declarative, dialogue-heavy | A Farewell to Arms, The Sun Also Rises (93K words) | **Trained** |
| `austen` | Jane Austen | Regency social, character-driven | Pride and Prejudice, Emma, Sense and Sensibility (404K words) | **Trained** |
| `tolkien-adjacent` | Lord Dunsany / William Morris | Archaic fantasy, epic prose | King of Elfland's Daughter, Well at World's End (262K words) | **Trained** |

### Historical

| Pack | Author | Style | Corpus | Status |
|------|--------|-------|--------|--------|
| `lincoln` | Abraham Lincoln | Eloquent, principled, plainspoken | Collected Writings Vols 1-7 (885K words) | **Trained** |
| `shakespeare` | William Shakespeare | Poetic, dramatic, iambic | Complete Works (935K words) | **Trained** |
| `dickens` | Charles Dickens | Vivid, satirical, ornate | 7 novels (1.5M words) | **Trained** |

**13 trained voice packs** across 4 categories. Adapters hosted on [HuggingFace](https://huggingface.co/jcrowan3/voice-pack-adapters).

### Future Packs

- **C.S. Lewis** — requires public domain sources (copyright varies by country)
- **Dostoevsky** — Constance Garnett translations on Gutenberg
- **Aristotle** — available via Gutenberg translations

## Voice Pack Structure

```
voice-packs/
├── registry.yaml              # Index of all available packs
├── aquinas/
│   ├── voice-pack.yaml        # Metadata + PersonaNexus integration
│   ├── training-config.yaml   # How this pack was trained
│   ├── adapters/              # LoRA adapter weights
│   │   └── adapters.safetensors
│   ├── eval/                  # Evaluation results
│   │   └── metrics.json
│   └── samples/               # Example generations
│       └── samples.md
├── augustine/
│   └── ...
└── chesterton/
    └── ...
```

## Evaluation Results

From 900+ generations with statistical significance (5 runs per condition, two model sizes):

| Voice | Repetition ↓ | Vocab Richness ↑ | Best For |
|-------|-------------|-------------------|----------|
| Base (no pack) | 0.237 | 0.482 | Generic text |
| **Newman** | **0.124** | **0.573** | Measured academic writing |
| Augustine | 0.192 | 0.469 | Reflective, narrative theology |
| Chesterton | 0.238 | 0.473 | Apologetics, accessible writing |
| Aquinas | 0.213 (1.7B) | 0.517 (1.7B) | Systematic theology |

**Key findings:**
- LoRA adapters beat prompt-only baselines in **6/8 comparisons**
- Up to **49% less personality drift** over long generation
- **Adapter blending** creates hybrid personalities better than either source
- Cross-domain validated across theology, classical philosophy, and literary fiction

## Usage with PersonaNexus

In a PersonaNexus identity YAML:

```yaml
metadata:
  name: fr-thomas
  version: "1.0"

personality:
  voice_pack: aquinas          # Load this LoRA adapter
  traits:
    rigor: 0.95
    warmth: 0.3
    formality: 0.9

interaction:
  style: scholastic
  mode: systematic-analysis
```

## Audit and Status Reporting

Use the repo audit to see what is actually ready, what assets are missing, and whether `registry.yaml` still matches each pack's own metadata.

```bash
# Human-readable audit (CLI)
voice-packs audit

# Machine-readable output for tooling / CI
voice-packs audit --json

# Fail fast if mismatches or missing required trained-pack assets are found
voice-packs audit --strict

# Point at a non-default registry or repo root
voice-packs audit --registry /path/to/registry.yaml --repo-root /path/to/repo
```

The legacy `train_pack.py --report` interface still works and delegates to the same logic:

```bash
python train_pack.py --report
python train_pack.py --report --json
python train_pack.py --report --strict
```

The audit checks, for each pack:
- registry status and category
- corpus size and evaluation metrics when available
- presence of `voice-pack.yaml`
- presence of adapter weights, eval artifacts, and sample artifacts
- metadata mismatches between `registry.yaml` and the pack-local YAML

This gives PersonaNexus a lightweight operational health view, not just a marketing list.

## Training Your Own

```bash
# 1. Prepare corpus
python prepare_personalities.py --author "Your Author" --source /path/to/texts

# 2. Train adapter
uv run mlx_lm.lora \
  --model HuggingFaceTB/SmolLM2-360M \
  --train --data ./data/personality-yourauthor \
  --num-layers 12 --iters 1000 --learning-rate 5e-5 \
  --adapter-path ./voice-packs/yourauthor/adapters

# 3. Test
uv run mlx_lm.generate \
  --model HuggingFaceTB/SmolLM2-360M \
  --adapter-path ./voice-packs/yourauthor/adapters \
  --prompt "Your test prompt"
```

## Base Models Supported

| Model | Params | VRAM | Speed | Quality |
|-------|--------|------|-------|---------|
| SmolLM2-135M | 135M | 0.3GB | 374 tok/s | Basic coherence |
| SmolLM2-360M | 360M | 0.8GB | 245 tok/s | Good quality, recommended |
| SmolLM2-1.7B | 1.7B | 3.5GB | ~60 tok/s | Best quality (untested) |

## License

MIT — voice pack weights are derivative of MIT-licensed base models trained on public domain texts.
