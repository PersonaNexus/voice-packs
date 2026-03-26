# PersonaNexus Voice Packs

Weight-level personality modules for language models. Each voice pack is a LoRA adapter trained on a specific author's corpus, producing text in that author's distinctive style.

Unlike system prompts, voice packs modify the model's weights — producing deeper personality transfer that resists drift over long conversations.

## How It Works

1. A **base model** (e.g., SmolLM2-360M) provides general language ability
2. A **voice pack** (LoRA adapter) is loaded on top, shifting the model's style
3. A **PersonaNexus identity YAML** defines the personality traits and behavioral rules
4. Together: the identity says *what* the agent does, the voice pack says *how* it sounds

## Available Voice Packs

### Philosophy & Theology

| Pack | Author | Style | Corpus | Status |
|------|--------|-------|--------|--------|
| `aquinas` | St. Thomas Aquinas | Systematic, scholastic, Q&A articles | Summa Theologica (547K words) | Trained |
| `augustine` | St. Augustine | Introspective, rhetorical, narrative | Confessions, City of God, On the Trinity (554K words) | Trained |
| `chesterton` | G.K. Chesterton | Witty, paradoxical, accessible | Orthodoxy, Heretics, Everlasting Man (332K words) | Trained |
| `newman` | John Henry Newman | Careful, apologetic, developmental | Apologia, Development of Doctrine | Planned |
| `john-of-the-cross` | St. John of the Cross | Mystical, poetic, contemplative | Dark Night, Ascent of Mount Carmel | Planned |
| `teresa-avila` | St. Teresa of Avila | Practical mysticism, direct, warm | Interior Castle, Way of Perfection | Planned |
| `francis-de-sales` | St. Francis de Sales | Gentle, pastoral, devotional | Introduction to the Devout Life | Planned |
| `cs-lewis` | C.S. Lewis | Analogical, clear, imaginative | *(requires public domain sources)* | Planned |

### Planned Categories

- **Classical Philosophy**: Plato, Aristotle, Seneca, Marcus Aurelius
- **Modern Theology**: Ratzinger/Benedict XVI, Hans Urs von Balthasar, Henri de Lubac
- **Spirituality**: Ignatius of Loyola, Brother Lawrence, Thomas Merton
- **Apologetics**: Frank Sheed, Peter Kreeft, Scott Hahn

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

## Evaluation Results (SmolLM2-360M)

From our systematic evaluation (60 generations, 5 categories, 15 prompts):

| Voice | Repetition ↓ | Vocab Richness ↑ | Theological Density | Best For |
|-------|-------------|-------------------|-------------------|----------|
| Base (no pack) | 0.261 | 0.491 | 0.042 | Generic text |
| Augustine | **0.147** | **0.505** | 0.034 | Reflective, narrative theology |
| Chesterton | 0.217 | 0.487 | 0.031 | Apologetics, accessible writing |
| Aquinas | 0.406 | 0.430 | **0.061** | Systematic theology *(needs data cleaning)* |

Key finding: Voice packs reduce personality drift by up to 44% compared to the base model.

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
