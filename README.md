# AI Item Tracker

An AI-powered office item tracking assistant built with small language models. Ask naturally where things are, or search the inventory directly.

## Features

- **Ask AI** — Chat naturally: *"Where is the projector?"*, *"What's in Room 101?"*
- **Quick Search** — Instant keyword lookup (no AI, instant results)
- **Inventory View** — Browse and filter the full inventory table
- **Compare Models** — See how two different LLMs answer the same question

## Models

| Model | Provider | Type |
|-------|----------|------|
| LFM2-350M | LiquidAI | Hybrid state-space + transformer |
| SmolLM2-360M | HuggingFace | Transformer (instruction-tuned) |

## Setup

```bash
# Clone the repo
git clone https://github.com/zainab-hafeez/ai-item-tracker.git
cd ai-item-tracker

# Install dependencies
pip install -r requirements.txt

# Run the app
python chatbot.py
```

Open **http://127.0.0.1:7860** in your browser (a public share link is also printed).

## Requirements

- Python 3.10+
- ~2GB RAM (models are ~350M parameters each)
- Internet connection (first run only, to download models)

## Tech Stack

- **PyTorch** — model inference
- **Transformers** — model loading from HuggingFace Hub
- **Gradio** — web UI
