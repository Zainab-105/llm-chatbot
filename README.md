# LLM Chatbot Comparison

A simple chatbot app to compare two small language models side by side:

- **LFM2-350M** by LiquidAI — hybrid state-space + transformer architecture
- **SmolLM2-360M** by HuggingFace — standard transformer, instruction-tuned

## Features

- Chat with either model individually
- Compare both models' responses side by side
- Performance stats (tokens/sec, response time)
- Runs locally on CPU (no GPU required)

## Setup

```bash
# Clone the repo
git clone https://github.com/yourusername/llm-chatbot.git
cd llm-chatbot

# Install dependencies
pip install -r requirements.txt

# Run the app
python chatbot.py
```

Then open **http://127.0.0.1:7860** in your browser.

## Requirements

- Python 3.10+
- ~2GB RAM (models are ~350M parameters each)
- Internet connection (first run only, to download models)

## Tech Stack

- **PyTorch** — model inference
- **Transformers** — model loading from HuggingFace
- **Gradio** — web UI
