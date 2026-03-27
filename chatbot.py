import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

MODELS = {
    "LFM2-350M": "LiquidAI/LFM2-350M",
    "SmolLM2-360M": "HuggingFaceTB/SmolLM2-360M-Instruct",
}

loaded_models = {}
device = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(name):
    if name in loaded_models:
        return loaded_models[name]
    repo = MODELS[name]
    print(f"Loading {name}...")
    tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
    dt = torch.float32 if device == "cpu" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        repo, dtype=dt, trust_remote_code=True,
        low_cpu_mem_usage=False,
    ).to(device).eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    loaded_models[name] = (model, tokenizer)
    return model, tokenizer


def generate(name, prompt, max_tokens=200, temp=0.7):
    model, tokenizer = load_model(name)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_tokens, temperature=temp,
            do_sample=True, top_p=0.9, pad_token_id=tokenizer.pad_token_id,
        )
    elapsed = time.time() - t0
    new_tok = out[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(new_tok, skip_special_tokens=True).strip()
    tps = len(new_tok) / elapsed if elapsed > 0 else 0
    return text, f"{len(new_tok)} tokens | {elapsed:.1f}s | {tps:.1f} tok/s"


def respond(message, history, model_name):
    if not message.strip():
        return history, "", ""
    # Build prompt
    prompt = ""
    for msg in history:
        role = msg.get("role", "")
        content = msg.get("content", "")
        prompt += f"{role}: {content}\n"
    prompt += f"user: {message}\nassistant:"

    try:
        text, stats = generate(model_name, prompt)
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": text})
        return history, "", stats
    except Exception as e:
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": f"Error: {e}"})
        return history, "", f"Error: {e}"


def compare(message):
    if not message.strip():
        return "", "", ""
    out1, stats1 = generate("LFM2-350M", message)
    out2, stats2 = generate("SmolLM2-360M", message)
    return f"**LFM2-350M**\n{stats1}\n\n{out1}", f"**SmolLM2-360M**\n{stats2}\n\n{out2}", ""


CSS = """
.main-wrap { max-width: 800px; margin: 0 auto; }
.header { text-align: center; padding: 20px 0 10px; }
.header h1 { font-size: 1.6em; margin: 0; }
.header p { color: #666; margin: 4px 0 0; font-size: 0.95em; }
.stats-bar { font-size: 0.8em; color: #888; text-align: right; min-height: 20px; }
"""

with gr.Blocks(css=CSS, title="LLM Chat") as app:

    with gr.Column(elem_classes="main-wrap"):
        # Header
        gr.HTML("""
            <div class="header">
                <h1>LLM Chat</h1>
                <p>Compare small language models side by side</p>
            </div>
        """)

        with gr.Tabs():
            # ---- Chat Tab ----
            with gr.Tab("Chat"):
                model_pick = gr.Radio(
                    choices=list(MODELS.keys()),
                    value="LFM2-350M",
                    label="Model",
                    interactive=True,
                )
                chat = gr.Chatbot(height=400, show_label=False, placeholder="Click Send to start chatting...")
                stats_display = gr.Markdown(elem_classes="stats-bar")
                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="Type a message...",
                        show_label=False,
                        scale=6,
                        container=False,
                    )
                    send = gr.Button("Send", variant="primary", scale=1, min_width=80)
                clear = gr.Button("Clear", size="sm")

                send.click(respond, [msg, chat, model_pick], [chat, msg, stats_display])
                msg.submit(respond, [msg, chat, model_pick], [chat, msg, stats_display])
                clear.click(lambda: ([], "", ""), outputs=[chat, msg, stats_display])

            # ---- Compare Tab ----
            with gr.Tab("Compare"):
                gr.Markdown("Send the same question to both models.")
                cmp_input = gr.Textbox(placeholder="Type a question...", show_label=False, container=False)
                cmp_btn = gr.Button("Compare", variant="primary")
                with gr.Row(equal_height=True):
                    col1 = gr.Markdown("", label="LFM2-350M")
                    col2 = gr.Markdown("", label="SmolLM2-360M")

                cmp_btn.click(compare, [cmp_input], [col1, col2, cmp_input])
                cmp_input.submit(compare, [cmp_input], [col1, col2, cmp_input])


if __name__ == "__main__":
    print(f"Device: {device}")
    print("Starting server...")
    app.launch(share=True)
