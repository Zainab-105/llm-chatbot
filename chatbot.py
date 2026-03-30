import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import re
import json
import os

# ── Inventory Database ──────────────────────────────────────────────
DATA_FILE = os.path.join(os.path.dirname(__file__), "inventory.json")

DEFAULT_INVENTORY = [
    {"item": "Laptop", "location": "Office Desk Drawer"},
    {"item": "Notebook", "location": "Shelf B, Room 102"},
    {"item": "Pen", "location": "Pen Holder on Desk"},
    {"item": "Projector", "location": "Conference Room Cabinet"},
    {"item": "Headphones", "location": "Locker 3, Room 101"},
    {"item": "Smartphone", "location": "Office Desk Surface"},
    {"item": "Charger", "location": "Drawer under Office Desk"},
    {"item": "Whiteboard Marker", "location": "Whiteboard Tray, Room 103"},
    {"item": "Router", "location": "Server Room Shelf 2"},
    {"item": "External Hard Drive", "location": "Cabinet 5, IT Department"},
    {"item": "Tablet", "location": "Meeting Room Shelf"},
    {"item": "Stapler", "location": "Drawer C, Room 104"},
    {"item": "Paper Clips", "location": "Drawer C, Room 104"},
    {"item": "Calendar", "location": "Wall, Room 101"},
    {"item": "Printer", "location": "Printer Room, Ground Floor"},
    {"item": "Scanner", "location": "Printer Room, Ground Floor"},
    {"item": "USB Drive", "location": "Office Desk Drawer"},
    {"item": "Keyboard", "location": "Office Desk Surface"},
    {"item": "Mouse", "location": "Office Desk Surface"},
    {"item": "Monitor", "location": "Office Desk Surface"},
    {"item": "Coffee Mug", "location": "Pantry Shelf 1"},
    {"item": "Water Bottle", "location": "Pantry Shelf 2"},
    {"item": "Chair Cushion", "location": "Locker 2, Room 102"},
    {"item": "Desk Lamp", "location": "Office Desk Surface"},
    {"item": "Sticky Notes", "location": "Drawer A, Room 101"},
    {"item": "Highlighter", "location": "Drawer A, Room 101"},
    {"item": "Binder", "location": "Shelf C, Room 102"},
    {"item": "File Folder", "location": "Shelf D, Room 102"},
    {"item": "Camera", "location": "Media Room Cabinet"},
    {"item": "Tripod", "location": "Media Room Cabinet"},
    {"item": "Microphone", "location": "Media Room Cabinet"},
    {"item": "Speaker", "location": "Conference Room Shelf"},
    {"item": "Extension Cord", "location": "Storage Room 1"},
    {"item": "Cleaning Spray", "location": "Storage Room 2"},
    {"item": "Dust Cloth", "location": "Storage Room 2"},
    {"item": "Fan", "location": "Office Desk Surface"},
    {"item": "Air Purifier", "location": "Corner, Room 101"},
    {"item": "Notebook Charger", "location": "Drawer B, Room 102"},
    {"item": "Pen Stand", "location": "Office Desk Surface"},
    {"item": "Whiteboard Eraser", "location": "Whiteboard Tray, Room 103"},
    {"item": "Projector Remote", "location": "Conference Room Cabinet"},
    {"item": "Laptop Bag", "location": "Locker 1, Room 101"},
    {"item": "Backpack", "location": "Locker 1, Room 101"},
    {"item": "Headset", "location": "Office Desk Surface"},
    {"item": "Notepad", "location": "Drawer D, Room 104"},
    {"item": "Sticky Tape", "location": "Drawer D, Room 104"},
    {"item": "Glue Stick", "location": "Drawer D, Room 104"},
    {"item": "Scissors", "location": "Drawer D, Room 104"},
    {"item": "Envelope", "location": "Shelf E, Room 102"},
    {"item": "Label Maker", "location": "Storage Room 3"},
    {"item": "Calculator", "location": "Drawer F, Room 103"},
    {"item": "Desk Organizer", "location": "Office Desk Surface"},
    {"item": "Clipboard", "location": "Shelf F, Room 104"},
    {"item": "Paper Ream", "location": "Storage Room 1"},
    {"item": "Whiteboard Stand", "location": "Corner, Room 103"},
    {"item": "Notice Board", "location": "Wall, Room 102"},
    {"item": "Magazine", "location": "Pantry Shelf 3"},
]


def load_inventory():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    return [dict(e) for e in DEFAULT_INVENTORY]


def save_inventory(inv):
    with open(DATA_FILE, "w") as f:
        json.dump(inv, f, indent=2)


INVENTORY = load_inventory()


# ── Inventory Operations (programmatic, no hallucination) ───────────
def search_items(query):
    """Search inventory by item name or location keyword."""
    q = query.lower().strip()
    return [e for e in INVENTORY if q in e["item"].lower() or q in e["location"].lower()]


def add_item(item_name, location):
    """Add a new item to inventory."""
    # Check if item already exists
    for e in INVENTORY:
        if e["item"].lower() == item_name.lower():
            old_loc = e["location"]
            e["location"] = location
            save_inventory(INVENTORY)
            return f"Updated **{e['item']}** location from *{old_loc}* to **{location}**."
    INVENTORY.append({"item": item_name.title(), "location": location})
    save_inventory(INVENTORY)
    return f"Added **{item_name.title()}** at **{location}**."


def remove_item(item_name):
    """Remove an item from inventory."""
    for i, e in enumerate(INVENTORY):
        if e["item"].lower() == item_name.lower():
            removed = INVENTORY.pop(i)
            save_inventory(INVENTORY)
            return f"Removed **{removed['item']}** (was at {removed['location']})."
    return f"Item **{item_name}** not found in inventory."


def move_item(item_name, new_location):
    """Move an item to a new location."""
    for e in INVENTORY:
        if e["item"].lower() == item_name.lower():
            old_loc = e["location"]
            e["location"] = new_location
            save_inventory(INVENTORY)
            return f"Moved **{e['item']}** from *{old_loc}* to **{new_location}**."
    return f"Item **{item_name}** not found in inventory."


# ── Intent Parser (no LLM needed — regex-based) ────────────────────
def parse_intent(message):
    """Parse user message into an action + parameters."""
    msg = message.strip()
    msg_lower = msg.lower()

    # ADD: "add laptop to room 101" / "store mouse in drawer A"
    add_match = re.match(
        r"(?:add|store|put|place|save|keep|register)\s+(.+?)\s+(?:to|in|at|on|into)\s+(.+)",
        msg_lower,
    )
    if add_match:
        return "add", add_match.group(1).strip(), add_match.group(2).strip()

    # REMOVE: "remove laptop" / "delete the mouse"
    remove_match = re.match(
        r"(?:remove|delete|discard|throw away|get rid of)\s+(?:the\s+)?(.+)",
        msg_lower,
    )
    if remove_match:
        return "remove", remove_match.group(1).strip(), None

    # MOVE: "move laptop to room 102" / "relocate mouse to drawer B"
    move_match = re.match(
        r"(?:move|relocate|transfer|shift)\s+(?:the\s+)?(.+?)\s+(?:to|into)\s+(.+)",
        msg_lower,
    )
    if move_match:
        return "move", move_match.group(1).strip(), move_match.group(2).strip()

    # LIST by location: "what's in room 101" / "show items in pantry"
    list_match = re.match(
        r"(?:what(?:'s| is| are)\s+in|show\s+(?:items\s+)?(?:in|at)|list\s+(?:items\s+)?(?:in|at))\s+(.+)",
        msg_lower,
    )
    if list_match:
        return "list_location", list_match.group(1).strip().rstrip("?"), None

    # LIST ALL: "show all items" / "list everything"
    if re.match(r"(?:show|list|display)\s+(?:all|every|full)", msg_lower):
        return "list_all", None, None

    # FIND: "where is the laptop" / "find the mouse" / "laptop?"
    find_match = re.match(
        r"(?:where\s+(?:is|are)\s+(?:the\s+)?|find\s+(?:the\s+)?|locate\s+(?:the\s+)?|look\s*(?:for|up)\s+(?:the\s+)?)(.+)",
        msg_lower,
    )
    if find_match:
        return "find", find_match.group(1).strip().rstrip("?"), None

    # COUNT: "how many items" / "total items"
    if re.match(r"(?:how many|count|total)\s+(?:items|things)", msg_lower):
        return "count", None, None

    # HELP
    if msg_lower in ("help", "what can you do", "commands", "?"):
        return "help", None, None

    # Default: treat as a search query
    return "find", msg_lower.rstrip("?"), None


# ── Extract keywords from user message ──────────────────────────────
def extract_keywords(message):
    """Use regex to pull meaningful keywords from the user's query."""
    msg = message.lower().strip()
    # Remove common filler words
    msg = re.sub(r"\b(where|what|which|is|are|the|a|an|my|our|in|on|at|to|of|do|we|have|can|you|find|show|tell|me|please|locate|look|for|get|items?|things?)\b", " ", msg)
    msg = re.sub(r"[?!.,']", " ", msg)
    # Get remaining meaningful words
    keywords = [w.strip() for w in msg.split() if len(w.strip()) > 1]
    return keywords


def search_by_keywords(keywords):
    """Search inventory matching any keyword in item name or location."""
    if not keywords:
        return []
    results = []
    for entry in INVENTORY:
        item_lower = entry["item"].lower()
        loc_lower = entry["location"].lower()
        for kw in keywords:
            if kw in item_lower or kw in loc_lower:
                results.append(entry)
                break
    return results


# ── Build focused prompt for LLM ────────────────────────────────────
def build_prompt(message, history, keywords, results):
    """Build a small, focused prompt with only relevant inventory data."""
    if results:
        data = "\n".join(f"- {r['item']} is located at {r['location']}" for r in results)
        context = f"Database search results for \"{', '.join(keywords)}\":\n{data}"
    else:
        context = f"No items found matching \"{', '.join(keywords)}\" in the database."

    prompt = (
        f"You are an office item tracking assistant.\n"
        f"{context}\n\n"
        f"Answer the user's question using ONLY the data above. "
        f"Do NOT invent items or locations. Keep it short (1-2 sentences).\n\n"
    )
    # Add recent chat history (last 4 messages only to save tokens)
    recent = history[-4:] if len(history) > 4 else history
    for msg in recent:
        role = msg.get("role", "")
        content = msg.get("content", "")
        prompt += f"{role}: {content}\n"
    prompt += f"user: {message}\nassistant:"
    return prompt


# ── Chat Response (regex search + LLM answer) ───────────────────────
def respond(message, history, model_name):
    if not message.strip():
        return history, "", ""

    # Handle add/remove/move programmatically (DB writes)
    intent, arg1, arg2 = parse_intent(message)
    if intent in ("add", "remove", "move"):
        if intent == "add":
            reply = add_item(arg1, arg2)
        elif intent == "remove":
            reply = remove_item(arg1)
        else:
            reply = move_item(arg1, arg2)
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": reply})
        return history, "", ""

    # Step 1: Regex extracts keywords from the question
    keywords = extract_keywords(message)

    # Step 2: Search inventory for only matching items
    results = search_by_keywords(keywords)

    # Step 3: Build a tiny prompt with ONLY the matched items
    prompt = build_prompt(message, history, keywords, results)

    # Step 4: Let the LLM answer based on the filtered data
    try:
        text, stats = generate(model_name, prompt, max_tokens=80, temp=0.3)
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": text})
        return history, "", stats
    except Exception as e:
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": f"Error: {e}"})
        return history, "", f"Error: {e}"


# ── Model Loading ───────────────────────────────────────────────────
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


# ── Quick Search ────────────────────────────────────────────────────
def quick_search(query):
    if not query.strip():
        return ""
    results = search_items(query)
    if not results:
        return f"No items found matching **'{query}'**."
    header = f"Found **{len(results)}** result(s) for **'{query}'**:\n\n"
    rows = "| Item | Location |\n|------|----------|\n"
    for r in results:
        rows += f"| {r['item']} | {r['location']} |\n"
    return header + rows


# ── Inventory Table ─────────────────────────────────────────────────
def get_inventory_table(filter_text=""):
    filtered = INVENTORY
    if filter_text.strip():
        q = filter_text.lower()
        filtered = [e for e in INVENTORY if q in e["item"].lower() or q in e["location"].lower()]
    return [[e["item"], e["location"]] for e in filtered]


# ── Compare Models (with chat history for both) ─────────────────────
def compare(message, history1, history2):
    if not message.strip():
        return history1, history2, "", "", ""

    keywords = extract_keywords(message)
    results = search_by_keywords(keywords)

    # Build prompts with each model's own chat history
    prompt1 = build_prompt(message, history1, keywords, results)
    prompt2 = build_prompt(message, history2, keywords, results)

    out1, stats1 = generate("LFM2-350M", prompt1, max_tokens=80, temp=0.3)
    out2, stats2 = generate("SmolLM2-360M", prompt2, max_tokens=80, temp=0.3)

    # Append to each model's history
    history1.append({"role": "user", "content": message})
    history1.append({"role": "assistant", "content": out1})
    history2.append({"role": "user", "content": message})
    history2.append({"role": "assistant", "content": out2})

    return history1, history2, "", stats1, stats2


# ── UI ──────────────────────────────────────────────────────────────
CSS = """
.main-wrap { max-width: 900px; margin: 0 auto; }
.header { text-align: center; padding: 20px 0 10px; }
.header h1 { font-size: 1.6em; margin: 0; }
.header p { color: #666; margin: 4px 0 0; font-size: 0.95em; }
.stats-bar { font-size: 0.8em; color: #888; text-align: right; min-height: 20px; }
.search-results { min-height: 50px; }
"""

with gr.Blocks(title="AI Item Tracker") as app:

    with gr.Column(elem_classes="main-wrap"):
        gr.HTML("""
            <div class="header">
                <h1>AI Item Tracker</h1>
                <p>Track, find, add, and manage office items with AI</p>
            </div>
        """)

        with gr.Tabs():
            # ── Ask AI Tab ──────────────────────────────────────────
            with gr.Tab("Ask AI"):
                gr.Markdown(
                    '*Try: "Where is the laptop?" · "Add coffee machine to Pantry" · '
                    '"Move mouse to Room 102" · "What\'s in Room 101?" · "help"*'
                )
                model_pick = gr.Radio(
                    choices=list(MODELS.keys()),
                    value="LFM2-350M",
                    label="Model",
                    interactive=True,
                )
                chat = gr.Chatbot(
                    height=400, show_label=False,
                    placeholder="Ask me to find, add, move, or remove items...",
                )
                stats_display = gr.Markdown(elem_classes="stats-bar")
                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="Where is the laptop?",
                        show_label=False, scale=6, container=False,
                    )
                    send = gr.Button("Send", variant="primary", scale=1, min_width=80)
                clear = gr.Button("Clear Chat", size="sm")

                send.click(respond, [msg, chat, model_pick], [chat, msg, stats_display])
                msg.submit(respond, [msg, chat, model_pick], [chat, msg, stats_display])
                clear.click(lambda: ([], "", ""), outputs=[chat, msg, stats_display])

            # ── Quick Search Tab ────────────────────────────────────
            with gr.Tab("Quick Search"):
                gr.Markdown("*Instant lookup — search by item name or location.*")
                search_input = gr.Textbox(
                    placeholder="Search items or locations...",
                    show_label=False, container=False,
                )
                search_btn = gr.Button("Search", variant="primary")
                search_results = gr.Markdown(elem_classes="search-results")

                search_btn.click(quick_search, [search_input], [search_results])
                search_input.submit(quick_search, [search_input], [search_results])

            # ── Inventory Tab ───────────────────────────────────────
            with gr.Tab("Inventory"):
                gr.Markdown("*Full inventory list. Use the filter to narrow down.*")
                filter_box = gr.Textbox(
                    placeholder="Filter by item or location...",
                    show_label=False, container=False,
                )
                inv_table = gr.Dataframe(
                    headers=["Item", "Location"],
                    value=get_inventory_table(),
                    interactive=False,
                )
                filter_box.change(
                    lambda f: get_inventory_table(f),
                    [filter_box], [inv_table],
                )

            # ── Compare Tab ─────────────────────────────────────────
            with gr.Tab("Compare Models"):
                gr.Markdown("*Both models remember the conversation. Ask follow-ups!*")
                with gr.Row(equal_height=True):
                    with gr.Column():
                        gr.Markdown("### LFM2-350M")
                        chat1 = gr.Chatbot(height=350, show_label=False)
                        stats1_display = gr.Markdown(elem_classes="stats-bar")
                    with gr.Column():
                        gr.Markdown("### SmolLM2-360M")
                        chat2 = gr.Chatbot(height=350, show_label=False)
                        stats2_display = gr.Markdown(elem_classes="stats-bar")
                with gr.Row():
                    cmp_input = gr.Textbox(
                        placeholder="Where is the projector remote?",
                        show_label=False, scale=6, container=False,
                    )
                    cmp_btn = gr.Button("Compare", variant="primary", scale=1, min_width=80)
                cmp_clear = gr.Button("Clear Both Chats", size="sm")

                cmp_btn.click(compare, [cmp_input, chat1, chat2], [chat1, chat2, cmp_input, stats1_display, stats2_display])
                cmp_input.submit(compare, [cmp_input, chat1, chat2], [chat1, chat2, cmp_input, stats1_display, stats2_display])
                cmp_clear.click(lambda: ([], [], "", "", ""), outputs=[chat1, chat2, cmp_input, stats1_display, stats2_display])


if __name__ == "__main__":
    print(f"Device: {device}")
    print(f"Tracking {len(INVENTORY)} items")
    print("Starting server...")
    app.launch(css=CSS, share=True)
