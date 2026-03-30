"""Benchmark: Which LLM is better for item tracking?"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from chatbot import (
    load_model, generate, extract_keywords, search_by_keywords,
    build_prompt, INVENTORY
)

# -- Test Cases ──────────────────────────────────────────────────────
# Each test: (question, expected_keywords_in_answer)
# We check if the LLM output contains the correct location/item info

TEST_CASES = [
    # Direct item lookup
    ("Where is the laptop?", ["office desk drawer"]),
    ("Where is the projector?", ["conference room cabinet"]),
    ("Where is the router?", ["server room shelf 2"]),
    ("Where are the scissors?", ["drawer d", "room 104"]),
    ("Where is the camera?", ["media room cabinet"]),
    ("Where is the printer?", ["printer room", "ground floor"]),
    ("Where is the coffee mug?", ["pantry shelf 1"]),
    ("Where is the air purifier?", ["corner", "room 101"]),
    ("Where is the label maker?", ["storage room 3"]),
    ("Where is the calculator?", ["drawer f", "room 103"]),

    # Location-based queries
    ("What items are on the Office Desk Surface?", ["keyboard", "mouse", "monitor"]),
    ("What is in the Media Room Cabinet?", ["camera", "tripod", "microphone"]),
    ("What is in Drawer D, Room 104?", ["notepad", "sticky tape", "scissors"]),

    # Item not in inventory
    ("Where is the television?", ["not"]),  # should say "not found" or similar

    # Synonym / natural phrasing
    ("Find the headphones", ["locker 3", "room 101"]),
    ("Locate the whiteboard marker", ["whiteboard tray", "room 103"]),
    ("I need the charger", ["drawer under office desk"]),
    ("Can you find the USB drive?", ["office desk drawer"]),
    ("Where did we put the stapler?", ["drawer c", "room 104"]),
    ("I'm looking for the backpack", ["locker 1", "room 101"]),
]


def check_answer(answer, expected_keywords):
    """Check if answer contains ALL expected keywords (case-insensitive)."""
    answer_lower = answer.lower()
    return all(kw in answer_lower for kw in expected_keywords)


def run_benchmark():
    print("=" * 70)
    print("  LLM ITEM TRACKING BENCHMARK")
    print("=" * 70)
    print(f"  Total test cases: {len(TEST_CASES)}")
    print(f"  Inventory size: {len(INVENTORY)} items")
    print("=" * 70)

    results = {"LFM2-350M": [], "SmolLM2-360M": []}

    for i, (question, expected) in enumerate(TEST_CASES, 1):
        print(f"\n-- Test {i}/{len(TEST_CASES)}: {question}")
        print(f"   Expected keywords: {expected}")

        keywords = extract_keywords(question)
        search_results = search_by_keywords(keywords)
        prompt = build_prompt(question, [], keywords, search_results)

        for model_name in ["LFM2-350M", "SmolLM2-360M"]:
            answer, stats = generate(model_name, prompt, max_tokens=80, temp=0.1)
            passed = check_answer(answer, expected)
            results[model_name].append(passed)

            status = "PASS" if passed else "FAIL"
            print(f"   [{model_name}] {status} | {stats}")
            # Show first 120 chars of answer
            short = answer.replace("\n", " ")[:120]
            print(f"      -> {short}")

    # -- Summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)

    for model_name in ["LFM2-350M", "SmolLM2-360M"]:
        passed = sum(results[model_name])
        total = len(results[model_name])
        pct = (passed / total) * 100
        bar = "#" * passed + "-" * (total - passed)
        print(f"\n  {model_name}:")
        print(f"    Score: {passed}/{total} ({pct:.0f}%)")
        print(f"    [{bar}]")

        # Show which tests failed
        failures = [i for i, p in enumerate(results[model_name]) if not p]
        if failures:
            print(f"    Failed tests:")
            for idx in failures:
                print(f"      - Test {idx+1}: {TEST_CASES[idx][0]}")

    # -- Winner ──────────────────────────────────────────────────────
    s1 = sum(results["LFM2-350M"])
    s2 = sum(results["SmolLM2-360M"])
    print("\n" + "=" * 70)
    if s1 > s2:
        print(f"  WINNER: LFM2-350M ({s1} vs {s2})")
    elif s2 > s1:
        print(f"  WINNER: SmolLM2-360M ({s2} vs {s1})")
    else:
        print(f"  TIE: Both scored {s1}/{len(TEST_CASES)}")
    print("=" * 70)


if __name__ == "__main__":
    run_benchmark()
