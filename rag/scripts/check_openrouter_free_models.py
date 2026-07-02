"""
OpenRouter Free Model Verifier
==============================
Run this before updating any free-tier model slug in rag/multi_llm.py.

Usage:
    python rag/scripts/check_openrouter_free_models.py

Prints all models currently listed at price=0 on OpenRouter's live API,
sorted by context length descending, plus spot-checks specific slugs.
"""
import urllib.request
import json

url = 'https://openrouter.ai/api/v1/models'
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
with urllib.request.urlopen(req, timeout=20) as r:
    data = json.loads(r.read())

models = data.get('data', [])
free_models = []
for m in models:
    mid = m.get('id', '')
    pricing = m.get('pricing', {})
    try:
        prompt_price = float(pricing.get('prompt') or '1')
    except (TypeError, ValueError):
        prompt_price = 1.0
    try:
        completion_price = float(pricing.get('completion') or '1')
    except (TypeError, ValueError):
        completion_price = 1.0

    # if is inside the for loop — intentional, evaluates per model
    if prompt_price == 0 and completion_price == 0:
        context = m.get('context_length', 0)
        free_models.append((mid, context))

free_models.sort(key=lambda x: -x[1])
print(f"=== All FREE models on OpenRouter ({len(free_models)} total) ===")
for mid, ctx in free_models:
    print(f"  {mid}  (context: {ctx})")

# Spot-check current _PROVIDERS slugs
print()
print("=== Slot verification for _PROVIDERS in rag/multi_llm.py ===")
slots = {
    "slot 2 (priority=2)": "meta-llama/llama-3.3-70b-instruct:free",
    "slot 3 (priority=3)": "qwen/qwen3-next-80b-a3b-instruct:free",
    "slot 5 (priority=5)": "google/gemma-4-31b-it:free",
    "slot 6 (priority=6)": "nvidia/nemotron-3-nano-30b-a3b:free",
    "slot 7 (priority=7)": "openrouter/free",
}
id_set   = {m.get('id', '') for m in models}
free_set = {mid for mid, _ in free_models}
for label, slug in slots.items():
    exists  = slug in id_set
    is_free = slug in free_set
    status  = "OK" if (exists and is_free) else "DEAD/PAID — UPDATE NEEDED"
    print(f"  {label}: {slug}")
    print(f"    -> {status}")
