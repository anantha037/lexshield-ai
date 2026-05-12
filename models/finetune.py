# models/finetune.py
"""
Fine-tune InLegalBERT on your 15-category synthetic dataset.
Run: python -m models.finetune
Time: ~45 minutes on CPU (8GB RAM safe)
Output: models/saved/inlegalbert_finetuned/
"""

import os
import gc
import json
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import torch
torch.set_num_threads(2)

from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME   = "law-ai/InLegalBERT"
SAMPLES_DIR  = Path("data/classifier_samples")
OUTPUT_DIR   = Path("models/saved/inlegalbert_finetuned")
MAX_LENGTH   = 256        # tokens — 512 causes OOM on 8GB, 256 is safe
BATCH_SIZE   = 4          # RAM safe
GRAD_ACCUM   = 4          # effective batch = 4×4 = 16
EPOCHS       = 5
LR           = 2e-5
WARMUP_RATIO = 0.1

CATEGORIES: dict[int, str] = {
    0:  "rental_agreement",
    1:  "fir",
    2:  "court_notice_summons",
    3:  "employment_contract",
    4:  "property_deed",
    5:  "sc_judgment",
    6:  "hc_judgment",
    7:  "legal_notice",
    8:  "affidavit",
    9:  "power_of_attorney",
    10: "cheque_bounce_notice",
    11: "bail_application",
    12: "consumer_complaint",
    13: "loan_agreement",
    14: "police_complaint",
}
CAT_TO_IDX = {v: k for k, v in CATEGORIES.items()}


# ── Dataset ───────────────────────────────────────────────────────────────────

class LegalDocDataset(Dataset):

    def __init__(self, samples: list[dict], tokenizer, max_length: int):
        self.samples    = samples
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item  = self.samples[idx]
        text  = item["text"]
        label = item["label"]

        # Sample first 800 chars — captures header/type signal
        # Most classification signal is in opening lines
        snippet = text[:800]

        enc = self.tokenizer(
            snippet,
            max_length     = self.max_length,
            padding        = "max_length",
            truncation     = True,
            return_tensors = "pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          torch.tensor(label, dtype=torch.long),
        }


def load_samples(samples_dir: Path) -> tuple[list[dict], list[dict]]:
    """Load all .txt files, return (train_samples, val_samples)."""
    all_samples = []

    for idx, category in CATEGORIES.items():
        cat_dir = samples_dir / category
        if not cat_dir.exists():
            print(f"  WARNING: {cat_dir} not found — run data_generator.py first")
            continue

        files = sorted(cat_dir.glob("*.txt"))
        if not files:
            print(f"  WARNING: No files in {cat_dir}")
            continue

        for fpath in files:
            try:
                text = fpath.read_text(encoding="utf-8", errors="ignore").strip()
                if len(text) > 80:
                    all_samples.append({"text": text, "label": idx, "category": category})
            except Exception:
                continue

        print(f"  {category:30s}: {len(files)} files loaded")

    # 80/20 split, stratified manually
    from collections import defaultdict
    import random
    random.seed(42)

    by_class: dict[int, list] = defaultdict(list)
    for s in all_samples:
        by_class[s["label"]].append(s)

    train, val = [], []
    for label, samples in by_class.items():
        random.shuffle(samples)
        cut = max(1, int(len(samples) * 0.8))
        train.extend(samples[:cut])
        val.extend(samples[cut:])

    random.shuffle(train)
    random.shuffle(val)

    print(f"\n  Train: {len(train)} samples | Val: {len(val)} samples")
    return train, val


# ── Training loop ─────────────────────────────────────────────────────────────

def train():
    print("=" * 60)
    print("LexShield AI — InLegalBERT Fine-Tuning")
    print("=" * 60)

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\n[1/5] Loading training data...")
    train_samples, val_samples = load_samples(SAMPLES_DIR)

    if len(train_samples) < 15:
        print("\nERROR: Not enough data. Run first:")
        print("  python -m models.data_generator --samples 50")
        return

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\n[2/5] Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels          = len(CATEGORIES),
        ignore_mismatched_sizes = True,
    )
    model.train()
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── DataLoaders ───────────────────────────────────────────────────────────
    train_dataset = LegalDocDataset(train_samples, tokenizer, MAX_LENGTH)
    val_dataset   = LegalDocDataset(val_samples,   tokenizer, MAX_LENGTH)

    train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader    = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)

    # ── Optimizer + scheduler ─────────────────────────────────────────────────
    print("\n[3/5] Setting up optimizer...")
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    total_steps   = (len(train_loader) // GRAD_ACCUM) * EPOCHS
    warmup_steps  = int(total_steps * WARMUP_RATIO)
    scheduler     = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps   = warmup_steps,
        num_training_steps = total_steps,
    )

    print(f"  Total steps: {total_steps} | Warmup: {warmup_steps}")

    # ── Training ──────────────────────────────────────────────────────────────
    print(f"\n[4/5] Training for {EPOCHS} epochs...")
    best_val_acc = 0.0
    best_epoch   = 0

    for epoch in range(EPOCHS):
        # Train
        model.train()
        total_loss    = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            outputs = model(
                input_ids      = batch["input_ids"],
                attention_mask = batch["attention_mask"],
                labels         = batch["label"],
            )
            loss = outputs.loss / GRAD_ACCUM
            loss.backward()
            total_loss += outputs.loss.item()

            if (step + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if (step + 1) % 20 == 0:
                print(f"  Epoch {epoch+1} | Step {step+1}/{len(train_loader)} "
                      f"| Loss: {total_loss/(step+1):.4f}")

        gc.collect()

        # Validate
        model.eval()
        correct = 0
        total   = 0
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(
                    input_ids      = batch["input_ids"],
                    attention_mask = batch["attention_mask"],
                )
                preds    = outputs.logits.argmax(dim=-1)
                correct += (preds == batch["label"]).sum().item()
                total   += len(batch["label"])

        val_acc = correct / total if total > 0 else 0
        avg_loss = total_loss / len(train_loader)

        print(f"\n  Epoch {epoch+1}/{EPOCHS} complete")
        print(f"  Train Loss: {avg_loss:.4f} | Val Accuracy: {val_acc*100:.1f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch   = epoch + 1
            # Save best checkpoint
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(OUTPUT_DIR)
            tokenizer.save_pretrained(OUTPUT_DIR)
            print(f"  ✓ Best model saved (val_acc={val_acc*100:.1f}%)")

        gc.collect()

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n[5/5] Training complete!")
    print(f"  Best val accuracy : {best_val_acc*100:.1f}% at epoch {best_epoch}")
    print(f"  Model saved to    : {OUTPUT_DIR}")

    # Save metadata
    meta = {
        "model":          MODEL_NAME,
        "best_val_acc":   round(best_val_acc, 4),
        "best_epoch":     best_epoch,
        "num_categories": len(CATEGORIES),
        "categories":     CATEGORIES,
        "max_length":     MAX_LENGTH,
    }
    with open(OUTPUT_DIR / "training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return meta


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    train()