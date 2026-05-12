"""
LexShield AI — Document Classifier FINAL
=========================================
Loads fine-tuned InLegalBERT from models/saved/inlegalbert_finetuned/.
Falls back to centroid mode if fine-tuned model not found.
Same predict() interface as all previous versions.
"""

import os
import gc
import logging
import numpy as np
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import torch
torch.set_num_threads(2)

logger = logging.getLogger(__name__)

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

CONFIDENCE_THRESHOLD = 0.15
FINETUNED_DIR        = Path("models/saved/inlegalbert_finetuned")
PRETRAINED_MODEL     = "law-ai/InLegalBERT"
SAMPLES_DIR          = Path("data/classifier_samples")

# Fallback centroid prototypes (used only if fine-tuned model missing)
_PROTOTYPES: dict[str, list[str]] = {
    "rental_agreement":     ["This rental agreement executed between landlord and tenant for residential premises.",
                             "Monthly rent payable by tenant to landlord. Security deposit refundable on vacating."],
    "fir":                  ["First Information Report registered at police station under Section 154 CrPC.",
                             "Cognizable offence reported and FIR registered by officer in charge."],
    "court_notice_summons": ["You are hereby summoned to appear before this Honourable Court.",
                             "Notice issued to show cause. Failure to appear results in ex-parte proceedings."],
    "employment_contract":  ["This appointment letter confirms selection as employee at monthly salary.",
                             "Employee subject to probation period service rules and notice period."],
    "property_deed":        ["This sale deed executed between vendor and purchaser for valuable consideration.",
                             "Property transferred absolutely free from encumbrances registered at Sub-Registrar."],
    "sc_judgment":          ["In the Supreme Court of India civil appellate jurisdiction judgment delivered.",
                             "Special leave petition disposed of by the apex court bench of Justices."],
    "hc_judgment":          ["In the High Court of Kerala writ petition allowed and impugned order quashed.",
                             "Division bench of High Court heard writ appeal and passed following order."],
    "legal_notice":         ["Take notice my client demands payment of outstanding dues within fifteen days.",
                             "Failing compliance my client shall initiate civil and criminal proceedings."],
    "affidavit":            ["I the deponent do hereby solemnly affirm and declare on oath as follows.",
                             "Sworn before notary public this affidavit for declaration of facts."],
    "power_of_attorney":    ["I hereby nominate constitute and appoint my attorney to act on my behalf.",
                             "General power of attorney authorizing grantee to execute documents and appear."],
    "cheque_bounce_notice": ["Cheque returned dishonoured with memo insufficient funds Section 138 NI Act.",
                             "Demand payment within fifteen days failing which criminal complaint shall be filed."],
    "bail_application":     ["Application for bail filed by accused before Sessions Court Section 437 CrPC.",
                             "Applicant seeks regular bail on grounds of first offender and family dependents."],
    "consumer_complaint":   ["Complaint filed before District Consumer Disputes Redressal Commission.",
                             "Deficiency in service and unfair trade practice under Consumer Protection Act 2019."],
    "loan_agreement":       ["Loan agreement between lender and borrower for principal amount in rupees.",
                             "Borrower agrees to repay loan with interest in monthly EMI instalments."],
    "police_complaint":     ["Complaint submitted to Station House Officer requesting registration of FIR.",
                             "Request to take appropriate legal action against the persons named herein."],
}


class DocumentClassifier:

    def __init__(self):
        self._tokenizer  = None
        self._model      = None
        self._centroids  = None
        self._mode       = "not_loaded"
        self._ready      = False
        self._load()

    def _load(self):
        # Try fine-tuned first
        if FINETUNED_DIR.exists() and (FINETUNED_DIR / "config.json").exists():
            self._load_finetuned()
        else:
            print("[Classifier] Fine-tuned model not found. Using centroid fallback.")
            print("  To get best accuracy run:")
            print("  1. python -m models.data_generator --samples 50")
            print("  2. python -m models.finetune")
            self._load_centroid()

    def _load_finetuned(self):
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            print(f"[Classifier] Loading fine-tuned model from {FINETUNED_DIR}...")
            self._tokenizer = AutoTokenizer.from_pretrained(str(FINETUNED_DIR))
            self._model     = AutoModelForSequenceClassification.from_pretrained(
                str(FINETUNED_DIR)
            )
            self._model.eval()
            self._mode  = "finetuned"
            self._ready = True
            print("[Classifier] Fine-tuned InLegalBERT loaded. ✓")
        except Exception as e:
            print(f"[Classifier] Fine-tuned load failed ({e}), falling back to centroid.")
            self._load_centroid()

    def _load_centroid(self):
        try:
            from transformers import AutoTokenizer, AutoModel

            print(f"[Classifier] Loading {PRETRAINED_MODEL} for centroid mode...")
            self._tokenizer  = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
            self._base_model = AutoModel.from_pretrained(PRETRAINED_MODEL)
            self._base_model.eval()

            print("[Classifier] Building centroids...")
            centroids = []
            for idx in range(len(CATEGORIES)):
                category  = CATEGORIES[idx]
                texts     = self._load_sample_texts(category)
                embeddings = [self._embed_text(t[:600]) for t in texts]
                centroid   = np.mean(embeddings, axis=0)
                centroid   = centroid / (np.linalg.norm(centroid) + 1e-9)
                centroids.append(centroid)
                gc.collect()

            self._centroids = np.stack(centroids)
            self._mode      = "centroid"
            self._ready     = True
            print("[Classifier] Centroid mode ready. ✓")
        except Exception as e:
            print(f"[Classifier] Centroid load also failed: {e}")

    def _load_sample_texts(self, category: str) -> list[str]:
        texts   = []
        cat_dir = SAMPLES_DIR / category
        if cat_dir.exists():
            for f in sorted(cat_dir.glob("*.txt"))[:20]:
                try:
                    t = f.read_text(encoding="utf-8", errors="ignore").strip()
                    if len(t) > 80:
                        texts.append(t)
                except Exception:
                    continue
        texts.extend(_PROTOTYPES.get(category, []))
        return texts or _PROTOTYPES.get(category, ["legal document"])

    @torch.no_grad()
    def _embed_text(self, text: str) -> np.ndarray:
        enc  = self._tokenizer(text, return_tensors="pt",
                               max_length=256, truncation=True, padding=True)
        out  = self._base_model(**enc)
        mask = enc["attention_mask"].unsqueeze(-1).float()
        emb  = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
        return emb.squeeze(0).numpy()

    @torch.no_grad()
    def predict(self, text: str) -> dict:
        if not self._ready:
            return self._err("Classifier not loaded.")
        if not text or len(text.strip()) < 20:
            return self._err("Text too short.")

        try:
            snippet = text[:800]

            if self._mode == "finetuned":
                enc     = self._tokenizer(snippet, return_tensors="pt",
                                        max_length=256, truncation=True, padding=True)
                outputs = self._model(**enc)
                logits  = outputs.logits.squeeze(0).numpy()
                exp_l   = np.exp(logits - logits.max())
                proba   = exp_l / exp_l.sum()
            else:
                emb     = self._embed_text(snippet)
                emb     = emb / (np.linalg.norm(emb) + 1e-9)
                sims    = self._centroids @ emb
                exp_l   = np.exp(sims * 5.0)
                proba   = exp_l / exp_l.sum()

            label      = int(np.argmax(proba))
            confidence = float(proba[label])
            uncertain  = confidence < CONFIDENCE_THRESHOLD

            # Always return the top predicted label — don't hide it as "uncertain"
            # uncertain flag still set so callers can decide how to handle
            label_name = CATEGORIES[label]

            out = {
                "label":      label,
                "label_name": label_name,
                "confidence": round(confidence, 4),
                "uncertain":  uncertain,
                "all_scores": {
                    CATEGORIES[i]: round(float(proba[i]), 4)
                    for i in range(len(CATEGORIES))
                },
                "mode": self._mode,
            }
            if uncertain:
                out["warning"] = (
                    f"Low confidence ({confidence:.2f}). "
                    f"Prediction may be unreliable."
                )
            return out

        except Exception as e:
            logger.error("predict() error: %s", e)
            return self._err(str(e))

    def is_ready(self)  -> bool: return self._ready
    def get_mode(self)  -> str:  return self._mode

    def reload(self) -> bool:
        self._tokenizer = self._model = self._centroids = None
        self._ready     = False
        self._load()
        return self._ready

    @staticmethod
    def _err(msg: str) -> dict:
        return {"label": -1, "label_name": "unknown", "confidence": 0.0,
                "uncertain": True, "all_scores": {}, "warning": msg}


classifier = DocumentClassifier()