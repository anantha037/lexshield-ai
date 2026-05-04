import os
import gc
import json
import pickle
from pathlib import Path
from typing import Optional

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

# ── Category map ──────────────────────────────────────────────────────────────
CATEGORIES: dict[int, str] = {
    0: "rental_agreement",
    1: "fir",
    2: "court_notice",
    3: "employment_contract",
    4: "property_deed",
    5: "sc_judgment",
    6: "hc_judgment",
    7: "legal_notice",
}
LABEL_TO_IDX: dict[str, int] = {v: k for k, v in CATEGORIES.items()}

# Model save paths
MODEL_DIR       = Path("models/saved")
CLASSIFIER_PATH = MODEL_DIR / "document_classifier.pkl"
VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.pkl"

# Prediction confidence threshold — below this, label as "uncertain"
CONFIDENCE_THRESHOLD = 0.45


# ── Training ──────────────────────────────────────────────────────────────────

def train(samples_per_class: int = 150, save: bool = True) -> dict:
    """
    Generate data → TF-IDF → XGBoost (regularised) → CV evaluate → save.

    Key changes from v1:
    - StratifiedKFold (5-fold) replaces single train/test split
    - max_depth reduced 6→4, colsample_bytree 0.8→0.7 (reduce capacity)
    - reg_alpha=0.5, reg_lambda=2.0 added (L1 + L2 regularisation)
    - min_child_weight=3 (min samples per leaf, combats over-specific splits)
    - sublinear_tf=True kept, min_df raised 2→3 (prune rare tokens)
    Returns evaluation metrics dict.
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.model_selection import StratifiedKFold, cross_validate
        from sklearn.metrics import (
            accuracy_score, classification_report, confusion_matrix
        )
        import xgboost as xgb
        import numpy as np
    except ImportError as e:
        raise ImportError(f"Missing dependency: {e}\n"
                          "Run: pip install scikit-learn xgboost")

    from models.training_data import generate_dataset

    print("=" * 60)
    print("LexShield AI — Document Classifier Training (v2)")
    print("=" * 60)

    # ── Step 1: Generate data ─────────────────────────────────────────────────
    print(f"\n[1/5] Generating {samples_per_class * 8} training samples "
          f"({len(CATEGORIES)} classes × {samples_per_class} samples each)...")
    dataset   = generate_dataset(samples_per_class)
    texts     = [d["text"]  for d in dataset]
    labels    = [d["label"] for d in dataset]

    # ── Step 2: TF-IDF vectorisation ──────────────────────────────────────────
    print("\n[2/5] TF-IDF vectorisation...")
    vectorizer = TfidfVectorizer(
        ngram_range   = (1, 2),       # unigrams + bigrams
        max_features  = 20_000,
        sublinear_tf  = True,         # log(tf) dampens high-frequency terms
        min_df        = 3,            # raised from 2: prune hapax features
        max_df        = 0.90,         # lowered from 0.95: prune near-universal terms
        strip_accents = "unicode",
        analyzer      = "word",
        token_pattern = r"(?u)\b\w[\w\-\.]+\b",   # keeps "Pvt.Ltd", "u/s"
    )
    X = vectorizer.fit_transform(texts)
    y = np.array(labels)
    print(f"  Feature matrix: {X.shape[0]} docs × {X.shape[1]} features")

    # ── Step 3: Define model with regularisation ───────────────────────────────
    print("\n[3/5] Configuring regularised XGBoost...")
    model = xgb.XGBClassifier(
        n_estimators      = 200,       # reduced from 300 (less capacity)
        max_depth         = 4,         # reduced from 6 (shallower = less overfit)
        learning_rate     = 0.1,
        subsample         = 0.8,
        colsample_bytree  = 0.7,       # reduced from 0.8
        min_child_weight  = 3,         # NEW: min samples per leaf node
        reg_alpha         = 0.5,       # NEW: L1 regularisation
        reg_lambda        = 2.0,       # NEW: L2 regularisation (default was 1.0)
        use_label_encoder = False,
        eval_metric       = "mlogloss",
        n_jobs            = 2,
        random_state      = 42,
        verbosity         = 0,
    )

    # ── Step 4: 5-fold stratified cross-validation ────────────────────────────
    print("\n[4/5] Running 5-fold Stratified Cross-Validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    cv_results = cross_validate(
        model, X, y,
        cv            = cv,
        scoring       = ["accuracy", "f1_macro"],
        return_train_score = True,
        n_jobs        = 1,            # sequential: avoid RAM spike on 8GB
    )

    mean_train_acc = cv_results["train_accuracy"].mean()
    mean_val_acc   = cv_results["test_accuracy"].mean()
    mean_val_f1    = cv_results["test_f1_macro"].mean()
    overfit_gap    = mean_train_acc - mean_val_acc

    print(f"\n  CV Train Accuracy (mean): {mean_train_acc * 100:.1f}%")
    print(f"  CV Val   Accuracy (mean): {mean_val_acc   * 100:.1f}%")
    print(f"  CV Val   F1-Macro (mean): {mean_val_f1    * 100:.1f}%")
    print(f"  Overfitting gap         : {overfit_gap    * 100:.1f}%  "
          f"{'✓ ACCEPTABLE (< 10%)' if overfit_gap < 0.10 else '✗ HIGH — consider more data or less capacity'}")

    # ── Step 5: Fit final model on full data + evaluate ───────────────────────
    print("\n[5/5] Fitting final model on full dataset...")
    model.fit(X, y, verbose=False)

    y_pred_full = model.predict(X)
    train_acc   = accuracy_score(y, y_pred_full)
    category_names = [CATEGORIES[i] for i in range(8)]
    report = classification_report(
        y, y_pred_full,
        target_names = category_names,
        output_dict  = True,
    )
    cm = confusion_matrix(y, y_pred_full)

    print(f"\n  Final model train accuracy : {train_acc * 100:.1f}%")
    print(f"  Validation accuracy (CV)   : {mean_val_acc * 100:.1f}%")
    print(f"\n  Per-class F1 scores (train, for reference):")
    for cat in category_names:
        f1 = report[cat]["f1-score"]
        print(f"    {cat:25s}  F1={f1:.3f}")

    print(f"\n  Confusion matrix:\n{cm}")

    target_met = mean_val_acc >= 0.85
    print(f"\n  {'✓ TARGET MET' if target_met else '✗ TARGET NOT MET'} "
          f"(CV val accuracy={mean_val_acc*100:.1f}%, target=85%)")

    # ── Save ──────────────────────────────────────────────────────────────────
    if save:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        with open(CLASSIFIER_PATH, "wb") as f:
            pickle.dump(model, f)
        with open(VECTORIZER_PATH, "wb") as f:
            pickle.dump(vectorizer, f)
        print(f"\n  Saved model      → {CLASSIFIER_PATH}")
        print(f"  Saved vectorizer → {VECTORIZER_PATH}")

    gc.collect()

    return {
        "cv_val_accuracy":    round(mean_val_acc,  4),
        "cv_train_accuracy":  round(mean_train_acc, 4),
        "cv_val_f1_macro":    round(mean_val_f1,    4),
        "overfit_gap":        round(overfit_gap,     4),
        "target_met":         target_met,
        "per_class_train_f1": {cat: round(report[cat]["f1-score"], 4) for cat in category_names},
        "confusion_matrix":   cm.tolist(),
    }


# ── Inference class ───────────────────────────────────────────────────────────

class DocumentClassifier:
    """
    Loads the trained XGBoost model and runs document classification.

    Usage:
        from models.classifier import classifier
        result = classifier.predict(text)
    """

    def __init__(self):
        self._model      = None
        self._vectorizer = None
        self._ready      = False
        self._load()

    def _load(self) -> None:
        if not CLASSIFIER_PATH.exists() or not VECTORIZER_PATH.exists():
            print("[Classifier] Model not found. Run: python -m models.train_classifier")
            return
        try:
            with open(CLASSIFIER_PATH, "rb") as f:
                self._model = pickle.load(f)
            with open(VECTORIZER_PATH, "rb") as f:
                self._vectorizer = pickle.load(f)
            self._ready = True
            print("[Classifier] Document classifier loaded.")
        except Exception as e:
            print(f"[Classifier] Load failed: {e}")

    def predict(self, text: str) -> dict:
        """
        Predict document type.

        Returns:
          {
            "label":       0,
            "label_name":  "rental_agreement",
            "confidence":  0.87,
            "uncertain":   False,
            "all_scores":  {"rental_agreement": 0.87, "fir": 0.05, ...}
          }

        Notes:
          - If confidence < CONFIDENCE_THRESHOLD (0.45), label_name is set to
            "uncertain" and uncertain=True. The pipeline should request more
            document text or skip risk scoring in this case.
          - Input is capped at 10,000 characters (same as training distribution).
        """
        if not self._ready:
            return {
                "label":      -1,
                "label_name": "unknown",
                "confidence": 0.0,
                "uncertain":  True,
                "all_scores": {},
                "warning":    "Classifier not loaded. Run train_classifier first.",
            }

        if not text or len(text.strip()) < 20:
            return {
                "label":      -1,
                "label_name": "unknown",
                "confidence": 0.0,
                "uncertain":  True,
                "all_scores": {},
                "warning":    "Text too short for classification.",
            }

        try:
            import numpy as np
            X     = self._vectorizer.transform([text[:10_000]])
            proba = self._model.predict_proba(X)[0]
            label = int(np.argmax(proba))
            conf  = float(proba[label])

            all_scores = {
                CATEGORIES[i]: round(float(p), 4)
                for i, p in enumerate(proba)
            }

            uncertain  = conf < CONFIDENCE_THRESHOLD
            label_name = "uncertain" if uncertain else CATEGORIES[label]

            
            result = {
                "label":      label if not uncertain else -1,
                "label_name": label_name,
                "confidence": round(conf, 4),
                "uncertain":  uncertain,
                "all_scores": all_scores,
            }
            if uncertain:
                result["warning"] = (
                    f"Low confidence ({conf:.2f} < {CONFIDENCE_THRESHOLD}). "
                    f"Top candidate: {CATEGORIES[label]}. Consider uploading more text."
                )
            return result

        except Exception as e:
            return {
                "label":      -1,
                "label_name": "unknown",
                "confidence": 0.0,
                "uncertain":  True,
                "all_scores": {},
                "warning":    f"Classification error: {e}",
            }

    def is_ready(self) -> bool:
        return self._ready
    
    def reload(self) -> bool:
        """
        Re-read model and vectorizer from disk.
        Call this after in-process retraining to hot-swap the singleton.
        Returns True if reload succeeded.
        """
        self._model      = None
        self._vectorizer = None
        self._ready      = False
        self._load()
        return self._ready


# ── Singleton ─────────────────────────────────────────────────────────────────
classifier = DocumentClassifier()


# ── CLI entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=150,
                        help="Samples per class (default: 80)")
    parser.add_argument("--generate-only", action="store_true",
                        help="Only generate data, skip training")
    args = parser.parse_args()

    if args.generate_only:
        from models.training_data import generate_dataset
        data = generate_dataset(args.samples)
        out  = Path("data/training/document_classifier_data.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved {len(data)} samples to {out}")
    else:
        metrics = train(samples_per_class=args.samples)
        print(f"\nFinal CV val accuracy : {metrics['cv_val_accuracy']*100:.1f}%")
        print(f"Overfitting gap       : {metrics['overfit_gap']*100:.1f}%")