"""
LexShield Embedder
==================
Converts text into 384-dimensional vectors using all-MiniLM-L6-v2.

Why this model:
  - Runs entirely on CPU (no GPU needed)
  - Model size: ~90MB
  - Speed: ~2,000 chunks in 30-40 min on i5 CPU
  - Vector size: 384 floats — small enough for fast ChromaDB search
  - Good semantic understanding of legal English

Usage:
  from rag.embedder import embedder
  vectors = embedder.embed(["section 420 cheating", "tenant deposit refund"])
"""
import os
# --- HARDWARE SAFETY LIMITS ---
# Force the AI to use only 2 CPU threads to prevent laptop overheating
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
import torch
torch.set_num_threads(2)
# ------------------------------

from sentence_transformers import SentenceTransformer
from typing import Union

MODEL_NAME = "all-MiniLM-L6-v2"


class LegalEmbedder:
    """
    Wrapper around SentenceTransformer.
    Loads the model once and reuses it for all embedding calls.
    """

    def __init__(self, model_name: str = MODEL_NAME):
        print(f"Loading embedding model: {model_name}")
        print("(First run downloads ~90MB — subsequent runs load from cache)")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.vector_dim = self.model.get_sentence_embedding_dimension()
        print(f"Model loaded. Vector dimension: {self.vector_dim}")

    def embed(
        self,
        texts: Union[str, list[str]],
        batch_size: int = 8,
        show_progress: bool = False,
    ) -> list[list[float]]:
        """
        Embed one or more texts into vectors.

        Args:
            texts       : single string or list of strings
            batch_size  : how many texts to process at once.
                          32 is safe for 8GB RAM. Lower to 16 if you see
                          memory warnings.
            show_progress: print a progress bar (useful for large batches)

        Returns:
            List of vectors, each a list of 384 floats.
        """
        if isinstance(texts, str):
            texts = [texts]

        vectors = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,   # cosine similarity works best normalised
        )

        # Convert numpy array → plain Python list (required by ChromaDB)
        return vectors.tolist()

    def embed_single(self, text: str) -> list[float]:
        """Convenience method for embedding one query at query time."""
        return self.embed([text])[0]


# ── Module-level singleton ────────────────────────────────────────────────────
# Import this directly:  from rag.embedder import embedder
# The model loads once when this module is first imported.
embedder = LegalEmbedder()