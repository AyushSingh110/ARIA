from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


class EmbeddingEngine:
    """Thin wrapper around SentenceTransformers for ARIA's embedding needs.

    Loaded once per process via get_engine(); subsequent calls hit the cache.
    All vectors are L2-normalised so cosine similarity == dot product.
    """

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer
        self._model: SentenceTransformer = SentenceTransformer(model_name)

    def embed(self, text: str) -> list[float]:
        vec = self._model.encode(text, normalize_embeddings=True)
        return vec.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        vecs = self._model.encode(texts, normalize_embeddings=True)
        return vecs.tolist()

    @staticmethod
    def cosine_distance(a: list[float], b: list[float]) -> float:
        """Return cosine distance (0 = identical, 1 = orthogonal, 2 = opposite)."""
        va = np.array(a, dtype=np.float32)
        vb = np.array(b, dtype=np.float32)
        similarity = float(np.dot(va, vb))   # already normalised
        return 1.0 - similarity

    @staticmethod
    def cosine_similarity(a: list[float], b: list[float]) -> float:
        va = np.array(a, dtype=np.float32)
        vb = np.array(b, dtype=np.float32)
        return float(np.dot(va, vb))


@lru_cache(maxsize=1)
def get_engine(model_name: str = "BAAI/bge-small-en-v1.5") -> EmbeddingEngine:
    return EmbeddingEngine(model_name)
