from __future__ import annotations

import hashlib
from typing import Protocol

import numpy as np


# bge-large is the base: best top-1 accuracy in the encoder sweep, and it
# fine-tunes reliably for the reranker's cross-encoder pairs.
DEFAULT_MODEL = "BAAI/bge-large-en-v1.5"


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    return (matrix / norms).astype(np.float32)


class Encoder(Protocol):
    """Turns text into unit-length embedding rows."""

    dim: int

    def encode(self, texts: list[str]) -> np.ndarray: ...

    def spec(self) -> dict: ...


class SentenceEncoder:
    """Frozen sentence-transformers encoder (bge-large)."""

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._model = SentenceTransformer(model_name, device=device)
        get_dim = getattr(
            self._model, "get_embedding_dimension", None
        ) or self._model.get_sentence_embedding_dimension
        self.dim = int(get_dim())

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(
            texts,
            batch_size=128,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.astype(np.float32)

    def spec(self) -> dict:
        return {"kind": "sentence-transformers", "model": self.model_name}


class HashingEncoder:
    """Deterministic, dependency-free bag-of-words hashing encoder.

    Not semantically strong; it exists so the framework and tests can run
    offline without downloading a model.
    """

    def __init__(self, dim: int = 256):
        self.dim = dim

    def encode(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in text.lower().split():
                digest = hashlib.md5(token.encode("utf-8")).digest()
                index = int.from_bytes(digest[:4], "big") % self.dim
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                matrix[row, index] += sign
        return _l2_normalize(matrix)

    def spec(self) -> dict:
        return {"kind": "hashing", "dim": self.dim}


def make_encoder(spec: dict) -> Encoder:
    """Rebuild an encoder from its saved spec."""

    kind = spec["kind"]
    if kind == "sentence-transformers":
        return SentenceEncoder(spec.get("model", DEFAULT_MODEL))
    if kind == "hashing":
        return HashingEncoder(int(spec.get("dim", 256)))
    raise ValueError(f"unknown encoder kind: {kind}")
