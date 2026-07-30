from __future__ import annotations

import hashlib
from typing import Protocol

import numpy as np


# bge-small beats MiniLM on the intent benchmarks (higher coverage and fire
# accuracy, lower near-OOS false-fire) while staying small and frozen.
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


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
    """Frozen sentence-transformers encoder (default: MiniLM)."""

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


class StaticEncoder:
    """Static (distilled) embeddings via Model2Vec: numpy-only, no torch.

    A real portability tier: a lookup-table encoder that runs in
    milliseconds on CPU with a tiny footprint, at some cost to accuracy on
    phrasing/negation-sensitive inputs. Install the extra with
    ``uv sync --extra static``.
    """

    DEFAULT = "minishlab/potion-base-8M"

    def __init__(self, model_name: str = DEFAULT):
        from model2vec import StaticModel

        self.model_name = model_name
        self._model = StaticModel.from_pretrained(model_name)
        self.dim = int(self._model.encode(["x"]).shape[1])

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = np.asarray(self._model.encode(list(texts)), dtype=np.float32)
        return _l2_normalize(vectors)

    def spec(self) -> dict:
        return {"kind": "static", "model": self.model_name}


class HashingEncoder:
    """Deterministic, dependency-free bag-of-words hashing encoder.

    Not semantically strong, but offline and instant. It exists so the
    framework and tests can run without downloading a model; it also
    doubles as a portability floor for trivially simple, keyword-separable
    intents.
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
    """Rebuild an encoder from its spec dict."""

    kind = spec["kind"]
    if kind == "sentence-transformers":
        return SentenceEncoder(spec.get("model", DEFAULT_MODEL))
    if kind == "static":
        return StaticEncoder(spec.get("model", StaticEncoder.DEFAULT))
    if kind == "hashing":
        return HashingEncoder(int(spec.get("dim", 256)))
    raise ValueError(f"unknown encoder kind: {kind}")
