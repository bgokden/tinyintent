from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-8)
    return (matrix / norms).astype(np.float32)


class IdentityTransform:
    """No-op transform: embeddings are used as the encoder produced them."""

    name = "none"

    def fit(self, vectors: np.ndarray, y: np.ndarray) -> None:
        return None

    def apply(self, vectors: np.ndarray) -> np.ndarray:
        return vectors.astype(np.float32)

    def save(self, directory: str | Path) -> None:
        Path(directory).mkdir(parents=True, exist_ok=True)

    @classmethod
    def load(cls, directory: str | Path) -> "IdentityTransform":
        return cls()


class LdaTransform:
    """Linear discriminant projection learned on the frozen embeddings.

    A portable metric-learning step: fit once on the training embeddings, it
    projects onto the directions that best separate the intents (shrinkage-
    regularized for few-shot stability), then re-normalizes so cosine still
    applies. The learned model is just a mean vector and a matrix, so the
    encoder stays frozen and the artifact stays tiny.
    """

    name = "lda"

    def __init__(self) -> None:
        self.mean: np.ndarray | None = None
        self.matrix: np.ndarray | None = None

    def fit(self, vectors: np.ndarray, y: np.ndarray) -> None:
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        self.mean = vectors.mean(axis=0).astype(np.float32)
        lda = LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto")
        lda.fit(vectors, y)
        self.matrix = np.asarray(lda.scalings_, dtype=np.float32)

    def apply(self, vectors: np.ndarray) -> np.ndarray:
        projected = (vectors.astype(np.float32) - self.mean) @ self.matrix
        return _l2_normalize(projected)

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.savez(directory / "lda.npz", mean=self.mean, matrix=self.matrix)

    @classmethod
    def load(cls, directory: str | Path) -> "LdaTransform":
        data = np.load(Path(directory) / "lda.npz")
        transform = cls()
        transform.mean = data["mean"].astype(np.float32)
        transform.matrix = data["matrix"].astype(np.float32)
        return transform


def make_transform(name: str):
    transforms = {"none": IdentityTransform, "lda": LdaTransform}
    if name not in transforms:
        raise ValueError(f"unknown transform: {name} (choose {sorted(transforms)})")
    return transforms[name]()


def load_transform(name: str, directory: str | Path):
    return {"none": IdentityTransform, "lda": LdaTransform}[name].load(directory)
