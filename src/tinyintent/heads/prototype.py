from __future__ import annotations

from pathlib import Path

import numpy as np


class PrototypeHead:
    """Nearest-example head: score a label by its best-matching example.

    No training beyond storing the (unit) example vectors. A label's score
    for a query is the maximum cosine similarity to any of that label's
    examples, which keeps multi-modal intents intact (unlike averaging
    every example into a single centroid).
    """

    name = "prototype"

    def __init__(self) -> None:
        self.vectors: np.ndarray | None = None
        self.example_label: np.ndarray | None = None
        self.n_labels: int = 0

    def fit(self, vectors: np.ndarray, y: np.ndarray, n_labels: int) -> None:
        self.vectors = vectors.astype(np.float32)
        self.example_label = y.astype(np.int64)
        self.n_labels = n_labels

    def scores(self, vectors: np.ndarray) -> np.ndarray:
        sims = vectors.astype(np.float32) @ self.vectors.T  # [n, E]
        out = np.full((vectors.shape[0], self.n_labels), -1.0, dtype=np.float32)
        for label in range(self.n_labels):
            mask = self.example_label == label
            if mask.any():
                out[:, label] = sims[:, mask].max(axis=1)
        return out

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.savez(
            directory / "prototype.npz",
            vectors=self.vectors,
            example_label=self.example_label,
            n_labels=np.array([self.n_labels]),
        )

    @classmethod
    def load(cls, directory: str | Path) -> "PrototypeHead":
        data = np.load(Path(directory) / "prototype.npz")
        head = cls()
        head.vectors = data["vectors"].astype(np.float32)
        head.example_label = data["example_label"].astype(np.int64)
        head.n_labels = int(data["n_labels"][0])
        return head
