from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class ExemplarScorer:
    """The single scoring method: per-class maximum cosine similarity.

    Each class is represented by its example vectors (not a single
    centroid), so multi-modal intents stay intact. A query's score for a
    class is the largest cosine similarity to any of that class's
    exemplars. These are *absolute* similarities in [-1, 1], deliberately
    not softmax-normalized: an input far from every class scores low
    everywhere, which is the signal the conformal layer uses to abstain on
    out-of-scope input. A relative softmax would hide that by always
    naming a most-similar class.
    """

    def __init__(self) -> None:
        self.vectors: np.ndarray | None = None       # [E, D] unit rows
        self.exemplar_label: np.ndarray | None = None  # [E]
        self.n_labels = 0

    def fit(self, vectors: np.ndarray, y: np.ndarray, n_labels: int) -> None:
        self.vectors = vectors.astype(np.float32)
        self.exemplar_label = y.astype(np.int64)
        self.n_labels = n_labels

    def scores(self, vectors: np.ndarray) -> np.ndarray:
        """Per-class max cosine similarity, shape [n, n_labels], in [-1, 1]."""

        sims = vectors.astype(np.float32) @ self.vectors.T   # [n, E]
        out = np.full((vectors.shape[0], self.n_labels), -1.0, dtype=np.float32)
        for label in range(self.n_labels):
            mask = self.exemplar_label == label
            if mask.any():
                out[:, label] = sims[:, mask].max(axis=1)
        return out

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.savez(
            directory / "scorer.npz",
            vectors=self.vectors,
            exemplar_label=self.exemplar_label,
        )
        (directory / "scorer.json").write_text(
            json.dumps({"n_labels": self.n_labels}), encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: str | Path) -> "ExemplarScorer":
        directory = Path(directory)
        config = json.loads((directory / "scorer.json").read_text(encoding="utf-8"))
        data = np.load(directory / "scorer.npz")
        scorer = cls()
        scorer.vectors = data["vectors"].astype(np.float32)
        scorer.exemplar_label = data["exemplar_label"].astype(np.int64)
        scorer.n_labels = int(config["n_labels"])
        return scorer
