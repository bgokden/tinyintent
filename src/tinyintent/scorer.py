from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def make_scorer(name: str):
    scorers = {"exemplar": ExemplarScorer, "linear": LinearScorer}
    if name not in scorers:
        raise ValueError(f"unknown classifier: {name} (choose {sorted(scorers)})")
    return scorers[name]()


def load_scorer(name: str, directory: str | Path):
    return {"exemplar": ExemplarScorer, "linear": LinearScorer}[name].load(directory)


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

    name = "exemplar"

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


class LinearScorer:
    """Logistic-regression classifier over the embeddings.

    The strongest head for pure top-1 accuracy: on frozen embeddings it
    clearly beats nearest-exemplar (e.g. CLINC .92 -> .96), because it
    learns a decision boundary rather than trusting the single closest
    example. Use it when the goal is "always decide the best intent".
    Weights are stored as plain arrays, so the artifact stays portable.
    """

    name = "linear"

    def __init__(self, C: float = 10.0, max_iter: int = 1000) -> None:
        self.C = C
        self.max_iter = max_iter
        self.n_labels = 0
        self._model = None

    def fit(self, vectors: np.ndarray, y: np.ndarray, n_labels: int) -> None:
        from sklearn.linear_model import LogisticRegression

        self.n_labels = n_labels
        self._model = LogisticRegression(C=self.C, max_iter=self.max_iter)
        self._model.fit(vectors, y)

    def scores(self, vectors: np.ndarray) -> np.ndarray:
        proba = self._model.predict_proba(vectors)
        out = np.zeros((vectors.shape[0], self.n_labels), dtype=np.float32)
        for col, label in enumerate(self._model.classes_):
            out[:, int(label)] = proba[:, col]
        return out

    def save(self, directory: str | Path) -> None:
        from pathlib import Path as _Path

        directory = _Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.savez(
            directory / "scorer.npz",
            coef=self._model.coef_.astype(np.float32),
            intercept=self._model.intercept_.astype(np.float32),
            classes=self._model.classes_.astype(np.int64),
        )
        (directory / "scorer.json").write_text(
            json.dumps({"n_labels": self.n_labels, "C": self.C, "max_iter": self.max_iter}),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: str | Path) -> "LinearScorer":
        from sklearn.linear_model import LogisticRegression

        directory = Path(directory)
        config = json.loads((directory / "scorer.json").read_text(encoding="utf-8"))
        data = np.load(directory / "scorer.npz")
        scorer = cls(C=config["C"], max_iter=config["max_iter"])
        scorer.n_labels = int(config["n_labels"])
        model = LogisticRegression(C=scorer.C, max_iter=scorer.max_iter)
        model.coef_ = data["coef"].astype(np.float64)
        model.intercept_ = data["intercept"].astype(np.float64)
        model.classes_ = data["classes"].astype(np.int64)
        model.n_features_in_ = model.coef_.shape[1]
        scorer._model = model
        return scorer
