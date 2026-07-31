from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class LinearScorer:
    """Logistic-regression head over the frozen embeddings.

    The single classifier head: it learns a decision boundary rather than
    trusting the nearest example, which is what wins for top-1 accuracy.
    Weights are stored as plain arrays, so the artifact stays portable.
    """

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
        directory = Path(directory)
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
