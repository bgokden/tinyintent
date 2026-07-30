from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression


class LogRegHead:
    """Multinomial logistic regression over frozen embeddings.

    A tiny linear head: fast to train, calibratable probabilities, and a
    strong baseline whenever there are more than a handful of examples per
    label. Weights are saved as plain arrays (no pickle) so the artifact
    stays portable.
    """

    name = "logreg"

    def __init__(self, C: float = 10.0, max_iter: int = 1000) -> None:
        self.C = C
        self.max_iter = max_iter
        self.n_labels: int = 0
        self._model: LogisticRegression | None = None

    def fit(self, vectors: np.ndarray, y: np.ndarray, n_labels: int) -> None:
        self.n_labels = n_labels
        self._model = LogisticRegression(C=self.C, max_iter=self.max_iter)
        self._model.fit(vectors, y)

    def scores(self, vectors: np.ndarray) -> np.ndarray:
        proba = self._model.predict_proba(vectors)
        out = np.zeros((vectors.shape[0], self.n_labels), dtype=np.float32)
        # Map sklearn's class order back onto the full label index space.
        for col, label in enumerate(self._model.classes_):
            out[:, int(label)] = proba[:, col]
        return out

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.savez(
            directory / "logreg.npz",
            coef=self._model.coef_.astype(np.float32),
            intercept=self._model.intercept_.astype(np.float32),
            classes=self._model.classes_.astype(np.int64),
        )
        (directory / "logreg.json").write_text(
            json.dumps({"n_labels": self.n_labels, "C": self.C, "max_iter": self.max_iter}),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: str | Path) -> "LogRegHead":
        directory = Path(directory)
        config = json.loads((directory / "logreg.json").read_text(encoding="utf-8"))
        data = np.load(directory / "logreg.npz")

        head = cls(C=config["C"], max_iter=config["max_iter"])
        head.n_labels = int(config["n_labels"])

        model = LogisticRegression(C=head.C, max_iter=head.max_iter)
        model.coef_ = data["coef"].astype(np.float64)
        model.intercept_ = data["intercept"].astype(np.float64)
        model.classes_ = data["classes"].astype(np.int64)
        # A binary problem stores a single weight row; keep sklearn happy.
        model.n_features_in_ = model.coef_.shape[1]
        head._model = model
        return head
