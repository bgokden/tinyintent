from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exps = np.exp(shifted)
    return exps / exps.sum(axis=1, keepdims=True)


class LinearScorer:
    """Logistic-regression head over the frozen embeddings.

    The single classifier head: it learns a decision boundary rather than
    trusting the nearest example, which is what wins for top-1 accuracy.

    scikit-learn fits it, but nothing more: the fitted weights are copied out
    into plain arrays and every prediction is computed from those. Loading a
    saved model therefore does not reconstruct an estimator by assigning its
    private attributes (``coef_``, ``classes_``, ``n_features_in_``), which is
    not a supported interface and can break silently across releases. The
    artifact is the weights, and the weights are all inference needs.
    """

    def __init__(self, C: float = 10.0, max_iter: int = 1000) -> None:
        self.C = C
        self.max_iter = max_iter
        self.n_labels = 0
        self.coef: np.ndarray | None = None
        self.intercept: np.ndarray | None = None
        self.classes: np.ndarray | None = None

    def fit(self, vectors: np.ndarray, y: np.ndarray, n_labels: int) -> None:
        from sklearn.linear_model import LogisticRegression

        self.n_labels = n_labels
        model = LogisticRegression(C=self.C, max_iter=self.max_iter)
        model.fit(vectors, y)
        self.coef = model.coef_.astype(np.float64)
        self.intercept = model.intercept_.astype(np.float64)
        self.classes = model.classes_.astype(np.int64)

    def _probabilities(self, vectors: np.ndarray) -> np.ndarray:
        """Reproduce ``LogisticRegression.predict_proba`` from the weights.

        Two shapes, matching scikit-learn: one row of coefficients means a
        binary fit scored through a sigmoid, more than one means a multinomial
        fit scored through a softmax (the lbfgs default for 3+ classes).
        """

        logits = vectors @ self.coef.T + self.intercept
        if logits.shape[1] == 1:
            positive = 1.0 / (1.0 + np.exp(-logits[:, 0]))
            return np.column_stack([1.0 - positive, positive])
        return _softmax(logits)

    def scores(self, vectors: np.ndarray) -> np.ndarray:
        proba = self._probabilities(np.asarray(vectors, dtype=np.float64))
        out = np.zeros((vectors.shape[0], self.n_labels), dtype=np.float32)
        for col, label in enumerate(self.classes):
            out[:, int(label)] = proba[:, col]
        return out

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.savez(
            directory / "scorer.npz",
            coef=self.coef.astype(np.float32),
            intercept=self.intercept.astype(np.float32),
            classes=self.classes.astype(np.int64),
        )
        (directory / "scorer.json").write_text(
            json.dumps({"n_labels": self.n_labels, "C": self.C, "max_iter": self.max_iter}),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: str | Path) -> "LinearScorer":
        directory = Path(directory)
        config = json.loads((directory / "scorer.json").read_text(encoding="utf-8"))
        data = np.load(directory / "scorer.npz")
        scorer = cls(C=config["C"], max_iter=config["max_iter"])
        scorer.n_labels = int(config["n_labels"])
        scorer.coef = data["coef"].astype(np.float64)
        scorer.intercept = data["intercept"].astype(np.float64)
        scorer.classes = data["classes"].astype(np.int64)
        return scorer
