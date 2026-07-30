from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Conformal:
    """Split-conformal calibration for set-valued prediction (LAC).

    The score of a class is the exemplar cosine similarity from
    :class:`ExemplarScorer` (absolute, not softmax). On a held-out set at
    risk level ``alpha`` the nonconformity of a labelled example is
    ``1 - score(true class)``, and ``q`` is the finite-sample-corrected
    ``(1 - alpha)`` quantile. The prediction set for a new input is every
    class scoring at least ``1 - q``. On exchangeable in-scope data the
    true class lands in the set with probability at least ``1 - alpha``.

    Because the threshold ``1 - q`` is an absolute similarity cutoff, set
    size drives the decision cleanly: 0 means abstain (nothing is similar
    enough, including out-of-scope input), 1 means fire, 2 or more means
    ambiguous.
    """

    alpha: float
    q: float

    @classmethod
    def calibrate(cls, scores: np.ndarray, y: np.ndarray, alpha: float = 0.1) -> "Conformal":
        rows = np.arange(len(y))
        nonconformity = 1.0 - scores[rows, y]

        n = len(y)
        level = np.ceil((n + 1) * (1.0 - alpha)) / n
        level = float(np.clip(level, 0.0, 1.0))
        q = float(np.quantile(nonconformity, level, method="higher"))
        return cls(alpha=alpha, q=q)

    def prediction_set(self, scores: np.ndarray) -> np.ndarray:
        """Boolean mask [n, C] of class membership in each prediction set."""

        return scores >= (1.0 - self.q)

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "conformal.json").write_text(
            json.dumps({"alpha": self.alpha, "q": self.q}), encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: str | Path) -> "Conformal":
        config = json.loads(
            (Path(directory) / "conformal.json").read_text(encoding="utf-8")
        )
        return cls(alpha=config["alpha"], q=config["q"])
