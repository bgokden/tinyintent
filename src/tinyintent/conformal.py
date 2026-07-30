from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _lac_quantile(nonconformity: np.ndarray, alpha: float) -> float:
    n = len(nonconformity)
    if n == 0:
        return 1.0
    level = float(np.clip(np.ceil((n + 1) * (1.0 - alpha)) / n, 0.0, 1.0))
    return float(np.quantile(nonconformity, level, method="higher"))


@dataclass
class Conformal:
    """Split-conformal prediction sets (LAC), class-conditional by default.

    The score of a class is the exemplar cosine similarity from
    :class:`ExemplarScorer` (absolute, not softmax). Nonconformity of a
    labelled example is ``1 - score(true class)``. A single global quantile
    (plain LAC) gives marginal coverage but tends to over-include tightly
    clustered classes in unrelated queries' sets, inflating ambiguity.
    Calibrating a separate threshold per class (Mondrian conformal) lets
    each intent set its own bar, which shrinks sets while preserving
    coverage. Classes with too few calibration examples fall back to the
    global threshold.

    The prediction set for a new input is every class scoring at least its
    threshold ``1 - q_class``; set size drives the decision (0 abstain,
    1 fire, 2+ ambiguous), and the absolute cutoff means out-of-scope input
    that is similar to nothing yields an empty set.
    """

    name = "lac"

    alpha: float
    q: float                       # global fallback
    q_by_class: dict[int, float]
    floor: float = -1.0            # absolute similarity floor from OOS negatives

    @classmethod
    def calibrate(
        cls,
        scores: np.ndarray,
        y: np.ndarray,
        alpha: float = 0.1,
        mondrian: bool = False,
        min_per_class: int = 50,
        oos_scores: np.ndarray | None = None,
        oos_reject: float = 0.8,
    ) -> "Conformal":
        """Calibrate the conformal threshold(s).

        A single global threshold (``mondrian=False``, the default) is the
        robust choice at few-shot scale. Class-conditional thresholds
        (``mondrian=True``) can tighten sets, but only with enough
        calibration examples per class; below ``min_per_class`` a class
        falls back to the global threshold, since the per-class
        high-confidence quantile is unstable on little data.

        If ``oos_scores`` (per-class similarities for known out-of-scope
        examples) are given, an absolute ``floor`` is raised to reject about
        ``oos_reject`` of them. It trades a little in-scope coverage for
        fewer false fires; leave it unset to keep the pure conformal
        guarantee.
        """

        rows = np.arange(len(y))
        nonconformity = 1.0 - scores[rows, y]
        q_global = _lac_quantile(nonconformity, alpha)

        q_by_class: dict[int, float] = {}
        if mondrian:
            for label in range(scores.shape[1]):
                class_scores = nonconformity[y == label]
                if len(class_scores) >= min_per_class:
                    q_by_class[label] = _lac_quantile(class_scores, alpha)
                else:
                    q_by_class[label] = q_global

        floor = -1.0
        if oos_scores is not None and len(oos_scores):
            oos_top = oos_scores.max(axis=1)
            floor = float(np.quantile(oos_top, oos_reject))

        return cls(alpha=alpha, q=q_global, q_by_class=q_by_class, floor=floor)

    def _thresholds(self, n_labels: int) -> np.ndarray:
        base = np.array(
            [1.0 - self.q_by_class.get(c, self.q) for c in range(n_labels)],
            dtype=np.float32,
        )
        return np.maximum(base, self.floor)

    def prediction_set(self, scores: np.ndarray) -> np.ndarray:
        """Boolean mask [n, C] of class membership in each prediction set."""

        return scores >= self._thresholds(scores.shape[1])[None, :]

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "conformal.json").write_text(
            json.dumps(
                {
                    "alpha": self.alpha,
                    "q": self.q,
                    "q_by_class": {str(k): v for k, v in self.q_by_class.items()},
                    "floor": self.floor,
                }
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: str | Path) -> "Conformal":
        config = json.loads(
            (Path(directory) / "conformal.json").read_text(encoding="utf-8")
        )
        return cls(
            alpha=config["alpha"],
            q=config["q"],
            q_by_class={int(k): v for k, v in config["q_by_class"].items()},
            floor=config.get("floor", -1.0),
        )
