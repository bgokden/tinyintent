from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Report:
    """Evaluation summary: top-1 accuracy over in-scope examples."""

    n_in_scope: int
    accuracy: float

    def as_dict(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


def score_predictions(true_labels: list[str], predictions: list[str]) -> Report:
    total = len(true_labels)
    correct = sum(t == p for t, p in zip(true_labels, predictions, strict=True))
    return Report(n_in_scope=total, accuracy=correct / total if total else 0.0)
