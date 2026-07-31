from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Report:
    """Evaluation summary: top-1 accuracy plus F1 over the routed intents."""

    n: int
    accuracy: float
    macro_f1: float
    weighted_f1: float
    per_class: dict[str, dict] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            "accuracy": round(self.accuracy, 4),
            "macro_f1": round(self.macro_f1, 4),
            "weighted_f1": round(self.weighted_f1, 4),
        }

    def table(self) -> str:
        """A per-intent precision/recall/F1 breakdown, one row per intent."""

        head = f"{'intent':<20}{'precision':>10}{'recall':>9}{'f1':>7}{'support':>9}"
        rows = [head, "-" * len(head)]
        for label in sorted(self.per_class):
            m = self.per_class[label]
            rows.append(f"{label:<20}{m['precision']:>10.3f}{m['recall']:>9.3f}"
                        f"{m['f1']:>7.3f}{m['support']:>9d}")
        return "\n".join(rows)


def score_predictions(true_labels: list[str], predictions: list[str]) -> Report:
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_recall_fscore_support,
    )

    if not true_labels:
        return Report(0, 0.0, 0.0, 0.0, {})

    labels = sorted(set(true_labels) | set(predictions))
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels, predictions, labels=labels, zero_division=0
    )
    per_class = {
        label: {"precision": float(p), "recall": float(r), "f1": float(f), "support": int(s)}
        for label, p, r, f, s in zip(labels, precision, recall, f1, support)
        if s > 0                       # only intents actually present in the truth
    }
    return Report(
        n=len(true_labels),
        accuracy=float(accuracy_score(true_labels, predictions)),
        macro_f1=float(f1_score(true_labels, predictions, average="macro", zero_division=0)),
        weighted_f1=float(f1_score(true_labels, predictions, average="weighted", zero_division=0)),
        per_class=per_class,
    )
