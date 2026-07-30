from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Report:
    """Evaluation summary for an intent model.

    ``accuracy`` and ``coverage`` are over in-scope examples; ``false_fire``
    and ``abstain_recall`` are over out-of-scope examples (only meaningful
    when the eval set contains them).
    """

    n_in_scope: int
    n_oos: int
    accuracy: float          # correct / fired, over in-scope
    coverage: float          # fired / total, over in-scope
    false_fire: float        # fired / total, over OOS
    abstain_recall: float    # abstained / total, over OOS

    def as_dict(self) -> dict:
        return {
            "n_in_scope": self.n_in_scope,
            "n_oos": self.n_oos,
            "accuracy": round(self.accuracy, 4),
            "coverage": round(self.coverage, 4),
            "false_fire": round(self.false_fire, 4),
            "abstain_recall": round(self.abstain_recall, 4),
        }


def score_predictions(
    true_labels: list[str],
    predicted: list[str | None],
    oos_label: str,
) -> Report:
    """Compute the report from aligned true labels and predictions.

    A prediction of ``None`` means the model abstained.
    """

    in_scope_total = correct = fired = 0
    oos_total = oos_fired = 0

    for truth, pred in zip(true_labels, predicted, strict=True):
        if truth == oos_label:
            oos_total += 1
            if pred is not None:
                oos_fired += 1
        else:
            in_scope_total += 1
            if pred is not None:
                fired += 1
                if pred == truth:
                    correct += 1

    return Report(
        n_in_scope=in_scope_total,
        n_oos=oos_total,
        accuracy=correct / fired if fired else 0.0,
        coverage=fired / in_scope_total if in_scope_total else 0.0,
        false_fire=oos_fired / oos_total if oos_total else 0.0,
        abstain_recall=1.0 - (oos_fired / oos_total) if oos_total else 1.0,
    )
