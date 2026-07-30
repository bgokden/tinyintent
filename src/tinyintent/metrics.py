from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tinyintent.model import Prediction


@dataclass
class Report:
    """Evaluation summary for a selective intent classifier.

    In-scope metrics describe behaviour on genuine intents; out-of-scope
    metrics describe restraint on inputs that belong to no intent.
    ``coverage`` is the conformal target: the true intent should land in
    the prediction set at least ``1 - risk`` of the time.
    """

    n_in_scope: int
    n_oos: int
    coverage: float          # true label in the set, over in-scope
    fire_accuracy: float     # correct, among single-intent fires
    fire_rate: float         # decision == fire, over in-scope
    ambiguous_rate: float    # decision == ambiguous, over in-scope
    abstain_rate: float      # decision == abstain, over in-scope
    oos_false_fire: float    # decision == fire, over OOS
    oos_abstain: float       # decision == abstain, over OOS

    def as_dict(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


def score_predictions(
    true_labels: list[str],
    predictions: "list[Prediction]",
    label_names: list[str],
    oos_label: str,
) -> Report:
    in_total = fired = correct_fire = ambiguous = abstained = covered = 0
    oos_total = oos_fired = oos_abstained = 0

    for truth, pred in zip(true_labels, predictions, strict=True):
        set_labels = {label for label, _ in pred.set_}
        if truth == oos_label:
            oos_total += 1
            if pred.decision == "fire":
                oos_fired += 1
            elif pred.decision == "abstain":
                oos_abstained += 1
        else:
            in_total += 1
            if truth in set_labels:
                covered += 1
            if pred.decision == "fire":
                fired += 1
                if pred.intent == truth:
                    correct_fire += 1
            elif pred.decision == "ambiguous":
                ambiguous += 1
            else:
                abstained += 1

    def frac(num: int, den: int) -> float:
        return num / den if den else 0.0

    return Report(
        n_in_scope=in_total,
        n_oos=oos_total,
        coverage=frac(covered, in_total),
        fire_accuracy=frac(correct_fire, fired),
        fire_rate=frac(fired, in_total),
        ambiguous_rate=frac(ambiguous, in_total),
        abstain_rate=frac(abstained, in_total),
        oos_false_fire=frac(oos_fired, oos_total),
        oos_abstain=frac(oos_abstained, oos_total),
    )
