from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Policy:
    threshold: float
    margin: float


def _decide(scores: np.ndarray, threshold: float, margin: float):
    """Return top label index and whether the model fires, per row."""

    order = np.argsort(scores, axis=1)[:, ::-1]
    top = order[:, 0]
    s1 = np.take_along_axis(scores, top[:, None], axis=1)[:, 0]
    if scores.shape[1] > 1:
        s2 = np.take_along_axis(scores, order[:, 1][:, None], axis=1)[:, 0]
    else:
        s2 = np.full_like(s1, -np.inf)
    fire = (s1 >= threshold) & ((s1 - s2) >= margin)
    return top, fire


def calibrate(
    scores: np.ndarray,
    y_true: np.ndarray,
    max_false_fire: float = 0.02,
    margins: tuple[float, ...] = (0.0, 0.02, 0.05, 0.08, 0.12),
) -> Policy:
    """Pick a threshold and margin from validation scores.

    ``y_true`` holds the gold label index per row, or ``-1`` for
    out-of-scope rows. The policy maximizes the number of correct in-scope
    fires while keeping the out-of-scope false-fire rate within
    ``max_false_fire``. With no out-of-scope rows, wrong in-scope fires are
    bounded instead, so the threshold still learns to abstain when unsure.
    """

    top_all = scores.max(axis=1)
    candidates = np.unique(np.round(top_all, 4))
    # Also allow a "fire on everything" threshold.
    thresholds = np.concatenate([[float(candidates.min()) - 1e-3], candidates])

    is_oos = y_true < 0
    n_oos = int(is_oos.sum())

    best: Policy | None = None
    best_key = (-1, 1 << 30, -1.0)  # (correct_fires, wrong_fires, threshold)

    for threshold in thresholds:
        for margin in margins:
            top, fire = _decide(scores, float(threshold), float(margin))

            in_fire = fire & ~is_oos
            correct = int(((top == y_true) & in_fire).sum())
            wrong = int(((top != y_true) & in_fire).sum())

            if n_oos:
                oos_fire = int((fire & is_oos).sum())
                if oos_fire / n_oos > max_false_fire:
                    continue
            else:
                total_in = int((~is_oos).sum())
                if total_in and wrong / total_in > max_false_fire:
                    continue

            key = (correct, -wrong, float(threshold))
            if (key[0], -key[1], key[2]) > (best_key[0], best_key[1], best_key[2]):
                best_key = (correct, wrong, float(threshold))
                best = Policy(threshold=float(threshold), margin=float(margin))

    # Fallback: if nothing satisfied the constraint, be conservative.
    if best is None:
        best = Policy(threshold=float(top_all.max()) + 1.0, margin=0.0)
    return best
