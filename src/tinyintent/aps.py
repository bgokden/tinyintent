from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _lac_level(n: int, alpha: float) -> float:
    return float(np.clip(np.ceil((n + 1) * (1.0 - alpha)) / n, 0.0, 1.0))


def _fit_temperature(scores: np.ndarray, y: np.ndarray) -> float:
    rows = np.arange(len(y))

    def nll(temp: float) -> float:
        probs = _softmax(scores / temp)
        return float(-np.log(np.clip(probs[rows, y], 1e-9, 1.0)).mean())

    grid = np.geomspace(0.02, 5.0, 40)
    best = min(grid, key=nll)
    fine = np.linspace(best * 0.5, best * 1.5, 21)
    return float(min(fine, key=nll))


@dataclass
class Aps:
    """Two-stage selective prediction: abstain gate + APS set.

    Stage 1 asks "is this in scope at all?" with an *absolute* similarity
    gate: if the best class similarity is below ``gate_tau`` the input is
    rejected (empty set), which is how out-of-scope is handled — before any
    softmax, so the relative distribution can never mask a far input.

    Stage 2 asks "which intent(s)?" with Adaptive Prediction Sets (APS,
    Romano et al. 2020) over temperature-scaled probabilities: classes are
    added in descending probability until the cumulative mass reaches the
    calibrated ``q_aps``. APS adapts the set size per query -- confident
    inputs give a singleton, genuinely uncertain ones grow -- which cuts the
    ambiguity that a single absolute LAC threshold produces on close
    intents, while keeping the ``1 - alpha`` coverage guarantee.
    """

    name = "aps"

    alpha: float
    temperature: float
    q_aps: float
    gate_tau: float

    @classmethod
    def calibrate(
        cls,
        scores: np.ndarray,
        y: np.ndarray,
        alpha: float = 0.1,
        gate_risk: float = 0.05,
        oos_scores: np.ndarray | None = None,
        oos_reject: float = 0.8,
    ) -> "Aps":
        temperature = _fit_temperature(scores, y)
        probs = _softmax(scores / temperature)
        rows = np.arange(len(y))

        # APS nonconformity: cumulative mass down to and including the true
        # class in descending-probability order (non-randomized).
        order = np.argsort(probs, axis=1)[:, ::-1]
        sorted_p = np.take_along_axis(probs, order, axis=1)
        cum = np.cumsum(sorted_p, axis=1)
        true_pos = (order == y[:, None]).argmax(axis=1)
        aps_scores = cum[rows, true_pos]
        q_aps = float(np.quantile(aps_scores, _lac_level(len(y), alpha), method="higher"))

        # Abstain gate: keep >= 1 - gate_risk of in-scope inputs; raise it to
        # reject known out-of-scope negatives when provided.
        s_max = scores.max(axis=1)
        gate_tau = float(np.quantile(s_max, gate_risk, method="lower"))
        if oos_scores is not None and len(oos_scores):
            gate_tau = max(gate_tau, float(np.quantile(oos_scores.max(axis=1), oos_reject)))

        return cls(alpha=alpha, temperature=temperature, q_aps=q_aps, gate_tau=gate_tau)

    def prediction_set(self, scores: np.ndarray) -> np.ndarray:
        probs = _softmax(scores / self.temperature)
        n, n_labels = probs.shape

        order = np.argsort(probs, axis=1)[:, ::-1]
        cum = np.cumsum(np.take_along_axis(probs, order, axis=1), axis=1)
        reached = cum >= self.q_aps
        last = np.where(reached.any(axis=1), reached.argmax(axis=1), n_labels - 1)

        mask = np.zeros((n, n_labels), dtype=bool)
        for i in range(n):
            mask[i, order[i, : last[i] + 1]] = True

        gated = scores.max(axis=1) < self.gate_tau
        mask[gated, :] = False
        return mask

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "aps.json").write_text(
            json.dumps(
                {
                    "alpha": self.alpha,
                    "temperature": self.temperature,
                    "q_aps": self.q_aps,
                    "gate_tau": self.gate_tau,
                }
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: str | Path) -> "Aps":
        config = json.loads((Path(directory) / "aps.json").read_text(encoding="utf-8"))
        return cls(
            alpha=config["alpha"],
            temperature=config["temperature"],
            q_aps=config["q_aps"],
            gate_tau=config["gate_tau"],
        )
