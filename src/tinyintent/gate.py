from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class DecisiveGate:
    """Fire the top intent, or reject — no ambiguous bucket, no escalation.

    For a system with no LLM fallback every input must reach a terminal
    decision. This policy makes exactly two: if the best intent similarity
    clears ``gate_tau`` it fires that single intent, otherwise it rejects
    the input as out-of-scope. There is no set-valued "ambiguous" outcome,
    so nothing is deferred to a downstream model.

    ``reject_level`` is the fraction of in-scope inputs sacrificed to the
    reject gate (its similarity quantile sets the bar); OOS negatives, when
    given, can raise the bar further. Lower it to fire on almost everything
    (fewer rejects, more misroutes), raise it to reject more aggressively.
    """

    name = "gate"

    gate_tau: float
    reject_level: float

    @classmethod
    def calibrate(
        cls,
        scores: np.ndarray,
        y: np.ndarray,
        reject_level: float = 0.1,
        oos_scores: np.ndarray | None = None,
        oos_reject: float = 0.9,
    ) -> "DecisiveGate":
        in_scope_top = scores.max(axis=1)
        gate_tau = float(np.quantile(in_scope_top, reject_level, method="lower"))
        if oos_scores is not None and len(oos_scores):
            gate_tau = max(
                gate_tau, float(np.quantile(oos_scores.max(axis=1), oos_reject))
            )
        return cls(gate_tau=gate_tau, reject_level=reject_level)

    def prediction_set(self, scores: np.ndarray) -> np.ndarray:
        n, n_labels = scores.shape
        mask = np.zeros((n, n_labels), dtype=bool)
        top = scores.argmax(axis=1)
        fired = scores.max(axis=1) >= self.gate_tau
        rows = np.arange(n)[fired]
        mask[rows, top[fired]] = True
        return mask

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "gate.json").write_text(
            json.dumps({"gate_tau": self.gate_tau, "reject_level": self.reject_level}),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: str | Path) -> "DecisiveGate":
        config = json.loads((Path(directory) / "gate.json").read_text(encoding="utf-8"))
        return cls(gate_tau=config["gate_tau"], reject_level=config["reject_level"])
