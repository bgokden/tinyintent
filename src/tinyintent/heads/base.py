from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np


class Head(Protocol):
    """A classifier head over frozen embeddings.

    ``scores`` returns a per-label score matrix where higher is better and
    values are comparable across labels for the same head, so the model's
    abstention threshold and top1-top2 margin work uniformly. The scale
    differs by head (cosine for prototype, probability for logreg/mlp),
    which is exactly what calibration exists to absorb.
    """

    name: str

    def fit(self, vectors: np.ndarray, y: np.ndarray, n_labels: int) -> None: ...

    def scores(self, vectors: np.ndarray) -> np.ndarray: ...

    def save(self, directory: str | Path) -> None: ...

    @classmethod
    def load(cls, directory: str | Path) -> "Head": ...


def make_head(name: str):
    """Construct an empty head by name."""

    from tinyintent.heads.logreg import LogRegHead
    from tinyintent.heads.mlp import MLPHead
    from tinyintent.heads.prototype import PrototypeHead

    heads = {
        "prototype": PrototypeHead,
        "logreg": LogRegHead,
        "mlp": MLPHead,
    }
    if name not in heads:
        raise ValueError(f"unknown head: {name} (choose from {sorted(heads)})")
    return heads[name]()


def load_head(name: str, directory: str | Path) -> Head:
    from tinyintent.heads.logreg import LogRegHead
    from tinyintent.heads.mlp import MLPHead
    from tinyintent.heads.prototype import PrototypeHead

    heads = {
        "prototype": PrototypeHead,
        "logreg": LogRegHead,
        "mlp": MLPHead,
    }
    return heads[name].load(directory)
