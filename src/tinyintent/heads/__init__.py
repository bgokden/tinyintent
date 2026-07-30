from __future__ import annotations

from tinyintent.heads.base import Head, load_head, make_head
from tinyintent.heads.logreg import LogRegHead
from tinyintent.heads.mlp import MLPHead
from tinyintent.heads.prototype import PrototypeHead


__all__ = [
    "Head",
    "LogRegHead",
    "MLPHead",
    "PrototypeHead",
    "load_head",
    "make_head",
]
