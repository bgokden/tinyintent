from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# Label used to mark out-of-scope / none-of-the-above examples. When a
# dataset contains examples with this label, calibration treats them as
# cases the model should abstain on rather than classify.
OOS_LABEL = "oos"


@dataclass
class Example:
    text: str
    label: str


def load_jsonl(path: str | Path) -> list[Example]:
    """Load examples from a JSON Lines file of {"text", "label"} objects."""

    examples: list[Example] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        examples.append(Example(text=str(record["text"]), label=str(record["label"])))
    return examples


def from_fewshot(mapping: dict[str, list[str]]) -> list[Example]:
    """Build examples from a {label: [utterances]} mapping."""

    return [
        Example(text=text, label=label)
        for label, texts in mapping.items()
        for text in texts
    ]


def labels_of(examples: list[Example], include_oos: bool = False) -> list[str]:
    """Sorted unique in-scope labels (OOS excluded unless requested)."""

    labels = {ex.label for ex in examples}
    if not include_oos:
        labels.discard(OOS_LABEL)
    return sorted(labels)


def split(
    examples: list[Example],
    test_frac: float = 0.2,
    seed: int = 0,
) -> tuple[list[Example], list[Example]]:
    """Stratified split so every label keeps a share on both sides."""

    rng = np.random.default_rng(seed)
    by_label: dict[str, list[Example]] = defaultdict(list)
    for ex in examples:
        by_label[ex.label].append(ex)

    train: list[Example] = []
    test: list[Example] = []
    for label, items in by_label.items():
        order = rng.permutation(len(items))
        cut = int(round(len(items) * (1.0 - test_frac)))
        cut = min(max(cut, 1), len(items))  # keep at least one in train
        for position, idx in enumerate(order):
            (train if position < cut else test).append(items[idx])

    return train, test
