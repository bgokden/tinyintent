"""Benchmark the tinyintent pipeline on real intent datasets.

Trains the full pipeline (frozen bge-large + linear head + cross-encoder
reranker) on few-shot splits of CLINC150 and Banking77 and reports top-1
accuracy, averaged over seeds.

    uv run python scripts/benchmark.py
"""

from __future__ import annotations

import random
from statistics import mean

from datasets import load_dataset

from tinyintent import Example, IntentModel

SHOTS = 20
SEEDS = 2


def load_pools(dataset: str):
    if dataset == "clinc":
        by: dict[str, list[str]] = {}
        for row in load_dataset("FastFit/clinc_150", split="train"):
            by.setdefault(row["label"], []).append(row["text"])
        return by, by, True
    train_by: dict[str, list[str]] = {}
    test_by: dict[str, list[str]] = {}
    for row in load_dataset("mteb/banking77", split="train"):
        train_by.setdefault(row["label_text"], []).append(row["text"])
    for row in load_dataset("mteb/banking77", split="test"):
        test_by.setdefault(row["label_text"], []).append(row["text"])
    return train_by, test_by, False


def build(train_by, test_by, same_pool, shots, seed):
    rng = random.Random(seed)
    fit, test = [], []
    for label in sorted(train_by):
        pool = train_by[label][:]
        rng.shuffle(pool)
        fit += [Example(t, label) for t in pool[:shots]]
        held = pool[shots:shots + 20] if same_pool else test_by[label][:20]
        test += [Example(t, label) for t in held]
    return fit, test


def main() -> None:
    for dataset in ("clinc", "banking"):
        train_by, test_by, same_pool = load_pools(dataset)
        accs = []
        for seed in range(SEEDS):
            fit, test = build(train_by, test_by, same_pool, SHOTS, seed)
            model = IntentModel.fit(fit)
            accs.append(model.accuracy(test))
        print(f"{dataset:>8} {SHOTS}-shot: top-1 accuracy {mean(accs):.3f} ({SEEDS} seeds)")


if __name__ == "__main__":
    main()
