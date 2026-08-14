"""Benchmark the tinyintent pipeline on real intent datasets.

Trains the full pipeline (frozen bge-large + linear head + cross-encoder
reranker) on few-shot splits of CLINC150 and Banking77 and reports top-1
accuracy, averaged over seeds.

Two protocols per dataset, because they measure different things and the
numbers differ enough to matter:

  official     20 training examples per intent from the train split, scored on
               the dataset's own test split -- unseen text, collected
               separately. This is what published CLINC150 / Banking77 numbers
               mean, so it is the comparable figure.
  train-pool   20 training examples per intent, scored on the next 20 from the
               same train split. Easier: the held-out slice comes from the same
               collection pass as the training text.

Each model is trained once and scored under both, so the second costs only its
inference.

    uv run python scripts/benchmark.py
"""

from __future__ import annotations

import random
from statistics import mean

from datasets import load_dataset

from tinyintent import Example, IntentModel

SHOTS = 20
HELD_OUT = 20
SEEDS = 2

DATASETS = {
    "clinc": ("FastFit/clinc_150", "label", "text"),
    "banking": ("mteb/banking77", "label_text", "text"),
}


def group(rows, label_key: str, text_key: str) -> dict[str, list[str]]:
    by: dict[str, list[str]] = {}
    for row in rows:
        by.setdefault(row[label_key], []).append(row[text_key])
    return by


def load_pools(dataset: str):
    repo, label_key, text_key = DATASETS[dataset]
    train_by = group(load_dataset(repo, split="train"), label_key, text_key)
    test_by = group(load_dataset(repo, split="test"), label_key, text_key)
    return train_by, test_by


def build(train_by, test_by, shots: int, seed: int):
    """One training set, two test sets -- see the module docstring."""

    rng = random.Random(seed)
    fit, official, train_pool = [], [], []
    for label in sorted(train_by):
        pool = train_by[label][:]
        rng.shuffle(pool)
        fit += [Example(t, label) for t in pool[:shots]]
        train_pool += [Example(t, label) for t in pool[shots:shots + HELD_OUT]]
        official += [Example(t, label) for t in test_by[label][:HELD_OUT]]
    return fit, official, train_pool


def main() -> None:
    for dataset in DATASETS:
        train_by, test_by = load_pools(dataset)
        scores: dict[str, list[float]] = {"official": [], "train-pool": []}
        for seed in range(SEEDS):
            fit, official, train_pool = build(train_by, test_by, SHOTS, seed)
            model = IntentModel.fit(fit)
            scores["official"].append(model.accuracy(official))
            scores["train-pool"].append(model.accuracy(train_pool))

        print(f"{dataset} {SHOTS}-shot, {SEEDS} seeds, {len(train_by)} intents")
        for protocol, values in scores.items():
            print(f"  {protocol:<12} top-1 accuracy {mean(values):.3f}")


if __name__ == "__main__":
    main()
