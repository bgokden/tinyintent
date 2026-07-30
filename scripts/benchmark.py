"""Benchmark tinyintent top-1 accuracy on real intent datasets.

Few-shot in-scope training, averaged over seeds. Compares classifier heads
(linear vs exemplar) and optionally fine-tuning the encoder.

Run:
    uv run python scripts/benchmark.py --dataset banking --seeds 3
    uv run python scripts/benchmark.py --dataset clinc --shots 10 --classifier linear
"""

from __future__ import annotations

import argparse
import random
from statistics import mean

from datasets import load_dataset

from tinyintent import Example, IntentModel, SentenceEncoder
from tinyintent.finetune import finetune_encoder


def load_pools(dataset: str):
    """Return (train_by_label, test_by_label, same_pool)."""

    if dataset == "clinc":
        ds = load_dataset("FastFit/clinc_150", split="train")
        by: dict[str, list[str]] = {}
        for row in ds:
            by.setdefault(row["label"], []).append(row["text"])
        return by, by, True

    if dataset == "banking":
        train_by: dict[str, list[str]] = {}
        test_by: dict[str, list[str]] = {}
        for row in load_dataset("mteb/banking77", split="train"):
            train_by.setdefault(row["label_text"], []).append(row["text"])
        for row in load_dataset("mteb/banking77", split="test"):
            test_by.setdefault(row["label_text"], []).append(row["text"])
        return train_by, test_by, False

    raise SystemExit(f"unknown dataset: {dataset} (choose clinc or banking)")


def build(train_by, test_by, same_pool, shots, seed):
    labels = sorted(train_by)
    rng = random.Random(seed)

    fit, test = [], []
    for label in labels:
        pool = train_by[label][:]
        rng.shuffle(pool)
        fit += [Example(t, label) for t in pool[:shots]]
        test += [
            Example(t, label)
            for t in (pool[shots:shots + 20] if same_pool else test_by[label][:20])
        ]
    return fit, test, len(labels)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="banking", choices=["clinc", "banking"])
    parser.add_argument("--encoder-model", default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--shots", type=int, default=20)
    parser.add_argument("--classifier", default="linear", choices=["exemplar", "linear"])
    parser.add_argument("--finetune", action="store_true")
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()

    train_by, test_by, same_pool = load_pools(args.dataset)
    frozen = None if args.finetune else SentenceEncoder(args.encoder_model)

    print(f"dataset={args.dataset} encoder={args.encoder_model.split('/')[-1]} "
          f"finetune={args.finetune} shots={args.shots} "
          f"classifier={args.classifier} seeds={args.seeds}")

    accs = []
    for seed in range(args.seeds):
        fit, test, n_labels = build(train_by, test_by, same_pool, args.shots, seed)
        if args.finetune:
            encoder = finetune_encoder(
                fit, out_dir=f"/tmp/tinyintent_ft_{args.dataset}_{seed}",
                base_model=args.encoder_model,
            )
        else:
            encoder = frozen
        model = IntentModel.fit(fit, encoder=encoder, classifier=args.classifier)
        accs.append(model.accuracy(test))

    print(f"top-1 accuracy: {mean(accs):.3f}  ({n_labels} intents)")


if __name__ == "__main__":
    main()
