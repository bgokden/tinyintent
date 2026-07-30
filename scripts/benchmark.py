"""Benchmark tinyintent on real intent datasets, averaged over seeds.

Few-shot in-scope training with some intents held out entirely as unseen
out-of-scope (the hard, near-OOS case), with a slice of those used as
calibration negatives. Compares the decision policies:

- lac  : single absolute-similarity threshold (decisive)
- aps  : two-stage gate + Adaptive Prediction Sets (safe)
- raps : APS with a set-size penalty (safe, more decisive)

Run:
    uv run python scripts/benchmark.py --dataset banking --seeds 3
    uv run python scripts/benchmark.py --dataset clinc --seeds 3 --risk 0.2
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


def build(train_by, test_by, same_pool, shots, n_oos, seed):
    labels = sorted(train_by)
    rng = random.Random(seed)
    rng.shuffle(labels)
    oos, in_scope = set(labels[:n_oos]), labels[n_oos:]

    fit, cal, test = [], [], []
    for label in in_scope:
        pool = train_by[label][:]
        rng.shuffle(pool)
        fit += [Example(t, label) for t in pool[:shots]]
        cal += [Example(t, label) for t in pool[shots:shots + 20]]
        test += [
            Example(t, label)
            for t in (pool[shots + 20:shots + 40] if same_pool else test_by[label][:20])
        ]
    for label in oos:
        pool = train_by[label][:]
        rng.shuffle(pool)
        cal += [Example(t, "oos") for t in pool[:10]]
        test += [
            Example(t, "oos")
            for t in (pool[10:30] if same_pool else test_by[label][:20])
        ]
    return fit, cal, test, len(in_scope), len(oos)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="banking", choices=["clinc", "banking"])
    parser.add_argument("--encoder-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--shots", type=int, default=20)
    parser.add_argument("--n-oos", type=int, default=12)
    parser.add_argument("--risk", type=float, default=0.2)
    parser.add_argument("--transform", default="none", choices=["none", "lda"])
    parser.add_argument("--finetune", action="store_true")
    parser.add_argument("--reject-level", type=float, default=0.1)
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()

    train_by, test_by, same_pool = load_pools(args.dataset)
    frozen = None if args.finetune else SentenceEncoder(args.encoder_model)
    policies = [("gate", "gate", 0.0), ("lac", "lac", 0.0), ("aps", "aps", 0.0)]

    fields = ["coverage", "fire_rate", "fire_accuracy", "ambiguous_rate",
              "abstain_rate", "oos_false_fire", "oos_abstain"]
    print(f"dataset={args.dataset} encoder={args.encoder_model.split('/')[-1]} "
          f"finetune={args.finetune} shots={args.shots} risk={args.risk} "
          f"seeds={args.seeds}")
    print("  ".join(f"{h:>13}" for h in ["policy", *fields]))

    # Fit each seed's model once (fine-tuning is the expensive part), reused
    # across policies.
    seed_models: list[tuple] = []
    for seed in range(args.seeds):
        fit, cal, test, n_in, n_oos = build(
            train_by, test_by, same_pool, args.shots, args.n_oos, seed
        )
        if args.finetune:
            encoder = finetune_encoder(
                fit, out_dir=f"/tmp/tinyintent_ft_{args.dataset}_{seed}",
                base_model=args.encoder_model,
            )
        else:
            encoder = frozen
        model = IntentModel.fit(fit, encoder=encoder, transform=args.transform)
        seed_models.append((model, cal, test))

    for name, method, reg in policies:
        runs = []
        for model, cal, test in seed_models:
            model.calibrate(cal, risk=args.risk, method=method, reg_lambda=reg,
                            reject_level=args.reject_level)
            runs.append(model.evaluate(test).as_dict())
        avg = [mean(r[f] for r in runs) for f in fields]
        print("  ".join([f"{name:>13}", *[f"{v:>13.3f}" for v in avg]]))

    print(f"({n_in} in-scope + {n_oos} OOS intents)")


if __name__ == "__main__":
    main()
