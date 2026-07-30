"""Benchmark tinyintent on CLINC150.

Few-shot in-scope training with 20 intents held out entirely as unseen
out-of-scope, so the report measures both in-scope quality and genuine
novelty rejection. Downloads the dataset from the Hugging Face Hub.

Run:
    uv run python scripts/benchmark.py --shots 20
"""

from __future__ import annotations

import argparse
import random

from datasets import load_dataset

from tinyintent import Example, IntentModel, SentenceEncoder


def build_splits(shots: int, n_oos: int, seed: int):
    ds = load_dataset("FastFit/clinc_150", split="train")
    by_label: dict[str, list[str]] = {}
    for row in ds:
        by_label.setdefault(row["label"], []).append(row["text"])

    labels = sorted(by_label)
    rng = random.Random(seed)
    rng.shuffle(labels)
    oos_labels = set(labels[:n_oos])
    in_scope = labels[n_oos:]

    fit, cal, test = [], [], []
    for label in in_scope:
        texts = by_label[label][:]
        rng.shuffle(texts)
        fit += [Example(t, label) for t in texts[:shots]]
        cal += [Example(t, label) for t in texts[shots:shots + 20]]
        test += [Example(t, label) for t in texts[shots + 20:shots + 40]]
    for label in oos_labels:
        # Split each held-out intent: some examples as OOS negatives for
        # calibration (the abstain floor), the rest as OOS at test time.
        cal += [Example(t, "oos") for t in by_label[label][:10]]
        test += [Example(t, "oos") for t in by_label[label][10:30]]

    return fit, cal, test, len(in_scope), len(oos_labels)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shots", type=int, default=20)
    parser.add_argument("--n-oos", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    fit, cal, test, n_in, n_oos = build_splits(args.shots, args.n_oos, args.seed)
    print(f"{n_in} in-scope intents ({args.shots}-shot), {n_oos} held-out OOS intents")
    print(f"fit={len(fit)} cal={len(cal)} test={len(test)}")

    encoder = SentenceEncoder()
    model = IntentModel.fit(fit, encoder=encoder)

    header = ("risk", "coverage", "fire_rate", "fire_acc", "ambiguous", "abstain",
              "oos_falsefire", "oos_abstain")
    print("\n" + "  ".join(f"{h:>13}" for h in header))
    for risk in (0.05, 0.10, 0.20):
        model.calibrate(cal, risk=risk)
        r = model.evaluate(test)
        row = (risk, r.coverage, r.fire_rate, r.fire_accuracy, r.ambiguous_rate,
               r.abstain_rate, r.oos_false_fire, r.oos_abstain)
        print("  ".join(f"{v:>13.3f}" for v in row))


if __name__ == "__main__":
    main()
