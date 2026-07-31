"""Measure how often the agent routes to the right intent (path).

Holds out part of the conversational-flow data, trains on the rest, and scores
the routing on the unseen utterances: top-1 accuracy, macro / weighted F1, and a
per-intent precision/recall/F1 breakdown. Predictions are pooled across seeds so
the per-intent numbers are stable on this small toy set.

    uv run python examples/eval_flow.py
"""

from __future__ import annotations

from statistics import mean

from tinyintent import IntentModel, OOS_LABEL, load_jsonl, score_predictions, split

DATA = "examples/sales_flow.jsonl"
SEEDS = 3
TEST_FRAC = 0.4


def main() -> None:
    data = load_jsonl(DATA)
    accs, macros = [], []
    true_pool, pred_pool = [], []

    for seed in range(SEEDS):
        train, test = split(data, test_frac=TEST_FRAC, seed=seed)
        model = IntentModel.fit(train)
        report = model.evaluate(test)                 # excludes oos
        accs.append(report.accuracy)
        macros.append(report.macro_f1)

        in_scope = [ex for ex in test if ex.label != OOS_LABEL]
        preds = model.classify_batch([ex.text for ex in in_scope])
        true_pool += [ex.label for ex in in_scope]
        pred_pool += preds

    pooled = score_predictions(true_pool, pred_pool)
    print(f"routing over {SEEDS} splits of {DATA} (test_frac={TEST_FRAC})")
    print(f"  accuracy    : {mean(accs):.3f}  (per-split mean)")
    print(f"  macro F1    : {mean(macros):.3f}  (per-split mean)")
    print(f"  weighted F1 : {pooled.weighted_f1:.3f}  (pooled)")
    print(f"  pooled n    : {pooled.n}\n")
    print(pooled.table())


if __name__ == "__main__":
    main()
