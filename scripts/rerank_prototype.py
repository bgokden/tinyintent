"""Prototype: cross-encoder reranking tier over top-k candidate intents (DNNC-style).

Stage 1: frozen encoder + linear head -> top-k candidate intents.
Stage 2: a cross-encoder scores (query, candidate-exemplar) pairs and re-ranks
the k candidates (max over each intent's exemplars).

First measures oracle@k (is the true intent in the top-k?) to size the ceiling,
then optionally runs the cross-encoder rerank and reports the lift.

    uv run python scripts/rerank_prototype.py --dataset banking            # oracle only
    uv run python scripts/rerank_prototype.py --dataset banking --rerank
"""

from __future__ import annotations

import argparse
import os
import sys
from statistics import mean

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiments import build_split, load_pools  # noqa: E402

CROSS_ENCODERS = {
    "ms-marco": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "nli": "cross-encoder/nli-deberta-v3-small",
}


def encode(model, texts):
    return model.encode(
        texts, batch_size=128, convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=False,
    ).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="banking", choices=["clinc", "banking"])
    parser.add_argument("--base", default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--shots", type=int, default=20)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--rerank", action="store_true", help="run the cross-encoder stage")
    args = parser.parse_args()

    from sentence_transformers import CrossEncoder, SentenceTransformer

    enc = SentenceTransformer(args.base)
    cross = {n: CrossEncoder(m) for n, m in CROSS_ENCODERS.items()} if args.rerank else {}
    tr, te, same = load_pools(args.dataset)

    ks = sorted({1, 3, args.k, 10})
    oracle = {k: [] for k in ks}
    rerank_acc = {n: [] for n in cross}

    for seed in range(args.seeds):
        ftx, fy, ttx, ty = build_split(tr, te, same, args.shots, seed)
        ftx, ttx = list(ftx), list(ttx)
        Xtr, Xte = encode(enc, ftx), encode(enc, ttx)
        clf = LogisticRegression(C=10, max_iter=1000).fit(Xtr, fy)
        classes = clf.classes_
        order = np.argsort(-clf.predict_proba(Xte), axis=1)

        for k in ks:
            topk = classes[order[:, :k]]
            oracle[k].append(float(np.mean([ty[i] in topk[i] for i in range(len(ty))])))

        if not cross:
            continue

        cand = classes[order[:, :args.k]]          # [n_test, k] candidate intents
        exemplars: dict[int, list[str]] = {}
        for t, y in zip(ftx, fy):
            exemplars.setdefault(int(y), []).append(t)

        for name, ce in cross.items():
            pairs, meta = [], []
            for i in range(len(ttx)):
                for c in cand[i]:
                    for ex in exemplars[int(c)]:
                        pairs.append([ttx[i], ex])
                        meta.append((i, int(c)))
            scores = np.asarray(ce.predict(pairs, batch_size=256, show_progress_bar=False))
            if scores.ndim == 2:                    # NLI: [contra, entail, neutral]
                scores = scores[:, 1]
            best: dict[tuple[int, int], float] = {}
            for (i, c), s in zip(meta, scores):
                if (i, c) not in best or s > best[(i, c)]:
                    best[(i, c)] = float(s)
            pred = np.array([
                max(cand[i], key=lambda c: best[(i, int(c))]) for i in range(len(ttx))
            ])
            rerank_acc[name].append(float((pred == ty).mean()))

    print(f"{args.dataset} {args.shots}-shot, base={args.base.split('/')[-1]}, "
          f"k={args.k}, {args.seeds} seeds")
    print(f"stage-1 top1 : {mean(oracle[1]):.3f}")
    for k in ks:
        if k != 1:
            print(f"oracle@{k:<2}    : {mean(oracle[k]):.3f}   (ceiling for rerank@{k})")
    for name in cross:
        print(f"rerank {name:9}: {mean(rerank_acc[name]):.3f}")


if __name__ == "__main__":
    main()
