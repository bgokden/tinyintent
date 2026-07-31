"""Prototype: DNNC-style trained cross-encoder reranker.

Off-the-shelf cross-encoders (relevance/NLI) hurt, because they do not encode
"same intent". DNNC trains the cross-encoder on the task: same-intent utterance
pairs are positives (1), different-intent pairs are negatives (0). At inference,
stage-1 (frozen encoder + linear head) proposes top-k intents, and the trained
cross-encoder scores the query against each candidate intent's exemplars (max),
re-ranking the k.

    uv run python scripts/dnnc_rerank.py --dataset banking --k 5 --seeds 2
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from statistics import mean

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiments import build_split, load_pools  # noqa: E402


def encode(model, texts):
    return model.encode(
        texts, batch_size=128, convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=False,
    ).astype(np.float32)


def _nearest_intents(Xtr, fy, labels, near_m):
    cent = {}
    for c in labels:
        rows = Xtr[fy == c]
        v = rows.mean(axis=0)
        cent[c] = v / max(float(np.linalg.norm(v)), 1e-8)
    near = {}
    for c in labels:
        sims = sorted(((c2, float(cent[c] @ cent[c2])) for c2 in labels if c2 != c),
                      key=lambda x: -x[1])
        near[c] = [c2 for c2, _ in sims[:near_m]]
    return near


def make_pairs(ftx, fy, Xtr, n_pos, n_neg, hard, near_m, seed):
    rng = random.Random(seed)
    by: dict[int, list[str]] = {}
    for t, y in zip(ftx, fy):
        by.setdefault(int(y), []).append(t)
    labels = sorted(by)
    near = _nearest_intents(Xtr, fy, labels, near_m) if hard else None

    pairs, out = [], []
    for c, texts in by.items():
        neg_labels = near[c] if hard else [lab for lab in labels if lab != c]
        neg_pool = [t for lab in neg_labels for t in by[lab]]
        for a in texts:
            same = [t for t in texts if t != a]
            for b in rng.sample(same, min(n_pos, len(same))):
                pairs.append([a, b])
                out.append(1.0)
            for b in rng.sample(neg_pool, min(n_neg, len(neg_pool))):
                pairs.append([a, b])
                out.append(0.0)
    return pairs, out


def train_cross_encoder(base, pairs, labels, epochs, seed):
    from sentence_transformers import InputExample
    from sentence_transformers.cross_encoder import CrossEncoder
    from torch.utils.data import DataLoader

    # ignore_mismatched_sizes lets us re-head a 3-label NLI checkpoint to 1 logit
    model = CrossEncoder(base, num_labels=1,
                         model_kwargs={"ignore_mismatched_sizes": True})
    examples = [InputExample(texts=p, label=y) for p, y in zip(pairs, labels)]
    loader = DataLoader(examples, batch_size=32, shuffle=True)
    model.fit(train_dataloader=loader, epochs=epochs, warmup_steps=int(0.1 * len(loader)),
              show_progress_bar=False)
    return model


def _zscore_rows(matrix):
    mean = matrix.mean(axis=1, keepdims=True)
    std = matrix.std(axis=1, keepdims=True)
    return (matrix - mean) / np.maximum(std, 1e-8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="banking", choices=["clinc", "banking"])
    parser.add_argument("--base", default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--ce-base", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--shots", type=int, default=20)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--n-pos", type=int, default=8)
    parser.add_argument("--n-neg", type=int, default=8)
    parser.add_argument("--hard-neg", action="store_true", help="mine negatives from nearest intents")
    parser.add_argument("--near-m", type=int, default=10, help="how many nearest intents for hard negs")
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer

    enc = SentenceTransformer(args.base)
    tr, te, same = load_pools(args.dataset)

    betas = [0.0, 0.5, 1.0, 2.0, 5.0, 100.0]   # 0 = pure stage-1; 100 ~ pure rerank
    stage1, oracle_k, pure_rerank = [], [], []
    ens = {b: [] for b in betas}

    for seed in range(args.seeds):
        ftx, fy, ttx, ty = build_split(tr, te, same, args.shots, seed)
        ftx, ttx = list(ftx), list(ttx)
        Xtr, Xte = encode(enc, ftx), encode(enc, ttx)
        clf = LogisticRegression(C=10, max_iter=1000).fit(Xtr, fy)
        classes = clf.classes_
        proba = clf.predict_proba(Xte)
        order = np.argsort(-proba, axis=1)
        cand = classes[order[:, :args.k]]                       # [n, k]
        cand_cols = order[:, :args.k]                           # column indices for proba

        stage1.append(float((classes[order[:, 0]] == ty).mean()))
        oracle_k.append(float(np.mean([ty[i] in cand[i] for i in range(len(ty))])))

        pairs, labels = make_pairs(ftx, fy, Xtr, args.n_pos, args.n_neg,
                                   args.hard_neg, args.near_m, seed)
        ce = train_cross_encoder(args.ce_base, pairs, labels, args.epochs, seed)

        exemplars: dict[int, list[str]] = {}
        for t, y in zip(ftx, fy):
            exemplars.setdefault(int(y), []).append(t)

        ce_pairs, meta = [], []
        for i in range(len(ttx)):
            for j, c in enumerate(cand[i]):
                for ex in exemplars[int(c)]:
                    ce_pairs.append([ttx[i], ex])
                    meta.append((i, j))
        raw = np.asarray(ce.predict(ce_pairs, batch_size=256, show_progress_bar=False))
        grouped: dict[tuple[int, int], list[float]] = {}
        for (i, j), s in zip(meta, raw):
            grouped.setdefault((i, j), []).append(float(s))
        ce_score = np.full((len(ttx), args.k), -1e9)            # mean of top-3 exemplars
        for (i, j), vals in grouped.items():
            top = sorted(vals, reverse=True)[:3]
            ce_score[i, j] = sum(top) / len(top)

        stage1_score = np.log(np.maximum(proba[np.arange(len(ttx))[:, None], cand_cols], 1e-12))
        s1z, cez = _zscore_rows(stage1_score), _zscore_rows(ce_score)

        pure_rerank.append(float((cand[np.arange(len(ttx)), ce_score.argmax(1)] == ty).mean()))
        for b in betas:
            pick = (s1z + b * cez).argmax(1)
            ens[b].append(float((cand[np.arange(len(ttx)), pick] == ty).mean()))

    print(f"{args.dataset} {args.shots}-shot, base={args.base.split('/')[-1]}, "
          f"ce={args.ce_base.split('/')[-1]}, epochs={args.epochs}, k={args.k}, "
          f"hard_neg={args.hard_neg} near_m={args.near_m}, {args.seeds} seeds")
    print(f"stage-1 top1     : {mean(stage1):.3f}")
    print(f"oracle@{args.k}         : {mean(oracle_k):.3f}   (ceiling)")
    print(f"pure rerank      : {mean(pure_rerank):.3f}")
    for b in betas:
        tag = "pure stage-1" if b == 0 else ("~pure rerank" if b == 100 else "")
        print(f"ensemble b={b:<5}: {mean(ens[b]):.3f}   {tag}")


if __name__ == "__main__":
    main()
