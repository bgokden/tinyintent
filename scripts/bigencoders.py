"""Benchmark a single (larger/newer) encoder for top-1 accuracy.

Loads one model into an isolated cache, scores CLINC150 + Banking77 at 20-shot
with the linear head, prints one result line, and exits. A shell driver runs
this once per encoder and deletes the scratch cache between models so the disk
footprint stays bounded. Reuses the split/head harness from experiments.py.
"""

from __future__ import annotations

import argparse
import os
import sys
from statistics import mean

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiments import build_split, head_accuracy, load_pools


def encode_with(model, texts, prefix, normalize=True):
    if prefix:
        texts = [f"{prefix}{t}" for t in texts]
    vectors = model.encode(
        texts, batch_size=64, convert_to_numpy=True,
        normalize_embeddings=normalize, show_progress_bar=False,
    )
    return vectors.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--cache", default=None, help="cache_folder for the download")
    parser.add_argument("--prefix", default=None, help="task prefix for query-instruct models")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--C", type=float, default=10.0)
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer

    kwargs = {"trust_remote_code": True}
    if args.cache:
        kwargs["cache_folder"] = args.cache
    try:
        model = SentenceTransformer(args.model, **kwargs)
    except Exception as exc:  # report and move on; the driver handles the next model
        print(f"{args.model}\tLOAD_FAILED\t{type(exc).__name__}: {str(exc)[:160]}")
        return

    cells = []
    for name in ("clinc", "banking"):
        tr, te, same = load_pools(name)
        accs = []
        for seed in range(args.seeds):
            ftx, fy, ttx, ty = build_split(tr, te, same, 20, seed)
            Xtr = encode_with(model, ftx, args.prefix)
            Xte = encode_with(model, ttx, args.prefix)
            n = int(fy.max()) + 1
            accs.append(head_accuracy("linear", Xtr, fy, Xte, ty, n, C=args.C))
        cells.append(f"{name}={mean(accs):.3f}")
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"{args.model}\t{params:.0f}M\t" + "\t".join(cells))


if __name__ == "__main__":
    main()
