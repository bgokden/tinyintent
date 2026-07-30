"""Experiment campaign for tinyintent top-1 accuracy.

Encode once per (encoder, seed), then sweep heads / regularization / shots on
the cached embeddings so most of the matrix is nearly free. Extra heads
(centroid, knn) are built here for comparison only; the shipped package keeps
just the linear and exemplar heads.

Run one experiment:
    uv run python scripts/experiments.py --exp heads
    uv run python scripts/experiments.py --exp reg
    uv run python scripts/experiments.py --exp encoders
    uv run python scripts/experiments.py --exp curve
    uv run python scripts/experiments.py --exp finetune
    uv run python scripts/experiments.py --exp static
"""

from __future__ import annotations

import argparse
import random
from statistics import mean

import numpy as np
from datasets import load_dataset


ENCODERS = {
    "bge-small": "BAAI/bge-small-en-v1.5",
    "bge-base": "BAAI/bge-base-en-v1.5",
    "bge-large": "BAAI/bge-large-en-v1.5",
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "gte-small": "thenlper/gte-small",
    "e5-small": "intfloat/e5-small-v2",
}

_MODEL_CACHE: dict[str, object] = {}


# -- data ------------------------------------------------------------------

def load_pools(dataset: str):
    if dataset == "clinc":
        by: dict[str, list[str]] = {}
        for row in load_dataset("FastFit/clinc_150", split="train"):
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
    raise SystemExit(f"unknown dataset: {dataset}")


def build_split(train_by, test_by, same_pool, shots, seed):
    labels = sorted(train_by)
    index = {label: i for i, label in enumerate(labels)}
    rng = random.Random(seed)

    fit_texts, fit_y, test_texts, test_y = [], [], [], []
    for label in labels:
        pool = train_by[label][:]
        rng.shuffle(pool)
        fit_texts += pool[:shots]
        fit_y += [index[label]] * len(pool[:shots])
        held = pool[shots:shots + 20] if same_pool else test_by[label][:20]
        test_texts += held
        test_y += [index[label]] * len(held)
    return fit_texts, np.array(fit_y), test_texts, np.array(test_y)


# -- encoding --------------------------------------------------------------

def get_model(name: str):
    if name not in _MODEL_CACHE:
        from sentence_transformers import SentenceTransformer

        _MODEL_CACHE[name] = SentenceTransformer(ENCODERS[name])
    return _MODEL_CACHE[name]


def encode(name: str, texts: list[str], normalize: bool) -> np.ndarray:
    # e5 models expect a "query:" prefix; apply it for a fair comparison.
    if name == "e5-small":
        texts = [f"query: {t}" for t in texts]
    vectors = get_model(name).encode(
        texts, batch_size=128, convert_to_numpy=True,
        normalize_embeddings=normalize, show_progress_bar=False,
    )
    return vectors.astype(np.float32)


def _unit(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)


# -- heads (comparison only) ----------------------------------------------

def head_accuracy(head, Xtr, ytr, Xte, yte, n_labels, C=10.0, k=5) -> float:
    if head == "linear":
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression(C=C, max_iter=1000).fit(Xtr, ytr)
        pred = clf.predict(Xte)
    elif head == "exemplar":
        A, B = _unit(Xtr), _unit(Xte)
        sims = B @ A.T
        scores = np.full((B.shape[0], n_labels), -1.0, dtype=np.float32)
        for label in range(n_labels):
            mask = ytr == label
            if mask.any():
                scores[:, label] = sims[:, mask].max(axis=1)
        pred = scores.argmax(axis=1)
    elif head == "centroid":
        A = _unit(Xtr)
        cents = np.stack([_unit(A[ytr == c].mean(axis=0, keepdims=True))[0]
                          for c in range(n_labels)])
        pred = (_unit(Xte) @ cents.T).argmax(axis=1)
    elif head == "knn":
        from sklearn.neighbors import KNeighborsClassifier

        clf = KNeighborsClassifier(n_neighbors=k, metric="cosine").fit(Xtr, ytr)
        pred = clf.predict(Xte)
    else:
        raise ValueError(head)
    return float((pred == yte).mean())


# -- experiments -----------------------------------------------------------

def datasets_arg(names):
    return [(n, *load_pools(n)) for n in names]


def exp_heads(seeds):
    heads = ["linear", "exemplar", "centroid", "knn"]
    print("E1 heads | frozen bge-small, 20-shot, normalized embeddings")
    print(f"{'dataset':>9} " + " ".join(f"{h:>9}" for h in heads))
    for name, tr, te, same in datasets_arg(["clinc", "banking"]):
        rows = {h: [] for h in heads}
        for seed in range(seeds):
            ftx, fy, ttx, ty = build_split(tr, te, same, 20, seed)
            Xtr, Xte = encode("bge-small", ftx, True), encode("bge-small", ttx, True)
            n = int(fy.max()) + 1
            for h in heads:
                rows[h].append(head_accuracy(h, Xtr, fy, Xte, ty, n))
        print(f"{name:>9} " + " ".join(f"{mean(rows[h]):>9.3f}" for h in heads))


def exp_reg(seeds):
    Cs = [0.1, 1.0, 10.0, 100.0]
    print("E2 linear reg | frozen bge-small, 20-shot")
    for norm in (True, False):
        tag = "normalized" if norm else "raw"
        print(f"  embeddings={tag}")
        print(f"{'dataset':>9} " + " ".join(f"{'C=' + str(c):>9}" for c in Cs))
        for name, tr, te, same in datasets_arg(["clinc", "banking"]):
            rows = {c: [] for c in Cs}
            for seed in range(seeds):
                ftx, fy, ttx, ty = build_split(tr, te, same, 20, seed)
                Xtr, Xte = encode("bge-small", ftx, norm), encode("bge-small", ttx, norm)
                n = int(fy.max()) + 1
                for c in Cs:
                    rows[c].append(head_accuracy("linear", Xtr, fy, Xte, ty, n, C=c))
            print(f"{name:>9} " + " ".join(f"{mean(rows[c]):>9.3f}" for c in Cs))


def exp_encoders(seeds, C):
    names = list(ENCODERS)
    print(f"E3 encoders | frozen, linear head C={C}, 20-shot, normalized")
    print(f"{'dataset':>9} " + " ".join(f"{n:>10}" for n in names))
    for name, tr, te, same in datasets_arg(["clinc", "banking"]):
        rows = {e: [] for e in names}
        for seed in range(seeds):
            ftx, fy, ttx, ty = build_split(tr, te, same, 20, seed)
            n = int(fy.max()) + 1
            for e in names:
                Xtr, Xte = encode(e, ftx, True), encode(e, ttx, True)
                rows[e].append(head_accuracy("linear", Xtr, fy, Xte, ty, n, C=C))
        print(f"{name:>9} " + " ".join(f"{mean(rows[e]):>10.3f}" for e in names))


def exp_curve(seeds, encoder, C):
    shots_list = [1, 5, 10, 20, 50, 100]
    print(f"E4 learning curve | frozen {encoder}, linear head C={C}, normalized")
    print(f"{'dataset':>9} " + " ".join(f"{str(s) + 'sh':>7}" for s in shots_list))
    for name, tr, te, same in datasets_arg(["clinc", "banking"]):
        rows = {s: [] for s in shots_list}
        for seed in range(seeds):
            for s in shots_list:
                ftx, fy, ttx, ty = build_split(tr, te, same, s, seed)
                if len(ftx) == 0 or len(ttx) == 0:      # not enough data at this shot count
                    continue
                Xtr, Xte = encode(encoder, ftx, True), encode(encoder, ttx, True)
                n = int(fy.max()) + 1
                rows[s].append(head_accuracy("linear", Xtr, fy, Xte, ty, n, C=C))
        cells = [f"{mean(rows[s]):>7.3f}" if rows[s] else f"{'-':>7}" for s in shots_list]
        print(f"{name:>9} " + " ".join(cells))


def exp_finetune(seeds, encoder, C):
    import shutil

    from tinyintent import Example, IntentModel
    from tinyintent.finetune import finetune_encoder
    from tinyintent.encoder import SentenceEncoder

    print(f"E5 fine-tune | {encoder}, linear head C={C}")
    print(f"{'dataset':>9} {'shots':>6} {'frozen':>8} {'ft-e1':>8} {'ft-e2':>8} {'ft-e3':>8}")
    frozen = SentenceEncoder(ENCODERS[encoder])
    work_dir = "/tmp/ti_ft_work"      # reused + overwritten so disk stays bounded
    try:
        for name, tr, te, same in datasets_arg(["clinc", "banking"]):
            for shots in (10, 20):
                frz, fts = [], {1: [], 2: [], 3: []}
                for seed in range(seeds):
                    ftx, fy, ttx, ty = build_split(tr, te, same, shots, seed)
                    fit = [Example(t, str(int(y))) for t, y in zip(ftx, fy)]
                    test = [Example(t, str(int(y))) for t, y in zip(ttx, ty)]
                    m = IntentModel.fit(fit, encoder=frozen, classifier="linear")
                    frz.append(m.accuracy(test))
                    for ep in (1, 2, 3):
                        enc = finetune_encoder(
                            fit, out_dir=work_dir,
                            base_model=ENCODERS[encoder], epochs=ep,
                        )
                        m = IntentModel.fit(fit, encoder=enc, classifier="linear")
                        fts[ep].append(m.accuracy(test))
                print(f"{name:>9} {shots:>6} {mean(frz):>8.3f} "
                      f"{mean(fts[1]):>8.3f} {mean(fts[2]):>8.3f} {mean(fts[3]):>8.3f}")
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def exp_static(seeds, C):
    from model2vec import StaticModel

    model = StaticModel.from_pretrained("minishlab/potion-base-8M")

    def enc_static(texts):
        v = np.asarray(model.encode(list(texts)), dtype=np.float32)
        return _unit(v)

    print(f"E6 static tier | Model2Vec potion-base-8M, linear head C={C}, 20-shot")
    print(f"{'dataset':>9} {'static':>8} {'bge-small':>10}")
    for name, tr, te, same in datasets_arg(["clinc", "banking"]):
        st, bg = [], []
        for seed in range(seeds):
            ftx, fy, ttx, ty = build_split(tr, te, same, 20, seed)
            n = int(fy.max()) + 1
            st.append(head_accuracy("linear", enc_static(ftx), fy, enc_static(ttx), ty, n, C=C))
            bg.append(head_accuracy("linear", encode("bge-small", ftx, True), fy,
                                    encode("bge-small", ttx, True), ty, n, C=C))
        print(f"{name:>9} {mean(st):>8.3f} {mean(bg):>10.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp", required=True,
                        choices=["heads", "reg", "encoders", "curve", "finetune", "static"])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--encoder", default="bge-small")
    parser.add_argument("--C", type=float, default=10.0)
    args = parser.parse_args()

    if args.exp == "heads":
        exp_heads(args.seeds)
    elif args.exp == "reg":
        exp_reg(args.seeds)
    elif args.exp == "encoders":
        exp_encoders(args.seeds, args.C)
    elif args.exp == "curve":
        exp_curve(args.seeds, args.encoder, args.C)
    elif args.exp == "finetune":
        exp_finetune(args.seeds, args.encoder, args.C)
    elif args.exp == "static":
        exp_static(args.seeds, args.C)


if __name__ == "__main__":
    main()
