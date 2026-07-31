from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

DEFAULT_CE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _nearest_intents(vectors: np.ndarray, y: np.ndarray, labels: list[int], near_m: int):
    """For each intent, the ``near_m`` most similar other intents by centroid."""

    cent = {}
    for c in labels:
        v = vectors[y == c].mean(axis=0)
        cent[c] = v / max(float(np.linalg.norm(v)), 1e-8)
    near = {}
    for c in labels:
        sims = sorted(((c2, float(cent[c] @ cent[c2])) for c2 in labels if c2 != c),
                      key=lambda x: -x[1])
        near[c] = [c2 for c2, _ in sims[:near_m]]
    return near


def _make_pairs(texts, y, vectors, n_pos, n_neg, hard_neg, near_m, seed):
    """Same-intent pairs (label 1) and different-intent pairs (label 0).

    With ``hard_neg`` the negatives are drawn from each intent's nearest
    neighbours, so the cross-encoder trains on the confusable pairs it must
    actually disambiguate rather than easy random ones.
    """

    rng = random.Random(seed)
    by: dict[int, list[str]] = {}
    for t, label in zip(texts, y):
        by.setdefault(int(label), []).append(t)
    labels = sorted(by)
    near = _nearest_intents(vectors, y, labels, near_m) if hard_neg else None

    pairs, targets = [], []
    for c, own in by.items():
        neg_labels = near[c] if hard_neg else [lab for lab in labels if lab != c]
        neg_pool = [t for lab in neg_labels for t in by[lab]]
        if not neg_pool:
            continue
        for anchor in own:
            same = [t for t in own if t != anchor]
            for partner in rng.sample(same, min(n_pos, len(same))):
                pairs.append([anchor, partner])
                targets.append(1.0)
            for partner in rng.sample(neg_pool, min(n_neg, len(neg_pool))):
                pairs.append([anchor, partner])
                targets.append(0.0)
    return pairs, targets


def _zscore_rows(matrix: np.ndarray) -> np.ndarray:
    mean = matrix.mean(axis=1, keepdims=True)
    std = matrix.std(axis=1, keepdims=True)
    return (matrix - mean) / np.maximum(std, 1e-8)


class CrossEncoderReranker:
    """A trained cross-encoder that re-ranks stage-1's top-k candidate intents.

    Stage 1 (a bi-encoder + linear head) proposes the top-k intents cheaply.
    This reranker reads (query, candidate-exemplar) pairs together and adjusts
    the ranking, ensembling its signal with the stage-1 scores. The
    cross-encoder is trained on the labelled data (off-the-shelf ones do not
    encode "same intent" and hurt), so this needs a training step.
    """

    def __init__(self, k: int = 5, beta: float = 0.5,
                 base_model: str = DEFAULT_CE_MODEL):
        self.k = k
        self.beta = beta
        self.base_model = base_model
        self._ce = None
        self.exemplars: dict[int, list[str]] = {}

    def fit(self, texts, y, vectors, epochs: int = 3, n_pos: int = 8, n_neg: int = 8,
            hard_neg: bool = True, near_m: int = 10, batch_size: int = 32,
            seed: int = 0) -> "CrossEncoderReranker":
        from sentence_transformers import InputExample
        from sentence_transformers.cross_encoder import CrossEncoder
        from torch.utils.data import DataLoader

        y = np.asarray(y)
        pairs, targets = _make_pairs(texts, y, vectors, n_pos, n_neg,
                                     hard_neg, near_m, seed)
        # ignore_mismatched_sizes lets a 3-label NLI checkpoint re-head to 1 logit
        self._ce = CrossEncoder(self.base_model, num_labels=1,
                                model_kwargs={"ignore_mismatched_sizes": True})
        examples = [InputExample(texts=p, label=t) for p, t in zip(pairs, targets)]
        loader = DataLoader(examples, batch_size=batch_size, shuffle=True)
        self._ce.fit(train_dataloader=loader, epochs=epochs,
                     warmup_steps=int(0.1 * len(loader)), show_progress_bar=False)

        self.exemplars = {}
        for t, label in zip(texts, y):
            self.exemplars.setdefault(int(label), []).append(t)
        return self

    def rerank_scores(self, query_texts: list[str], stage1: np.ndarray) -> np.ndarray:
        """Return an adjusted score matrix; argmax/argsort then picks the rerank."""

        n, n_labels = stage1.shape
        k = min(self.k, n_labels)
        order = np.argsort(-stage1, axis=1)[:, :k]        # top-k label indices per row

        pairs, meta = [], []
        for i in range(n):
            for j, lab in enumerate(order[i]):
                for ex in self.exemplars.get(int(lab), []):
                    pairs.append([query_texts[i], ex])
                    meta.append((i, j))
        if not pairs:
            return stage1

        raw = np.asarray(self._ce.predict(pairs, batch_size=256, show_progress_bar=False))
        grouped: dict[tuple[int, int], list[float]] = {}
        for (i, j), s in zip(meta, raw):
            grouped.setdefault((i, j), []).append(float(s))
        ce_score = np.full((n, k), -1e9)
        for (i, j), vals in grouped.items():
            top = sorted(vals, reverse=True)[:3]          # mean of top-3 exemplars
            ce_score[i, j] = sum(top) / len(top)

        cand_scores = np.take_along_axis(stage1, order, axis=1)
        s1z = _zscore_rows(np.log(np.clip(cand_scores, 1e-12, None)))
        cez = _zscore_rows(ce_score)
        combined = s1z + self.beta * cez

        out = stage1.astype(float).copy()
        for i in range(n):
            base = float(out[i].max()) + 1.0              # keep candidates above the rest
            ranked = np.argsort(-combined[i])
            for rank, j in enumerate(ranked):
                out[i, order[i][j]] = base + (k - rank)
        return out

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self._ce.save(str(directory / "ce"))
        (directory / "reranker.json").write_text(
            json.dumps({
                "k": self.k, "beta": self.beta, "base_model": self.base_model,
                "exemplars": {str(c): ex for c, ex in self.exemplars.items()},
            }),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: str | Path) -> "CrossEncoderReranker":
        from sentence_transformers.cross_encoder import CrossEncoder

        directory = Path(directory)
        config = json.loads((directory / "reranker.json").read_text(encoding="utf-8"))
        reranker = cls(k=config["k"], beta=config["beta"], base_model=config["base_model"])
        reranker._ce = CrossEncoder(str(directory / "ce"))
        reranker.exemplars = {int(c): ex for c, ex in config["exemplars"].items()}
        return reranker
