from __future__ import annotations

import numpy as np

from tinyintent.reranker import CrossEncoderReranker, _make_pairs, _nearest_intents


class StubCE:
    """A stand-in cross-encoder: scores a pair high iff the exemplar is 'good'."""

    def predict(self, pairs, **kwargs):
        return np.array([2.0 if ex == "good" else -2.0 for _, ex in pairs])


def test_make_pairs_balanced_and_hard_negatives():
    texts = ["a1", "a2", "b1", "b2", "c1", "c2"]
    y = np.array([0, 0, 1, 1, 2, 2])
    # vectors: intent 0 and 1 close, 2 far -> hard negatives for 0 should be 1
    vectors = np.array([[1, 0], [1, 0.1], [0.9, 0.2], [0.9, 0.1], [0, 1], [0, 1.0]],
                       dtype=np.float32)
    pairs, labels = _make_pairs(texts, y, vectors, n_pos=1, n_neg=1,
                                hard_neg=True, near_m=1, seed=0)
    assert set(labels) == {0.0, 1.0}
    assert labels.count(1.0) > 0 and labels.count(0.0) > 0

    near = _nearest_intents(vectors, y, [0, 1, 2], near_m=1)
    assert near[0] == [1]           # intent 1 is the nearest to intent 0, not 2


def test_rerank_flips_to_cross_encoder_favorite():
    reranker = CrossEncoderReranker(k=2, beta=5.0)
    reranker._ce = StubCE()
    reranker.exemplars = {0: ["bad"], 1: ["good"]}   # CE loves label 1's exemplar

    stage1 = np.array([[0.6, 0.4]])                  # stage-1 prefers label 0
    out = reranker.rerank_scores(["query"], stage1)
    assert int(out.argmax(axis=1)[0]) == 1           # reranker flips to label 1


def test_rerank_respects_stage1_when_beta_zero():
    reranker = CrossEncoderReranker(k=2, beta=0.0)
    reranker._ce = StubCE()
    reranker.exemplars = {0: ["bad"], 1: ["good"]}

    stage1 = np.array([[0.6, 0.4]])
    out = reranker.rerank_scores(["query"], stage1)
    assert int(out.argmax(axis=1)[0]) == 0           # beta=0 keeps stage-1's pick
