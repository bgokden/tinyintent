"""Skipping the reranker is a supported choice, not a private code path.

Before this, `fit()` always trained it and the only way out was the private
`_fit_base`. On CLINC150 it buys +0.004 top-1 accuracy for ~15x inference
latency, which is a trade callers should be able to make.
"""

from __future__ import annotations

import numpy as np
import pytest

from tinyintent import Example, IntentModel
from tinyintent.cli import build_parser

from test_model import make_data  # tests/ is on sys.path


class StubSentenceEncoder:
    """Keyword-separable embeddings without touching the network."""

    def __init__(self, model_name="stub", device=None):
        self.model_name = model_name
        self.dim = 16

    def encode(self, texts):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in text.lower().split():
                out[row, hash(token) % self.dim] += 1.0
        norms = np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-8)
        return out / norms

    def spec(self):
        return {"kind": "sentence-transformers", "model": self.model_name}


@pytest.fixture
def offline(monkeypatch):
    # Both call sites: fit() builds one directly, load() goes via make_encoder.
    monkeypatch.setattr("tinyintent.model.SentenceEncoder", StubSentenceEncoder)
    monkeypatch.setattr("tinyintent.encoder.SentenceEncoder", StubSentenceEncoder)


def test_fit_without_reranker_skips_it_entirely(offline):
    """No reranker means no cross-encoder download and no training."""
    def explode(*args, **kwargs):                       # pragma: no cover
        raise AssertionError("the reranker must not be constructed")

    import tinyintent.model as model_module
    original = model_module.CrossEncoderReranker
    model_module.CrossEncoderReranker = explode
    try:
        model = IntentModel.fit(make_data(n=10), reranker=False)
    finally:
        model_module.CrossEncoderReranker = original

    assert model.reranker is None
    assert model.classify("alpha alpha request item") == "a"


def test_confidence_and_abstention_work_without_the_reranker(offline):
    data = make_data(n=12)
    model = IntentModel.fit(data, reranker=False)

    assert model.oos_threshold is not None, "abstention must not need the reranker"
    pred = model.predict("alpha alpha request item")
    assert 0.0 <= pred.confidence <= 1.0
    assert pred.abstain is False
    assert model.oos_rejection_rate(data) >= 0.8


def test_reranker_free_model_saves_and_loads(offline, tmp_path):
    model = IntentModel.fit(make_data(n=10), reranker=False)
    model.save(tmp_path / "m")
    reloaded = IntentModel.load(tmp_path / "m")

    assert reloaded.reranker is None
    text = "beta beta request item"
    assert reloaded.predict(text).intent == model.predict(text).intent
    np.testing.assert_allclose(reloaded.predict(text).confidence,
                               model.predict(text).confidence, atol=1e-5)


def test_default_still_trains_the_reranker(offline, monkeypatch):
    built = []

    class StubReranker:
        def __init__(self, *args, **kwargs):
            built.append(kwargs.get("device"))

        def fit(self, *args, **kwargs):
            return self

        def rerank_with_ce(self, texts, stage1):
            return stage1, np.full_like(stage1, -np.inf)

        def raw_ce_by_label(self, texts, stage1, exclude_self=False):
            return np.full_like(stage1, -np.inf)

    monkeypatch.setattr("tinyintent.model.CrossEncoderReranker", StubReranker)
    model = IntentModel.fit(make_data(n=10))
    assert model.reranker is not None and len(built) == 1


def test_cli_exposes_no_reranker():
    args = build_parser().parse_args(
        ["train", "--data", "d.jsonl", "--out", "m", "--no-reranker"]
    )
    assert args.no_reranker is True
    assert build_parser().parse_args(
        ["train", "--data", "d.jsonl", "--out", "m"]
    ).no_reranker is False
