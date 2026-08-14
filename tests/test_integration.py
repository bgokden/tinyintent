"""End-to-end coverage of the real `IntentModel.fit()` path.

Every other test injects `HashingEncoder` and stubs the cross-encoder, so the
pipeline users actually run -- bge-large plus a trained reranker -- had no
coverage at all. That is why two missing runtime dependencies shipped.

Marked `slow`: these download models from the HF Hub and train. Run with
`uv run pytest -m slow`.
"""

from __future__ import annotations

import numpy as np
import pytest

from tinyintent import Example, IntentModel
from tinyintent.encoder import HashingEncoder
from tinyintent.reranker import CrossEncoderReranker

pytestmark = pytest.mark.slow


def tiny_data() -> list[Example]:
    rows = [
        ("where is my order", "track"),
        ("when will my package arrive", "track"),
        ("has my parcel shipped yet", "track"),
        ("track my delivery", "track"),
        ("i forgot my password", "login"),
        ("cannot log into my account", "login"),
        ("reset my password please", "login"),
        ("locked out of my account", "login"),
    ]
    return [Example(text=t, label=label) for t, label in rows]


def test_fit_end_to_end_and_roundtrip(tmp_path):
    """The documented quickstart, start to finish."""
    model = IntentModel.fit(tiny_data())
    assert model.reranker is not None

    pred = model.predict("when is my delivery coming")
    assert pred.intent == "track"
    assert 0.0 <= pred.score <= 1.0
    assert [label for label, _ in pred.ranking] and pred.ranking[0][0] == pred.intent
    assert pred.explanation is not None

    model.save(tmp_path / "m")
    reloaded = IntentModel.load(tmp_path / "m")
    assert reloaded.reranker is not None, "reranker must survive save/load"

    texts = ["when is my delivery coming", "i cannot sign in"]
    before = model.predict_batch(texts)
    after = reloaded.predict_batch(texts)
    assert [p.intent for p in before] == [p.intent for p in after]
    assert np.allclose([p.score for p in before], [p.score for p in after], atol=1e-5)


def test_reranker_fit_is_reproducible_for_a_fixed_seed():
    """Same inputs + same seed must give the same model.

    `seed` used to feed only `_make_pairs`, leaving the re-headed classifier's
    init and the loader shuffle on the unseeded global torch RNG. Identical
    runs then produced different accuracy.
    """
    base = IntentModel._fit_base(tiny_data(), HashingEncoder(dim=512))
    texts, y, vectors = base._train_texts, base._train_y, base._train_vectors
    probe = ["when is my delivery coming", "i cannot sign in", "unrelated banana text"]
    stage1 = base.scorer.scores(base.encoder.encode(probe))

    first = CrossEncoderReranker().fit(texts, y, vectors, seed=0)
    second = CrossEncoderReranker().fit(texts, y, vectors, seed=0)

    np.testing.assert_allclose(
        first.rerank_scores(probe, stage1),
        second.rerank_scores(probe, stage1),
        atol=1e-5,
        err_msg="reranker training is not reproducible for a fixed seed",
    )
