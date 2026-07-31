from __future__ import annotations

import numpy as np

from tinyintent import Example, IntentModel
from tinyintent.encoder import HashingEncoder


def make_data(n=10):
    # Keyword-separable so the offline hashing encoder can distinguish them,
    # with an out-of-scope group whose vocabulary is disjoint.
    rows = []
    for word, label in [("alpha", "a"), ("beta", "b"), ("gamma", "c")]:
        for i in range(n):
            rows.append(Example(f"{word} {word} request item {i}", label))
    for i in range(n):
        rows.append(Example(f"zzz{i} unrelated foreign token {i}", "oos"))
    return rows


def make_model(dim=1024):
    # _fit_base is the stage-1 pipeline; tests inject the offline hashing
    # encoder and skip the (network-trained) reranker.
    return IntentModel._fit_base(make_data(n=14), HashingEncoder(dim=dim))


def test_labels_exclude_oos():
    model = IntentModel._fit_base(make_data(), HashingEncoder(dim=512))
    assert model.label_names == ["a", "b", "c"]


def test_classify_decides_top1():
    model = make_model()
    assert model.classify("alpha alpha request item") == "a"
    assert model.classify("beta beta request item") == "b"


def test_always_decides_never_oos():
    model = make_model()
    pred = model.predict("completely different banana vocabulary here")
    assert pred.intent in model.label_names


def test_predict_ranking_and_explanation():
    model = make_model()
    pred = model.predict("alpha alpha request item extra")
    assert pred.intent == "a"
    assert pred.ranking[0][0] == "a"
    assert len(pred.ranking) == len(model.label_names)
    assert pred.explanation is not None


def test_accuracy():
    model = make_model()
    assert model.accuracy(make_data(n=14)) >= 0.9


def test_save_load_roundtrip(tmp_path):
    model = make_model()
    before = model.predict("beta beta request item")
    model.save(tmp_path / "m")
    reloaded = IntentModel.load(tmp_path / "m")
    after = reloaded.predict("beta beta request item")

    assert before.intent == after.intent
    assert np.isclose(before.score, after.score, atol=1e-5)
    assert reloaded.reranker is None
