from __future__ import annotations

import numpy as np

from tinyintent import Example, HashingEncoder, IntentModel


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


def make_model(dim=1024, classifier="exemplar"):
    data = make_data(n=14)
    model = IntentModel.fit(data, encoder=HashingEncoder(dim=dim), classifier=classifier)
    return model, data


def test_labels_exclude_oos():
    model = IntentModel.fit(make_data(), encoder=HashingEncoder(dim=512))
    assert model.label_names == ["a", "b", "c"]


def test_classify_decides_top1():
    model, _ = make_model()
    assert model.classify("alpha alpha request item") == "a"
    assert model.classify("beta beta request item") == "b"


def test_always_decides_never_oos():
    model, _ = make_model()
    # Out-of-scope input still gets a best-guess intent, never "oos".
    pred = model.predict("completely different banana vocabulary here")
    assert pred.intent in model.label_names


def test_predict_ranking_and_explanation():
    model, _ = make_model()
    pred = model.predict("alpha alpha request item extra")
    assert pred.intent == "a"
    assert pred.ranking[0][0] == "a"
    assert len(pred.ranking) == len(model.label_names)
    assert pred.explanation is not None


def test_linear_head_accuracy():
    model, data = make_model(classifier="linear")
    assert model.accuracy(data) >= 0.9


def test_save_load_roundtrip(tmp_path):
    model, _ = make_model()
    before = model.predict("beta beta request item")
    model.save(tmp_path / "m")
    reloaded = IntentModel.load(tmp_path / "m")
    after = reloaded.predict("beta beta request item")

    assert before.intent == after.intent
    assert np.isclose(before.score, after.score, atol=1e-5)
