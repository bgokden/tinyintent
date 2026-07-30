from __future__ import annotations

import numpy as np
import pytest

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


def make_model(risk=0.1, dim=1024):
    data = make_data(n=14)
    model = IntentModel.fit_calibrate(
        data, encoder=HashingEncoder(dim=dim), risk=risk, calibrate_frac=0.4
    )
    return model, data


def test_uncalibrated_fires_top1():
    model = IntentModel.fit(make_data(), encoder=HashingEncoder(dim=512))
    pred = model.predict("alpha alpha request")
    assert pred.decision == "fire"
    assert pred.intent == "a"


def test_labels_exclude_oos():
    model = IntentModel.fit(make_data(), encoder=HashingEncoder(dim=512))
    assert model.label_names == ["a", "b", "c"]


def test_fires_on_clear_in_scope():
    model, _ = make_model()
    pred = model.predict("alpha alpha request item extra")
    assert "a" in {label for label, _ in pred.set_}
    assert pred.intent == "a"


def test_abstains_on_out_of_scope():
    model, _ = make_model()
    pred = model.predict("completely different banana vocabulary here")
    assert pred.decision == "abstain"
    assert pred.set_ == []


def test_conformal_coverage_holds():
    model, data = make_model(risk=0.1)
    report = model.evaluate(data)
    assert report.coverage >= 0.85          # target is 1 - risk
    assert report.oos_false_fire <= 0.15


def test_save_load_roundtrip(tmp_path):
    model, _ = make_model()
    before = model.predict("beta beta request item")
    model.save(tmp_path / "m")
    reloaded = IntentModel.load(tmp_path / "m")
    after = reloaded.predict("beta beta request item")

    assert before.decision == after.decision
    assert before.intent == after.intent
    assert np.isclose(before.top[1], after.top[1], atol=1e-5)
    assert reloaded.conformal.q == pytest.approx(model.conformal.q)
