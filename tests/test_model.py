from __future__ import annotations

import numpy as np
import pytest

from tinyintent import Example, HashingEncoder, IntentModel


def make_data():
    # Keyword-separable so the offline hashing encoder can distinguish them.
    rows = []
    for word, label in [("alpha", "a"), ("beta", "b"), ("gamma", "c")]:
        for i in range(8):
            rows.append(Example(f"{word} request number {i}", label))
    for i in range(6):
        rows.append(Example(f"noise{i} random unrelated {i}", "oos"))
    return rows


@pytest.mark.parametrize("head", ["prototype", "logreg", "mlp"])
def test_fit_predict_each_head(head):
    model = IntentModel.fit(make_data(), head=head, encoder=HashingEncoder(dim=512))
    # Uncalibrated model fires on everything (threshold = -inf).
    pred = model.predict("alpha please do this")
    assert pred.intent == "a"
    assert pred.alternatives[0][0] == "a"


def test_scores_shape_and_labels():
    model = IntentModel.fit(make_data(), head="prototype", encoder=HashingEncoder(dim=256))
    assert model.label_names == ["a", "b", "c"]        # oos not a class
    scores = model.head.scores(model.encoder.encode(["alpha thing", "beta thing"]))
    assert scores.shape == (2, 3)


def test_calibration_abstains_on_oos():
    data = make_data()
    model = IntentModel.fit(data, head="prototype", encoder=HashingEncoder(dim=512))
    model.calibrate(data, max_false_fire=0.0)
    # An out-of-scope-style utterance should be declined.
    assert model.predict("totally unrelated banana sentence").intent is None
    # A clearly in-scope utterance still fires.
    assert model.predict("alpha request number 3").intent == "a"


def test_uncalibrated_never_abstains():
    model = IntentModel.fit(make_data(), head="prototype", encoder=HashingEncoder(dim=512))
    # Before calibration the threshold is -inf, so it always commits.
    assert model.predict("totally unrelated banana sentence").intent is not None


def test_save_load_roundtrip(tmp_path):
    model = IntentModel.fit(make_data(), head="logreg", encoder=HashingEncoder(dim=512))
    model.calibrate(make_data(), max_false_fire=0.1)

    before = model.predict("beta request now")
    model.save(tmp_path / "m")
    reloaded = IntentModel.load(tmp_path / "m")
    after = reloaded.predict("beta request now")

    assert before.intent == after.intent
    assert np.isclose(before.score, after.score, atol=1e-5)
    assert reloaded.policy.threshold == model.policy.threshold
