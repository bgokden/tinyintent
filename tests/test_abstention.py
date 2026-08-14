"""Abstention: `confidence`, the fitted `oos` threshold, and `abstain`.

Background: `score` is a softmax over row-standardised values across the
reranked top-k. Standardising drops the magnitude and the softmax forces the
candidates to sum to 1, so out-of-scope input still scores high -- measured at
0.892 for "what time do you close on sundays" on an 8-intent support model.
`confidence` is the linear head's unnormalised probability and is what a
threshold can actually use.
"""

from __future__ import annotations

import numpy as np

from tinyintent import Example, IntentModel
from tinyintent.encoder import HashingEncoder

from test_model import make_data  # tests/ is on sys.path, not a package


class StubReranker:
    """Reranks by reversing stage-1, so `score` and `confidence` must diverge.

    Reports a flat cross-encoder match (0.0 -> sigmoid 0.5) for every label, so
    the blend scales stage-1 by a constant and stays proportional to it.
    """

    CE_LOGIT = 0.0

    def raw_ce_by_label(self, query_texts, stage1, exclude_self=False):
        return np.full_like(stage1, self.CE_LOGIT, dtype=float)

    def rerank_with_ce(self, query_texts, stage1):
        flipped = stage1[:, ::-1]
        normalised = flipped / flipped.sum(axis=1, keepdims=True)
        return normalised, self.raw_ce_by_label(query_texts, stage1)

    def rerank_scores(self, query_texts, stage1):
        return self.rerank_with_ce(query_texts, stage1)[0]

    def save(self, directory):
        import json
        from pathlib import Path

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "stub.json").write_text(json.dumps({"stub": True}))


def fitted_model(dim=1024):
    data = make_data(n=14)
    model = IntentModel._fit_base(data, HashingEncoder(dim=dim))
    model.fit_oos_threshold(data)
    return model, data


def test_confidence_is_unnormalised_not_the_softmax_score():
    model, _ = fitted_model()
    model.reranker = StubReranker()

    text = "alpha alpha request item"
    pred = model.predict(text)
    stage1 = model.scorer.scores(model.encoder.encode([text]))[0]
    index = model.label_names.index(pred.intent)
    expected = stage1[index] / (1.0 + np.exp(-StubReranker.CE_LOGIT))

    assert np.isclose(pred.confidence, expected), (
        "confidence must be the head's probability blended with the raw CE "
        "match, never the renormalised score"
    )
    assert not np.isclose(pred.score, pred.confidence), (
        "this stub reranker renormalises, so score and confidence must differ"
    )


class StubNormalisingReranker(StubReranker):
    """Keeps stage-1's ordering but renormalises, like the real reranker."""

    def rerank_with_ce(self, query_texts, stage1):
        top_two = np.sort(stage1, axis=1)[:, -2:]
        keep = stage1 >= top_two[:, :1]                  # top-2 candidates only
        normalised = np.where(keep, stage1, 0.0)
        normalised = normalised / normalised.sum(axis=1, keepdims=True)
        return normalised, self.raw_ce_by_label(query_texts, stage1)


def test_renormalising_the_ranking_does_not_move_confidence():
    """The bug in one line.

    Renormalising over a candidate subset inflates `score` for input that fits
    nothing -- the whole reason out-of-scope text scored 0.89. `confidence` is
    computed before that step, so it must not move.
    """
    model, _ = fitted_model()
    texts = ["alpha alpha request item", "zzz1 unrelated foreign token 1"]

    baseline = model.predict_batch(texts)
    model.reranker = StubNormalisingReranker()
    renormalised = model.predict_batch(texts)

    assert [p.intent for p in baseline] == [p.intent for p in renormalised]
    # score is dragged up by the normalisation; confidence is not.
    assert all(after.score > before.score
               for before, after in zip(baseline, renormalised))
    np.testing.assert_allclose(
        [p.confidence * 0.5 for p in baseline],   # stub's flat sigmoid(0) factor
        [p.confidence for p in renormalised],
        atol=1e-6,
    )


def test_score_still_sums_to_one_over_candidates():
    """`score` semantics are unchanged -- existing callers must be unaffected."""
    model, _ = fitted_model()
    model.reranker = StubReranker()
    pred = model.predict("beta beta request item")
    assert np.isclose(sum(value for _, value in pred.ranking), 1.0)
    assert pred.ranking[0][0] == pred.intent


def test_threshold_is_fitted_from_oos_examples():
    model, _ = fitted_model()
    assert model.oos_threshold is not None
    assert 0.0 < model.oos_threshold <= 1.0


def test_no_oos_examples_means_no_threshold_and_no_abstention():
    data = [ex for ex in make_data(n=14) if ex.label != "oos"]
    model = IntentModel._fit_base(data, HashingEncoder(dim=1024))
    assert model.fit_oos_threshold(data) is None
    assert model.oos_threshold is None
    assert model.predict("completely unrelated banana vocabulary").abstain is False


def test_abstains_on_out_of_scope_and_commits_on_in_scope():
    model, data = fitted_model()

    in_scope = model.predict("alpha alpha request item")
    assert in_scope.intent == "a"
    assert in_scope.abstain is False

    rejected = model.oos_rejection_rate(data)
    assert rejected is not None and rejected >= 0.8, (
        f"threshold rejects only {rejected:.0%} of oos examples"
    )


def test_threshold_is_fitted_out_of_fold_not_on_memorised_confidences():
    """The trap this guards against.

    Scoring training text with the head that was fitted on it gives near-1.0
    confidences. A threshold fitted against those lands far above where real
    traffic sits -- the first version of this feature rejected 44% of genuine
    held-out utterances. Cross-fitted positives must sit clearly below the
    in-sample ones, and the threshold must sit below them too.
    """
    model, _ = fitted_model()

    in_sample = model.scorer.scores(model._train_vectors).max(axis=1)
    out_of_fold = model._out_of_fold_confidence()

    assert out_of_fold.mean() < in_sample.mean(), (
        "cross-fitted confidences must be less optimistic than in-sample ones"
    )
    assert model.oos_threshold < float(np.quantile(out_of_fold, 0.5)), (
        "threshold sits above the median unseen in-scope confidence, so it "
        "would abstain on more than half of genuine traffic"
    )


def test_predict_still_returns_an_intent_when_abstaining():
    """Abstention is advisory: the caller decides, the label is still there."""
    model, _ = fitted_model()
    pred = model.predict("zzz999 unrelated foreign token 999")
    assert pred.intent in model.label_names
    assert pred.ranking


def test_threshold_survives_save_load(tmp_path):
    model, _ = fitted_model()
    model.save(tmp_path / "m")
    reloaded = IntentModel.load(tmp_path / "m")
    assert np.isclose(reloaded.oos_threshold, model.oos_threshold)


def test_model_saved_before_abstention_loads_with_none(tmp_path):
    """Older artefacts have no oos_threshold key; they must still load."""
    import json

    model, _ = fitted_model()
    model.save(tmp_path / "m")
    config_path = tmp_path / "m" / "config.json"
    config = json.loads(config_path.read_text())
    del config["oos_threshold"]           # pre-abstention artifacts have
    del config["oos_threshold_stage1"]    # neither cut
    config_path.write_text(json.dumps(config))

    reloaded = IntentModel.load(tmp_path / "m")
    assert reloaded.oos_threshold is None
    assert reloaded.predict("alpha alpha request item").abstain is False


def test_evaluate_still_ignores_oos_but_rejection_rate_reports_it():
    model, data = fitted_model()
    report = model.evaluate(data)
    assert report.accuracy >= 0.9                     # in-scope only, as documented
    assert model.oos_rejection_rate(data) is not None  # the number evaluate() omits


def test_toggling_the_reranker_off_keeps_abstention_working():
    """The bug: one threshold cannot serve two confidence scales.

    `confidence` blends the head's probability with the cross-encoder match
    when a reranker is attached, and the blend is strictly smaller because it
    multiplies by a sigmoid. Applying the blended cut to bare stage-1
    confidences let every out-of-scope query through -- measured 95% rejection
    to 0% -- the moment `model.reranker = None` was set, which the docs
    recommend for latency.
    """
    model, data = fitted_model()
    model.reranker = StubReranker()          # halves confidence via sigmoid(0)
    model.fit_oos_threshold(data)

    reranked_rate = model.oos_rejection_rate(data)
    reranked_cut = model.oos_threshold

    model.reranker = None
    stage1_rate = model.oos_rejection_rate(data)

    assert reranked_cut != model.oos_threshold, (
        "the two confidence scales must not share one cut"
    )
    assert stage1_rate >= 0.8, (
        f"abstention collapsed to {stage1_rate:.0%} when the reranker was "
        "removed; it must keep working"
    )
    assert reranked_rate >= 0.8


def test_both_cuts_survive_save_load(tmp_path):
    model, data = fitted_model()
    model.reranker = StubReranker()
    model.fit_oos_threshold(data)
    model.save(tmp_path / "m")

    # The stub cannot be rebuilt by load(), so read the config back directly
    # and re-attach it; the point is that both cuts round-trip.
    import json
    config = json.loads((tmp_path / "m" / "config.json").read_text())
    config["reranker"] = False
    (tmp_path / "m" / "config.json").write_text(json.dumps(config))

    reloaded = IntentModel.load(tmp_path / "m")
    reloaded.reranker = StubReranker()
    assert np.isclose(reloaded.oos_threshold, model.oos_threshold)

    reloaded.reranker = None
    model.reranker = None
    assert np.isclose(reloaded.oos_threshold, model.oos_threshold)


def test_setting_oos_threshold_by_hand_applies_to_both_scales():
    model, _ = fitted_model()
    model.oos_threshold = 0.42
    assert model.oos_threshold == 0.42
    model.reranker = StubReranker()
    assert model.oos_threshold == 0.42
