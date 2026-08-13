"""Bad input should fail with a message about intents, not about solvers."""

from __future__ import annotations

import pytest

from tinyintent import Example, IntentModel
from tinyintent.encoder import HashingEncoder

from test_model import make_data, make_model  # tests/ is on sys.path


def fit(examples):
    return IntentModel._fit_base(examples, HashingEncoder(dim=256))


def test_single_intent_names_the_problem():
    examples = [Example(f"alpha {i}", "a") for i in range(5)]
    with pytest.raises(ValueError, match="only one in-scope intent"):
        fit(examples)


def test_all_oos_names_the_problem():
    examples = [Example(f"whatever {i}", "oos") for i in range(5)]
    with pytest.raises(ValueError, match="every example is labelled"):
        fit(examples)


def test_no_examples():
    with pytest.raises(ValueError, match="no training examples"):
        fit([])


@pytest.mark.parametrize("blank", ["", "   ", "\n", "\t "])
def test_blank_input_is_rejected_rather_than_answered(blank):
    model = make_model()
    with pytest.raises(ValueError, match="empty or whitespace-only"):
        model.predict(blank)
    with pytest.raises(ValueError, match="empty or whitespace-only"):
        model.classify(blank)


def test_blank_input_in_a_batch_reports_its_position():
    model = make_model()
    with pytest.raises(ValueError, match="position 1"):
        model.predict_batch(["alpha alpha request item", "  "])


def test_two_intents_still_train():
    """The guard must not reject legitimately small datasets."""
    examples = ([Example(f"alpha {i}", "a") for i in range(5)]
                + [Example(f"beta {i}", "b") for i in range(5)])
    model = fit(examples)
    assert model.label_names == ["a", "b"]


def test_oos_only_alongside_intents_is_fine():
    model = fit(make_data(n=6))
    assert model.label_names == ["a", "b", "c"]
