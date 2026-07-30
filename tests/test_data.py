from __future__ import annotations

from collections import Counter

from tinyintent.data import Example, from_fewshot, labels_of, split


def test_from_fewshot_and_labels():
    data = from_fewshot({"a": ["x", "y"], "b": ["z"], "oos": ["q"]})
    assert len(data) == 4
    assert labels_of(data) == ["a", "b"]              # oos excluded
    assert labels_of(data, include_oos=True) == ["a", "b", "oos"]


def test_split_is_stratified_and_disjoint():
    data = [Example(f"t{i}", "a") for i in range(10)] + [
        Example(f"s{i}", "b") for i in range(10)
    ]
    train, test = split(data, test_frac=0.3, seed=0)

    assert len(train) + len(test) == len(data)
    train_texts = {e.text for e in train}
    assert not (train_texts & {e.text for e in test})   # disjoint

    for label in ("a", "b"):
        assert any(e.label == label for e in train)
        assert any(e.label == label for e in test)
