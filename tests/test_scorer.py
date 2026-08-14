"""The hand-rolled head must match scikit-learn exactly.

Inference no longer goes through a LogisticRegression object, so the arithmetic
here has to reproduce `predict_proba` to the last decimal -- including the two
different paths sklearn uses: sigmoid for a binary fit, softmax for multinomial.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from tinyintent.scorer import LinearScorer


def data(n_classes: int, n=120, dim=32, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, n_classes, size=n)
    centres = rng.normal(size=(n_classes, dim)) * 3
    vectors = centres[y] + rng.normal(size=(n, dim))
    return vectors.astype(np.float32), y.astype(np.int64)


@pytest.mark.parametrize("n_classes", [2, 3, 7])
def test_matches_sklearn_predict_proba(n_classes):
    vectors, y = data(n_classes)
    scorer = LinearScorer()
    scorer.fit(vectors, y, n_classes)

    reference = LogisticRegression(C=scorer.C, max_iter=scorer.max_iter)
    reference.fit(vectors, y)

    ours = scorer.scores(vectors)
    theirs = np.zeros_like(ours)
    for col, label in enumerate(reference.classes_):
        theirs[:, int(label)] = reference.predict_proba(vectors)[:, col]

    np.testing.assert_allclose(ours, theirs, atol=1e-5)


@pytest.mark.parametrize("n_classes", [2, 5])
def test_probabilities_are_a_distribution(n_classes):
    vectors, y = data(n_classes)
    scorer = LinearScorer()
    scorer.fit(vectors, y, n_classes)
    out = scorer.scores(vectors)
    np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-5)
    assert (out >= 0).all()


@pytest.mark.parametrize("n_classes", [2, 4])
def test_save_load_is_exact(tmp_path, n_classes):
    vectors, y = data(n_classes)
    scorer = LinearScorer()
    scorer.fit(vectors, y, n_classes)
    scorer.save(tmp_path / "s")

    reloaded = LinearScorer.load(tmp_path / "s")
    np.testing.assert_allclose(scorer.scores(vectors), reloaded.scores(vectors),
                               atol=1e-6)


def test_inference_does_not_import_sklearn(tmp_path):
    """A loaded model must score without sklearn present at all."""
    import subprocess
    import sys

    vectors, y = data(4)
    scorer = LinearScorer()
    scorer.fit(vectors, y, 4)
    scorer.save(tmp_path / "s")
    np.save(tmp_path / "v.npy", vectors[:5])

    program = f"""
import sys
sys.modules['sklearn'] = None      # any sklearn import now raises
import numpy as np
from tinyintent.scorer import LinearScorer
s = LinearScorer.load({str(tmp_path / "s")!r})
out = s.scores(np.load({str(tmp_path / "v.npy")!r}))
assert out.shape == (5, 4), out.shape
print("ok")
"""
    result = subprocess.run([sys.executable, "-c", program],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
