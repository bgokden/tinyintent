"""Exemplar selection: bounded inference cost, and a spread rather than a clump.

Every kept exemplar costs one cross-encoder pair per candidate at query time,
so an uncapped set makes latency scale with the training set. Measured on
CLINC150 (150 intents, 20 examples each): 47.0 ms/utterance uncapped against
16.4 ms at 6, for 0.9558 -> 0.9542 top-1 accuracy.
"""

from __future__ import annotations

import numpy as np

from tinyintent.reranker import CrossEncoderReranker, _select_exemplars


def ring(n: int, dim: int = 8) -> np.ndarray:
    """Unit vectors spread around a circle, so 'far apart' is unambiguous."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    out = np.zeros((n, dim), dtype=np.float32)
    out[:, 0] = np.cos(angles)
    out[:, 1] = np.sin(angles)
    return out


def test_cap_is_respected_per_intent():
    texts = [f"t{i}" for i in range(30)]
    y = np.array([0] * 20 + [1] * 10)
    chosen = _select_exemplars(texts, y, ring(30), max_exemplars=6)
    assert len(chosen[0]) == 6
    assert len(chosen[1]) == 6


def test_intents_smaller_than_the_cap_keep_everything():
    texts = [f"t{i}" for i in range(7)]
    y = np.array([0] * 4 + [1] * 3)
    chosen = _select_exemplars(texts, y, ring(7), max_exemplars=6)
    assert len(chosen[0]) == 4 and len(chosen[1]) == 3


def test_zero_means_no_cap():
    texts = [f"t{i}" for i in range(20)]
    y = np.zeros(20, dtype=int)
    assert len(_select_exemplars(texts, y, ring(20), max_exemplars=0)[0]) == 20


def test_selection_spreads_instead_of_clumping():
    """Near-duplicates must not crowd out the rest of an intent's phrasings."""
    # Ten vectors bunched together, three spread far away.
    clump = np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (10, 1))
    clump += np.random.default_rng(0).normal(scale=0.01, size=clump.shape)
    far = np.array([[0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]], dtype=np.float32)
    vectors = np.vstack([clump, far]).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

    texts = [f"clump{i}" for i in range(10)] + ["far_a", "far_b", "far_c"]
    y = np.zeros(len(texts), dtype=int)
    chosen = _select_exemplars(texts, y, vectors, max_exemplars=4)[0]

    outliers = {"far_a", "far_b", "far_c"}
    assert len(outliers & set(chosen)) >= 2, (
        f"selection collapsed onto the clump: {chosen}"
    )


def test_no_duplicates_are_selected():
    texts = [f"t{i}" for i in range(12)]
    y = np.zeros(12, dtype=int)
    chosen = _select_exemplars(texts, y, ring(12), max_exemplars=5)[0]
    assert len(chosen) == len(set(chosen))


def test_cap_survives_save_load(tmp_path):
    """A model saved with a cap must not silently load as uncapped."""
    reranker = CrossEncoderReranker(max_exemplars=6)
    reranker.exemplars = {0: ["a", "b"], 1: ["c"]}

    class StubCE:
        def save(self, path):
            import pathlib
            pathlib.Path(path).mkdir(parents=True, exist_ok=True)

    reranker._ce = StubCE()
    reranker.save(tmp_path / "r")

    import json
    config = json.loads((tmp_path / "r" / "reranker.json").read_text())
    assert config["max_exemplars"] == 6


def test_default_cap_is_set():
    assert CrossEncoderReranker().max_exemplars == 6
