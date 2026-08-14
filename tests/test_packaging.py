"""Guards against shipping a package that cannot run its own quickstart.

0.1.0 was published with `datasets` in the dev dependency group and
`accelerate` nowhere at all. CI stayed green because `uv sync` installs dev
groups and no test ever called the real `fit()`, so the ImportError only
appeared for people installing from PyPI. These tests fail on the metadata
alone, without downloading a model.
"""

from __future__ import annotations

import importlib.util
from importlib.metadata import requires

import pytest

# Third-party modules the training path imports at runtime. `datasets` and
# `accelerate` are not imported by tinyintent directly -- sentence-transformers
# reaches for them once CrossEncoder.fit() routes into CrossEncoderTrainer --
# but a user who lacks them cannot call fit(), so they are our dependencies.
TRAINING_RUNTIME_MODULES = ["numpy", "sklearn", "sentence_transformers", "torch",
                            "datasets", "accelerate"]

DECLARED_RUNTIME_DISTRIBUTIONS = ["numpy", "scikit-learn", "sentence-transformers",
                                  "datasets", "accelerate"]


def _declared_names() -> set[str]:
    """Runtime dependency names from installed metadata, extras excluded."""
    names = set()
    for spec in requires("tinyintent") or []:
        if "extra ==" in spec:
            continue
        name = spec.split(";")[0].strip()
        for delimiter in ("<", ">", "=", "!", "~", "[", " "):
            name = name.split(delimiter)[0]
        if name:
            names.add(name.lower())
    return names


@pytest.mark.parametrize("distribution", DECLARED_RUNTIME_DISTRIBUTIONS)
def test_training_dependency_is_declared_at_runtime(distribution):
    """A dev-group-only training dependency is invisible to PyPI users."""
    assert distribution.lower() in _declared_names(), (
        f"{distribution} is needed by IntentModel.fit() but is not a runtime "
        "dependency; a clean `pip install tinyintent` will raise ImportError"
    )


@pytest.mark.parametrize("module", TRAINING_RUNTIME_MODULES)
def test_training_module_is_importable(module):
    assert importlib.util.find_spec(module) is not None, (
        f"{module} is missing from this environment, so fit() cannot run"
    )


def test_sentence_transformers_is_upper_bounded():
    """fit() rides a deprecated ST v2 shim; a new major must be tested first."""
    spec = next(s for s in requires("tinyintent") if s.startswith("sentence-transformers"))
    assert "<" in spec, (
        "sentence-transformers needs an upper bound: fit() depends on the "
        "deprecated CrossEncoder.fit(train_dataloader=...) compatibility path"
    )
