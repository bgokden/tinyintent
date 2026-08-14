"""Saved-model compatibility in both directions."""

from __future__ import annotations

import json

import numpy as np
import pytest

from tinyintent import IntentModel
from tinyintent.model import FORMAT_VERSION

from test_model import make_model  # tests/ is on sys.path


def saved(tmp_path):
    model = make_model()
    model.save(tmp_path / "m")
    return model, tmp_path / "m"


def read_config(directory):
    return json.loads((directory / "config.json").read_text())


def write_config(directory, config):
    (directory / "config.json").write_text(json.dumps(config))


def test_save_stamps_the_format_version(tmp_path):
    _, directory = saved(tmp_path)
    assert read_config(directory)["format_version"] == FORMAT_VERSION


def test_unversioned_artifact_loads_as_version_1(tmp_path):
    """Models written before versioning must keep working."""
    model, directory = saved(tmp_path)
    config = read_config(directory)
    del config["format_version"]
    del config["oos_threshold"]          # also predates abstention
    write_config(directory, config)

    reloaded = IntentModel.load(directory)
    assert reloaded.oos_threshold is None
    assert reloaded.classify("alpha alpha request item") == "a"


def test_future_version_is_refused_with_a_useful_message(tmp_path):
    _, directory = saved(tmp_path)
    config = read_config(directory)
    config["format_version"] = FORMAT_VERSION + 1
    write_config(directory, config)

    with pytest.raises(ValueError, match="upgrade the package"):
        IntentModel.load(directory)


def test_roundtrip_preserves_predictions(tmp_path):
    model, directory = saved(tmp_path)
    texts = ["alpha alpha request item", "beta beta request item"]
    before = model.predict_batch(texts)
    after = IntentModel.load(directory).predict_batch(texts)

    assert [p.intent for p in before] == [p.intent for p in after]
    np.testing.assert_allclose([p.confidence for p in before],
                               [p.confidence for p in after], atol=1e-5)
