"""Device selection reaches the transformer models, and is not baked into the artifact."""

from __future__ import annotations

import numpy as np
import pytest

from tinyintent.cli import build_parser
from tinyintent.encoder import HashingEncoder, make_encoder


class RecordingEncoder:
    """Stands in for SentenceEncoder to capture what device it was given."""

    seen: list[str | None] = []

    def __init__(self, model_name="stub", device=None):
        RecordingEncoder.seen.append(device)
        self.model_name = model_name
        self.device = device
        self.dim = 8

    def encode(self, texts):
        return np.zeros((len(texts), self.dim), dtype=np.float32)

    def spec(self):
        return {"kind": "sentence-transformers", "model": self.model_name}


@pytest.fixture(autouse=True)
def _reset():
    RecordingEncoder.seen = []


def test_make_encoder_forwards_device(monkeypatch):
    monkeypatch.setattr("tinyintent.encoder.SentenceEncoder", RecordingEncoder)
    make_encoder({"kind": "sentence-transformers", "model": "m"}, device="cpu")
    assert RecordingEncoder.seen == ["cpu"]


def test_make_encoder_defaults_to_auto(monkeypatch):
    monkeypatch.setattr("tinyintent.encoder.SentenceEncoder", RecordingEncoder)
    make_encoder({"kind": "sentence-transformers", "model": "m"})
    assert RecordingEncoder.seen == [None]


def test_device_is_not_persisted_in_the_spec():
    """A model trained on a GPU box must load on a CPU-only one."""
    encoder = RecordingEncoder(device="cuda")
    assert "device" not in encoder.spec()


def test_hashing_encoder_ignores_device():
    """The offline encoder has no device; make_encoder must not choke."""
    encoder = make_encoder({"kind": "hashing", "dim": 64}, device="cuda")
    assert isinstance(encoder, HashingEncoder)
    assert encoder.encode(["a b"]).shape == (1, 64)


@pytest.mark.parametrize("command", [
    ["train", "--data", "d.jsonl", "--out", "m", "--device", "cuda"],
    ["evaluate", "--model", "m", "--data", "d.jsonl", "--device", "cpu"],
    ["predict", "--model", "m", "--device", "mps", "hello"],
])
def test_cli_accepts_device(command):
    args = build_parser().parse_args(command)
    assert args.device in {"cuda", "cpu", "mps"}


def test_cli_device_defaults_to_none():
    args = build_parser().parse_args(["predict", "--model", "m", "hello"])
    assert args.device is None
