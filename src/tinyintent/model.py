from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from tinyintent.data import OOS_LABEL, Example, labels_of
from tinyintent.encoder import Encoder, SentenceEncoder, make_encoder
from tinyintent.explain import nearest_example
from tinyintent.metrics import Report, score_predictions
from tinyintent.scorer import ExemplarScorer, load_scorer, make_scorer


@dataclass
class Prediction:
    """The outcome of classifying one utterance.

    ``intent`` is the single best intent (the model always decides).
    ``score`` is that intent's score, ``ranking`` lists every intent by
    score descending, and ``explanation`` is the nearest labelled example.
    """

    intent: str
    score: float
    ranking: list[tuple[str, float]] = field(default_factory=list)
    explanation: dict | None = None


class IntentModel:
    """A small, portable top-1 intent classifier.

    Frozen-encoder embeddings scored by a light classifier head (linear by
    default, or nearest-exemplar), returning the single best intent for
    every input. Train with :meth:`fit`, then :meth:`classify` or
    :meth:`predict`. Out-of-scope examples (label ``oos``) are never a
    class; they are ignored at fit time.
    """

    def __init__(self, encoder: Encoder, scorer: ExemplarScorer, label_names: list[str]):
        self.encoder = encoder
        self.scorer = scorer
        self.label_names = label_names
        self._train_vectors: np.ndarray | None = None
        self._train_y: np.ndarray | None = None
        self._train_texts: list[str] = []

    def _embed(self, texts: list[str]) -> np.ndarray:
        return self.encoder.encode(texts)

    # -- training -----------------------------------------------------------

    @classmethod
    def fit(
        cls,
        examples: list[Example],
        encoder: Encoder | None = None,
        classifier: str = "linear",
    ) -> "IntentModel":
        encoder = encoder or SentenceEncoder()
        label_names = labels_of(examples, include_oos=False)
        index = {label: i for i, label in enumerate(label_names)}

        in_scope = [ex for ex in examples if ex.label != OOS_LABEL]
        texts = [ex.text for ex in in_scope]
        y = np.array([index[ex.label] for ex in in_scope], dtype=np.int64)

        vectors = encoder.encode(texts)
        scorer = make_scorer(classifier)
        scorer.fit(vectors, y, len(label_names))

        model = cls(encoder, scorer, label_names)
        model._train_vectors = vectors
        model._train_y = y
        model._train_texts = texts
        return model

    # -- inference ----------------------------------------------------------

    def classify(self, text: str) -> str:
        return self.classify_batch([text])[0]

    def classify_batch(self, texts: list[str]) -> list[str]:
        """Return the single best intent for each text."""

        scores = self.scorer.scores(self._embed(texts))
        return [self.label_names[int(i)] for i in scores.argmax(axis=1)]

    def predict(self, text: str) -> Prediction:
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: list[str]) -> list[Prediction]:
        vectors = self._embed(texts)
        scores = self.scorer.scores(vectors)
        results: list[Prediction] = []

        for row in range(len(texts)):
            p = scores[row]
            order = np.argsort(p)[::-1]
            top_idx = int(order[0])
            ranking = [(self.label_names[int(i)], float(p[i])) for i in order]
            explanation = nearest_example(
                vectors[row], top_idx,
                self._train_vectors, self._train_y, self._train_texts,
            )
            results.append(
                Prediction(self.label_names[top_idx], float(p[top_idx]), ranking, explanation)
            )
        return results

    def accuracy(self, examples: list[Example]) -> float:
        """Top-1 accuracy on the in-scope examples."""

        in_scope = [ex for ex in examples if ex.label != OOS_LABEL]
        if not in_scope:
            return 0.0
        preds = self.classify_batch([ex.text for ex in in_scope])
        correct = sum(p == ex.label for p, ex in zip(preds, in_scope, strict=True))
        return correct / len(in_scope)

    def evaluate(self, examples: list[Example]) -> Report:
        in_scope = [ex for ex in examples if ex.label != OOS_LABEL]
        preds = self.classify_batch([ex.text for ex in in_scope])
        return score_predictions([ex.label for ex in in_scope], preds)

    # -- persistence --------------------------------------------------------

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.scorer.save(directory / "scorer")
        np.savez(directory / "train.npz", vectors=self._train_vectors, y=self._train_y)
        (directory / "texts.json").write_text(
            json.dumps(self._train_texts), encoding="utf-8"
        )
        (directory / "config.json").write_text(
            json.dumps(
                {
                    "encoder": self.encoder.spec(),
                    "label_names": self.label_names,
                    "classifier": self.scorer.name,
                }
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: str | Path) -> "IntentModel":
        directory = Path(directory)
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))

        model = cls(
            make_encoder(config["encoder"]),
            load_scorer(config.get("classifier", "linear"), directory / "scorer"),
            config["label_names"],
        )
        train = np.load(directory / "train.npz")
        model._train_vectors = train["vectors"].astype(np.float32)
        model._train_y = train["y"].astype(np.int64)
        model._train_texts = json.loads(
            (directory / "texts.json").read_text(encoding="utf-8")
        )
        return model
