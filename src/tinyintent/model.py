from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from tinyintent.calibrate import Policy, calibrate
from tinyintent.data import OOS_LABEL, Example, labels_of
from tinyintent.encoder import Encoder, SentenceEncoder, make_encoder
from tinyintent.explain import nearest_example
from tinyintent.heads.base import Head, load_head, make_head
from tinyintent.metrics import Report, score_predictions


@dataclass
class Prediction:
    intent: str | None            # None means the model abstained
    score: float
    abstained: bool
    alternatives: list[tuple[str, float]] = field(default_factory=list)
    explanation: dict | None = None


class IntentModel:
    """Frozen encoder + pluggable head + calibrated abstention.

    Train with :meth:`fit`, tune the abstain threshold with
    :meth:`calibrate`, then :meth:`predict`. Out-of-scope examples (label
    ``oos``) are never learned as a class; they are only used to calibrate
    when the model should decline to answer.
    """

    def __init__(self, encoder: Encoder, head: Head, label_names: list[str]) -> None:
        self.encoder = encoder
        self.head = head
        self.label_names = label_names
        self.policy = Policy(threshold=float("-inf"), margin=0.0)
        self.top_k = 3
        self._train_vectors: np.ndarray | None = None
        self._train_y: np.ndarray | None = None
        self._train_texts: list[str] = []

    # -- training -----------------------------------------------------------

    @classmethod
    def fit(
        cls,
        examples: list[Example],
        head: str = "prototype",
        encoder: Encoder | None = None,
    ) -> "IntentModel":
        encoder = encoder or SentenceEncoder()
        label_names = labels_of(examples, include_oos=False)
        index = {label: i for i, label in enumerate(label_names)}

        in_scope = [ex for ex in examples if ex.label != OOS_LABEL]
        texts = [ex.text for ex in in_scope]
        y = np.array([index[ex.label] for ex in in_scope], dtype=np.int64)
        vectors = encoder.encode(texts)

        head_impl = make_head(head)
        head_impl.fit(vectors, y, len(label_names))

        model = cls(encoder, head_impl, label_names)
        model._train_vectors = vectors
        model._train_y = y
        model._train_texts = texts
        return model

    # -- inference ----------------------------------------------------------

    def _score(self, texts: list[str]) -> np.ndarray:
        return self.head.scores(self.encoder.encode(texts))

    def predict(self, text: str) -> Prediction:
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: list[str]) -> list[Prediction]:
        scores = self._score(texts)
        vectors = self.encoder.encode(texts)
        results: list[Prediction] = []

        for row in range(len(texts)):
            s = scores[row]
            order = np.argsort(s)[::-1]
            top = int(order[0])
            s1 = float(s[top])
            s2 = float(s[order[1]]) if len(order) > 1 else float("-inf")
            fire = s1 >= self.policy.threshold and (s1 - s2) >= self.policy.margin

            alts = [(self.label_names[int(i)], float(s[i])) for i in order[: self.top_k]]
            explanation = nearest_example(
                vectors[row], top, self._train_vectors, self._train_y, self._train_texts
            )
            results.append(
                Prediction(
                    intent=self.label_names[top] if fire else None,
                    score=s1,
                    abstained=not fire,
                    alternatives=alts,
                    explanation=explanation,
                )
            )
        return results

    # -- calibration and evaluation ----------------------------------------

    def calibrate(self, examples: list[Example], max_false_fire: float = 0.02) -> Policy:
        index = {label: i for i, label in enumerate(self.label_names)}
        scores = self._score([ex.text for ex in examples])
        y_true = np.array(
            [index.get(ex.label, -1) for ex in examples], dtype=np.int64
        )
        self.policy = calibrate(scores, y_true, max_false_fire=max_false_fire)
        return self.policy

    def evaluate(self, examples: list[Example]) -> Report:
        preds = self.predict_batch([ex.text for ex in examples])
        return score_predictions(
            [ex.label for ex in examples],
            [p.intent for p in preds],
            OOS_LABEL,
        )

    # -- persistence --------------------------------------------------------

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        self.head.save(directory / "head")
        np.savez(
            directory / "train.npz",
            vectors=self._train_vectors,
            y=self._train_y,
        )
        (directory / "train_texts.json").write_text(
            json.dumps(self._train_texts), encoding="utf-8"
        )
        (directory / "config.json").write_text(
            json.dumps(
                {
                    "encoder": self.encoder.spec(),
                    "head": self.head.name,
                    "label_names": self.label_names,
                    "threshold": self.policy.threshold,
                    "margin": self.policy.margin,
                    "top_k": self.top_k,
                }
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: str | Path) -> "IntentModel":
        directory = Path(directory)
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))

        encoder = make_encoder(config["encoder"])
        head = load_head(config["head"], directory / "head")
        model = cls(encoder, head, config["label_names"])
        model.policy = Policy(threshold=config["threshold"], margin=config["margin"])
        model.top_k = config.get("top_k", 3)

        data = np.load(directory / "train.npz")
        model._train_vectors = data["vectors"].astype(np.float32)
        model._train_y = data["y"].astype(np.int64)
        model._train_texts = json.loads(
            (directory / "train_texts.json").read_text(encoding="utf-8")
        )
        return model
