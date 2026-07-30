from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from tinyintent.aps import Aps
from tinyintent.conformal import Conformal
from tinyintent.data import OOS_LABEL, Example, labels_of, split
from tinyintent.encoder import Encoder, SentenceEncoder, make_encoder
from tinyintent.explain import nearest_example
from tinyintent.metrics import Report, score_predictions
from tinyintent.scorer import ExemplarScorer
from tinyintent.transform import IdentityTransform, load_transform, make_transform


@dataclass
class Prediction:
    """The outcome of routing one utterance.

    ``decision`` is ``fire`` (a single confident intent), ``abstain`` (the
    prediction set is empty), or ``ambiguous`` (the set holds several
    intents). ``intent`` is set only when the decision is ``fire``. ``set_``
    is the conformal prediction set as (label, probability) pairs, and
    ``top`` is the single highest-probability label regardless of the set.
    """

    decision: str
    intent: str | None
    top: tuple[str, float]
    set_: list[tuple[str, float]] = field(default_factory=list)
    explanation: dict | None = None


class IntentModel:
    """A portable selective intent classifier.

    Frozen-encoder embeddings scored by :class:`ExemplarScorer`, with a
    :class:`Conformal` layer that turns per-class similarities into
    risk-controlled prediction sets. Train with :meth:`fit`, set the risk with
    :meth:`calibrate`, then :meth:`predict`. Out-of-scope examples (label
    ``oos``) are never a class; they only help measure false firing.
    """

    def __init__(self, encoder: Encoder, scorer: ExemplarScorer, label_names: list[str]):
        self.encoder = encoder
        self.scorer = scorer
        self.label_names = label_names
        self.transform = IdentityTransform()
        self.policy: Conformal | Aps | None = None
        self._train_texts: list[str] = []

    def _embed(self, texts: list[str]) -> np.ndarray:
        return self.transform.apply(self.encoder.encode(texts))

    # -- training -----------------------------------------------------------

    @classmethod
    def fit(
        cls,
        examples: list[Example],
        encoder: Encoder | None = None,
        transform: str = "none",
    ) -> "IntentModel":
        encoder = encoder or SentenceEncoder()
        label_names = labels_of(examples, include_oos=False)
        index = {label: i for i, label in enumerate(label_names)}

        in_scope = [ex for ex in examples if ex.label != OOS_LABEL]
        texts = [ex.text for ex in in_scope]
        y = np.array([index[ex.label] for ex in in_scope], dtype=np.int64)

        base = encoder.encode(texts)
        projector = make_transform(transform)
        projector.fit(base, y)
        vectors = projector.apply(base)

        scorer = ExemplarScorer()
        scorer.fit(vectors, y, len(label_names))

        model = cls(encoder, scorer, label_names)
        model.transform = projector
        model._train_texts = texts
        return model

    @classmethod
    def fit_calibrate(
        cls,
        examples: list[Example],
        encoder: Encoder | None = None,
        risk: float = 0.1,
        method: str = "aps",
        transform: str = "none",
        calibrate_frac: float = 0.25,
        seed: int = 0,
    ) -> "IntentModel":
        """Fit and calibrate in one call using an internal held-out split.

        Calibration must not reuse the fitted exemplars (they self-match at
        similarity 1.0 and collapse the threshold), so this splits
        ``examples`` stratified by label before fitting.
        """

        fit_set, cal_set = split(examples, test_frac=calibrate_frac, seed=seed)
        model = cls.fit(fit_set, encoder=encoder, transform=transform)
        model.calibrate(cal_set, risk=risk, method=method)
        return model

    # -- calibration --------------------------------------------------------

    def calibrate(
        self,
        examples: list[Example],
        risk: float = 0.1,
        method: str = "aps",
        mondrian: bool = False,
        use_oos: bool = True,
        reg_lambda: float = 0.0,
    ) -> Conformal | Aps:
        """Calibrate the decision policy at the given risk.

        ``risk`` (alpha) is the allowed chance of dropping the true intent
        from the set on in-scope data. ``method`` is ``aps`` (two-stage
        gate + adaptive sets; smaller sets on close intents) or ``lac``
        (a single absolute-similarity threshold). If the calibration data
        contains ``oos`` examples and ``use_oos`` is set, they raise the
        abstain bar to reduce false firing on novel input.
        """

        index = {label: i for i, label in enumerate(self.label_names)}
        in_scope = [ex for ex in examples if ex.label != OOS_LABEL]
        vectors = self._embed([ex.text for ex in in_scope])
        y = np.array([index[ex.label] for ex in in_scope], dtype=np.int64)
        scores = self.scorer.scores(vectors)

        oos_scores = None
        if use_oos:
            oos = [ex for ex in examples if ex.label == OOS_LABEL]
            if oos:
                oos_scores = self.scorer.scores(self._embed([ex.text for ex in oos]))

        if method == "aps":
            self.policy = Aps.calibrate(
                scores, y, alpha=risk, oos_scores=oos_scores, reg_lambda=reg_lambda
            )
        elif method == "lac":
            self.policy = Conformal.calibrate(
                scores, y, alpha=risk, mondrian=mondrian, oos_scores=oos_scores
            )
        else:
            raise ValueError(f"unknown method: {method} (choose aps or lac)")
        return self.policy

    # -- inference ----------------------------------------------------------

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
            top = (self.label_names[top_idx], float(p[top_idx]))

            if self.policy is None:
                members = [top_idx]                     # uncalibrated: fire top-1
            else:
                mask = self.policy.prediction_set(p[None, :])[0]
                members = [int(i) for i in order if mask[i]]

            set_ = [(self.label_names[i], float(p[i])) for i in members]
            if len(members) == 1:
                decision, intent = "fire", self.label_names[members[0]]
            elif len(members) == 0:
                decision, intent = "abstain", None
            else:
                decision, intent = "ambiguous", None

            explanation = nearest_example(
                vectors[row], top_idx,
                self.scorer.vectors, self.scorer.exemplar_label, self._train_texts,
            )
            results.append(Prediction(decision, intent, top, set_, explanation))
        return results

    def evaluate(self, examples: list[Example]) -> Report:
        preds = self.predict_batch([ex.text for ex in examples])
        return score_predictions(
            [ex.label for ex in examples], preds, self.label_names, OOS_LABEL
        )

    # -- persistence --------------------------------------------------------

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.scorer.save(directory / "scorer")
        self.transform.save(directory / "transform")
        if self.policy is not None:
            self.policy.save(directory / "policy")
        (directory / "texts.json").write_text(
            json.dumps(self._train_texts), encoding="utf-8"
        )
        (directory / "config.json").write_text(
            json.dumps(
                {
                    "encoder": self.encoder.spec(),
                    "label_names": self.label_names,
                    "transform": self.transform.name,
                    "policy": self.policy.name if self.policy is not None else None,
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
            ExemplarScorer.load(directory / "scorer"),
            config["label_names"],
        )
        model.transform = load_transform(
            config.get("transform", "none"), directory / "transform"
        )
        policy_name = config.get("policy")
        if policy_name == "lac":
            model.policy = Conformal.load(directory / "policy")
        elif policy_name == "aps":
            model.policy = Aps.load(directory / "policy")
        model._train_texts = json.loads(
            (directory / "texts.json").read_text(encoding="utf-8")
        )
        return model
