from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from tinyintent.aps import Aps
from tinyintent.conformal import Conformal
from tinyintent.gate import DecisiveGate
from tinyintent.data import OOS_LABEL, Example, labels_of, split
from tinyintent.encoder import Encoder, SentenceEncoder, make_encoder
from tinyintent.explain import nearest_example
from tinyintent.metrics import Report, score_predictions
from tinyintent.scorer import ExemplarScorer, load_scorer, make_scorer
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
        self.policy: Conformal | Aps | DecisiveGate | None = None
        self._train_vectors: np.ndarray | None = None
        self._train_y: np.ndarray | None = None
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
        classifier: str = "exemplar",
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

        scorer = make_scorer(classifier)
        scorer.fit(vectors, y, len(label_names))

        model = cls(encoder, scorer, label_names)
        model.transform = projector
        model._train_vectors = vectors
        model._train_y = y
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
        classifier: str = "exemplar",
        calibrate_frac: float = 0.25,
        seed: int = 0,
    ) -> "IntentModel":
        """Fit and calibrate in one call using an internal held-out split.

        Calibration must not reuse the fitted exemplars (they self-match at
        similarity 1.0 and collapse the threshold), so this splits
        ``examples`` stratified by label before fitting.
        """

        fit_set, cal_set = split(examples, test_frac=calibrate_frac, seed=seed)
        model = cls.fit(fit_set, encoder=encoder, transform=transform, classifier=classifier)
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
        reject_level: float = 0.1,
    ) -> Conformal | Aps | DecisiveGate:
        """Calibrate the decision policy at the given risk.

        ``method`` selects the decision behaviour:

        - ``aps`` (default): two-stage gate + adaptive prediction sets;
          safety-first, may return an ambiguous set to escalate.
        - ``lac``: a single absolute-similarity threshold.
        - ``gate``: decisive fire-top-1-or-reject with no ambiguous outcome,
          for systems with no fallback (tune with ``reject_level``).

        ``risk`` (alpha) is the conformal miss rate for ``aps``/``lac``. If
        the calibration data contains ``oos`` examples and ``use_oos`` is
        set, they raise the abstain/reject bar.
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
        elif method == "gate":
            self.policy = DecisiveGate.calibrate(
                scores, y, reject_level=reject_level, oos_scores=oos_scores
            )
        else:
            raise ValueError(f"unknown method: {method} (choose aps, lac, or gate)")
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
                self._train_vectors, self._train_y, self._train_texts,
            )
            results.append(Prediction(decision, intent, top, set_, explanation))
        return results

    def evaluate(self, examples: list[Example]) -> Report:
        preds = self.predict_batch([ex.text for ex in examples])
        return score_predictions(
            [ex.label for ex in examples], preds, self.label_names, OOS_LABEL
        )

    # -- always-decide top-1 (no policy, no abstain) ------------------------

    def classify(self, text: str) -> str:
        return self.classify_batch([text])[0]

    def classify_batch(self, texts: list[str]) -> list[str]:
        """Return the single best intent for each text, always deciding."""

        scores = self.scorer.scores(self._embed(texts))
        return [self.label_names[int(i)] for i in scores.argmax(axis=1)]

    def accuracy(self, examples: list[Example]) -> float:
        """Top-1 accuracy on the in-scope examples."""

        in_scope = [ex for ex in examples if ex.label != OOS_LABEL]
        if not in_scope:
            return 0.0
        preds = self.classify_batch([ex.text for ex in in_scope])
        correct = sum(p == ex.label for p, ex in zip(preds, in_scope, strict=True))
        return correct / len(in_scope)

    # -- persistence --------------------------------------------------------

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.scorer.save(directory / "scorer")
        self.transform.save(directory / "transform")
        if self.policy is not None:
            self.policy.save(directory / "policy")
        np.savez(directory / "train.npz", vectors=self._train_vectors, y=self._train_y)
        (directory / "texts.json").write_text(
            json.dumps(self._train_texts), encoding="utf-8"
        )
        (directory / "config.json").write_text(
            json.dumps(
                {
                    "encoder": self.encoder.spec(),
                    "label_names": self.label_names,
                    "transform": self.transform.name,
                    "classifier": self.scorer.name,
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
            load_scorer(config.get("classifier", "exemplar"), directory / "scorer"),
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
        elif policy_name == "gate":
            model.policy = DecisiveGate.load(directory / "policy")

        train = np.load(directory / "train.npz")
        model._train_vectors = train["vectors"].astype(np.float32)
        model._train_y = train["y"].astype(np.int64)
        model._train_texts = json.loads(
            (directory / "texts.json").read_text(encoding="utf-8")
        )
        return model
