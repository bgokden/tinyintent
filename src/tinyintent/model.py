from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from tinyintent.data import OOS_LABEL, Example, labels_of
from tinyintent.encoder import Encoder, SentenceEncoder, make_encoder
from tinyintent.explain import nearest_example
from tinyintent.metrics import Report, score_predictions
from tinyintent.reranker import CrossEncoderReranker
from tinyintent.scorer import LinearScorer


def _blend_confidence(stage1: np.ndarray, ce_by_label: np.ndarray) -> np.ndarray:
    """Combine the linear head's probability with the cross-encoder's match.

    Two independent readings of "does this belong here": how much probability
    mass the head puts on a label, and how well the query matches that label's
    exemplars. Multiplying them means a query has to satisfy both, which is
    what separates out-of-scope input -- it typically clears one and fails the
    other.

    The gain is scale-dependent. On a held-out 8-intent set the product
    separates in-scope from out-of-scope at 0.994 AUROC against 0.973 for the
    head alone; on CLINC150's 150 intents and 1000 out-of-scope queries it is
    0.970 against 0.968, so the cross-encoder adds almost nothing there. It
    does not hurt at either scale, and it is free -- the scores come from the
    reranker's existing forward pass.

    Labels the reranker never scored keep the head's probability unchanged.
    """

    ce_probability = 1.0 / (1.0 + np.exp(-np.clip(ce_by_label, -30.0, 30.0)))
    return np.where(np.isfinite(ce_by_label), stage1 * ce_probability, stage1)


@dataclass
class Prediction:
    """The outcome of classifying one utterance.

    ``intent`` is the single best intent (the model always decides). ``score``
    ranks it against the alternatives and ``ranking`` lists every intent by that
    same quantity (descending, so ``ranking[0]`` is ``intent`` and the margin to
    ``ranking[1]`` is non-negative). ``explanation`` is the nearest labelled
    example.

    **Use ``confidence``, not ``score``, to decide whether to act.** ``score``
    is normalised across the reranked candidates -- it is a softmax over
    row-standardised values, so it always sums to 1 over the top-k and says
    only which candidate won, never whether any of them fit. Out-of-scope input
    still produces a peaked ``score``. ``confidence`` is the linear head's
    unnormalised probability for ``intent``, so it drops for input unlike
    anything in training and can be thresholded.

    ``abstain`` is set when the model was trained with ``oos`` examples and
    ``confidence`` fell below the threshold fitted from them; it stays ``False``
    when no threshold was fitted.
    """

    intent: str
    score: float
    ranking: list[tuple[str, float]] = field(default_factory=list)
    explanation: dict | None = None
    confidence: float = 0.0
    abstain: bool = False


class IntentModel:
    """A portable top-1 intent classifier.

    One opinionated pipeline: frozen ``bge-large`` embeddings, a linear head,
    and a trained cross-encoder reranker. Train with :meth:`fit`, then
    :meth:`classify` or :meth:`predict`. Out-of-scope examples (label ``oos``)
    are ignored at fit time.
    """

    def __init__(self, encoder: Encoder, scorer: LinearScorer, label_names: list[str]):
        self.encoder = encoder
        self.scorer = scorer
        self.label_names = label_names
        self.reranker: CrossEncoderReranker | None = None
        # Fitted from ``oos`` examples when the training data contains them;
        # ``None`` means no abstention signal was learned and the model always
        # decides. Compared against Prediction.confidence, never .score.
        self.oos_threshold: float | None = None
        self._train_vectors: np.ndarray | None = None
        self._train_y: np.ndarray | None = None
        self._train_texts: list[str] = []

    def _embed(self, texts: list[str]) -> np.ndarray:
        return self.encoder.encode(texts)

    # -- training -----------------------------------------------------------

    @classmethod
    def _fit_base(cls, examples: list[Example], encoder: Encoder) -> "IntentModel":
        """Stage 1 only (encoder + linear head). Internal; no reranker."""

        label_names = labels_of(examples, include_oos=False)
        index = {label: i for i, label in enumerate(label_names)}

        in_scope = [ex for ex in examples if ex.label != OOS_LABEL]
        texts = [ex.text for ex in in_scope]
        y = np.array([index[ex.label] for ex in in_scope], dtype=np.int64)

        vectors = encoder.encode(texts)
        scorer = LinearScorer()
        scorer.fit(vectors, y, len(label_names))

        model = cls(encoder, scorer, label_names)
        model._train_vectors = vectors
        model._train_y = y
        model._train_texts = texts
        return model

    @classmethod
    def fit(cls, examples: list[Example]) -> "IntentModel":
        """Train the full pipeline: bge-large + linear head + reranker.

        Examples labelled ``oos`` do not become a class -- they are held out of
        the classifier and used to fit the abstention threshold instead.
        """

        model = cls._fit_base(examples, SentenceEncoder())
        model.reranker = CrossEncoderReranker().fit(
            model._train_texts, model._train_y, model._train_vectors
        )
        model.fit_oos_threshold(examples)
        return model

    def _out_of_fold_confidence(self, n_splits: int = 5) -> np.ndarray:
        """In-scope confidences as they look on *unseen* text.

        Scoring the training examples with the head that was fitted on them
        gives near-1.0 confidences, which drags any threshold derived from them
        far above where real traffic sits. Cross-fitting gives each training
        example a confidence from a head that never saw it.
        """

        from sklearn.model_selection import StratifiedKFold

        y = self._train_y
        smallest_class = int(np.bincount(y).min())
        n_splits = min(n_splits, smallest_class)
        if n_splits < 2:                      # too few examples to cross-fit
            return self.scorer.scores(self._train_vectors).max(axis=1)

        out_of_fold_stage1 = np.zeros((len(y), self.scorer.n_labels), dtype=float)
        folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
        for train_idx, test_idx in folds.split(self._train_vectors, y):
            fold_scorer = LinearScorer(C=self.scorer.C, max_iter=self.scorer.max_iter)
            fold_scorer.fit(self._train_vectors[train_idx], y[train_idx],
                            self.scorer.n_labels)
            out_of_fold_stage1[test_idx] = fold_scorer.scores(
                self._train_vectors[test_idx]
            )

        if self.reranker is None:
            return out_of_fold_stage1.max(axis=1)

        # Same blend predict() uses, or the threshold would be fitted on a
        # different quantity than it is compared against. exclude_self keeps a
        # training utterance from matching itself among the exemplars, which
        # would inflate every positive and push the threshold too high.
        ce = self.reranker.raw_ce_by_label(
            self._train_texts, out_of_fold_stage1, exclude_self=True
        )
        return _blend_confidence(out_of_fold_stage1, ce).max(axis=1)

    def fit_oos_threshold(self, examples: list[Example], oos_quantile: float = 0.85,
                          max_inscope_abstain: float = 0.5) -> float | None:
        """Learn the confidence below which the model should abstain.

        The cut sits at the ``oos_quantile`` of the ``oos`` examples'
        confidences, so roughly ``1 - oos_quantile`` of out-of-scope input gets
        through. Raise it to reject more junk at the cost of abstaining on real
        traffic; lower it to commit more often.

        ``max_inscope_abstain`` is a safety rail, not a target: if a dataset's
        ``oos`` examples sit as high as its intents the cut would swallow
        everything, so it is never placed above that quantile of the in-scope
        training confidences. It should not normally bind.

        Returns ``None`` -- and leaves the model always deciding -- when the
        data has no ``oos`` examples.
        """

        oos_texts = [ex.text for ex in examples if ex.label == OOS_LABEL]
        if not oos_texts or self._train_vectors is None or not len(self._train_y):
            self.oos_threshold = None
            return None

        pos = self._out_of_fold_confidence()
        # OOS examples are never trained on, so these need no cross-fitting.
        neg = self._scores(oos_texts)[1].max(axis=1)

        # Place the cut from the negative side. The positives are training data:
        # cross-fitting removes the linear head's memory of them, but the
        # cross-encoder was still trained on these very texts, so their blended
        # confidences stay optimistic. Fitting the cut to them -- by balanced
        # accuracy, or by a low positive quantile -- lands too high and
        # abstained on 19-27% of genuine held-out traffic. The negatives were
        # never trained on, so their distribution is honest.
        cut = min(
            np.quantile(neg, oos_quantile),
            np.quantile(pos, max_inscope_abstain),      # safety rail only
        )
        # Nudge above the quantile so examples sitting exactly on it count as
        # rejected: `abstain` tests `confidence < threshold`, and a degenerate
        # negative distribution (identical confidences, common with a coarse
        # encoder or duplicated oos text) otherwise lands the cut on the mass
        # itself and abstains on none of it.
        self.oos_threshold = float(np.nextafter(cut, np.inf))
        return self.oos_threshold

    # -- inference ----------------------------------------------------------

    def _scores(self, texts: list[str], vectors: np.ndarray | None = None):
        """Return ``(ranking_scores, confidences)``.

        The two are kept separate on purpose. The first orders the candidates
        and is reranked when a reranker is set. The second is unnormalised and
        is what survives thresholding, because nothing renormalises it over a
        candidate subset.
        """

        if vectors is None:
            vectors = self._embed(texts)
        stage1 = self.scorer.scores(vectors)
        if self.reranker is None:
            return stage1, stage1
        ranking, ce = self.reranker.rerank_with_ce(texts, stage1)
        return ranking, _blend_confidence(stage1, ce)

    def _confidence(self, texts: list[str]) -> np.ndarray:
        """The ranking scores. Retained for backwards compatibility."""

        return self._scores(texts)[0]

    def classify(self, text: str) -> str:
        return self.classify_batch([text])[0]

    def classify_batch(self, texts: list[str]) -> list[str]:
        """Return the single best intent for each text."""

        conf = self._confidence(texts)
        return [self.label_names[int(i)] for i in conf.argmax(axis=1)]

    def predict(self, text: str) -> Prediction:
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: list[str]) -> list[Prediction]:
        vectors = self._embed(texts)
        conf, stage1 = self._scores(texts, vectors)
        results: list[Prediction] = []

        for row in range(len(texts)):
            order = np.argsort(conf[row])[::-1]
            top_idx = int(order[0])
            ranking = [(self.label_names[int(i)], float(conf[row][i])) for i in order]
            explanation = nearest_example(
                vectors[row], top_idx,
                self._train_vectors, self._train_y, self._train_texts,
            )
            confidence = float(stage1[row][top_idx])
            results.append(
                Prediction(
                    self.label_names[top_idx], float(conf[row][top_idx]),
                    ranking, explanation,
                    confidence=confidence,
                    abstain=(self.oos_threshold is not None
                             and confidence < self.oos_threshold),
                )
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
        """Top-1 report over the in-scope examples.

        OOS examples are excluded, so this number never counts an out-of-scope
        mistake. Use :meth:`oos_rejection_rate` alongside it.
        """

        in_scope = [ex for ex in examples if ex.label != OOS_LABEL]
        preds = self.classify_batch([ex.text for ex in in_scope])
        return score_predictions([ex.label for ex in in_scope], preds)

    def oos_rejection_rate(self, examples: list[Example]) -> float | None:
        """Fraction of ``oos`` examples the fitted threshold abstains on.

        ``None`` when no threshold was fitted or the data has no ``oos``
        examples. Report this next to :meth:`evaluate`: a model can be perfect
        on in-scope traffic and still route every unrelated question to an
        intent.
        """

        oos_texts = [ex.text for ex in examples if ex.label == OOS_LABEL]
        if not oos_texts or self.oos_threshold is None:
            return None
        return float(np.mean([p.abstain for p in self.predict_batch(oos_texts)]))

    # -- persistence --------------------------------------------------------

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.scorer.save(directory / "scorer")
        if self.reranker is not None:
            self.reranker.save(directory / "reranker")
        np.savez(directory / "train.npz", vectors=self._train_vectors, y=self._train_y)
        (directory / "texts.json").write_text(
            json.dumps(self._train_texts), encoding="utf-8"
        )
        (directory / "config.json").write_text(
            json.dumps(
                {
                    "encoder": self.encoder.spec(),
                    "label_names": self.label_names,
                    "reranker": self.reranker is not None,
                    "oos_threshold": self.oos_threshold,
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
            LinearScorer.load(directory / "scorer"),
            config["label_names"],
        )
        if config.get("reranker"):
            model.reranker = CrossEncoderReranker.load(directory / "reranker")
        # Absent in models saved before abstention existed -> always decide.
        model.oos_threshold = config.get("oos_threshold")

        train = np.load(directory / "train.npz")
        model._train_vectors = train["vectors"].astype(np.float32)
        model._train_y = train["y"].astype(np.int64)
        model._train_texts = json.loads(
            (directory / "texts.json").read_text(encoding="utf-8")
        )
        return model
