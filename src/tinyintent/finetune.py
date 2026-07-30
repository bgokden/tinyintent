from __future__ import annotations

import random

from tinyintent.data import OOS_LABEL, Example
from tinyintent.encoder import DEFAULT_MODEL, SentenceEncoder


def finetune_encoder(
    examples: list[Example],
    out_dir: str,
    base_model: str = DEFAULT_MODEL,
    epochs: int = 2,
    pairs_per_example: int = 4,
    batch_size: int = 32,
    learning_rate: float = 2e-5,
    seed: int = 0,
) -> SentenceEncoder:
    """Contrastively fine-tune the encoder on the labelled examples.

    This is the optional "train" lever: it specializes the embedding space
    so same-intent utterances cluster and different intents separate, which
    is the only thing that moves the underlying ranking (frozen tricks
    cannot). It uses MultipleNegativesRankingLoss over same-intent pairs
    with in-batch negatives -- the SetFit body-tuning recipe -- then saves a
    plain SentenceTransformer that the rest of the framework uses frozen.

    Requires the training extra (``uv sync --extra train``).
    """

    from datasets import Dataset
    from sentence_transformers import (
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
    )
    from sentence_transformers.losses import MultipleNegativesRankingLoss

    rng = random.Random(seed)
    by_label: dict[str, list[str]] = {}
    for ex in examples:
        if ex.label != OOS_LABEL:
            by_label.setdefault(ex.label, []).append(ex.text)

    anchors: list[str] = []
    positives: list[str] = []
    for texts in by_label.values():
        if len(texts) < 2:
            continue
        for text in texts:
            others = [t for t in texts if t != text]
            for partner in rng.sample(others, min(pairs_per_example, len(others))):
                anchors.append(text)
                positives.append(partner)

    dataset = Dataset.from_dict({"anchor": anchors, "positive": positives})

    model = SentenceTransformer(base_model)
    loss = MultipleNegativesRankingLoss(model)
    args = SentenceTransformerTrainingArguments(
        output_dir=out_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_ratio=0.1,
        save_strategy="no",
        logging_strategy="no",
        report_to=[],
        disable_tqdm=True,
    )
    trainer = SentenceTransformerTrainer(
        model=model, args=args, train_dataset=dataset, loss=loss
    )
    trainer.train()
    model.save(out_dir)

    return SentenceEncoder(out_dir)
