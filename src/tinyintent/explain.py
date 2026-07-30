from __future__ import annotations

import numpy as np


def nearest_example(
    query_vector: np.ndarray,
    label_index: int,
    train_vectors: np.ndarray,
    train_y: np.ndarray,
    train_texts: list[str],
) -> dict | None:
    """The training example of ``label_index`` closest to the query.

    A head-agnostic explanation: whatever head made the decision, this
    shows the labelled utterance that most resembles the input, which is
    what a human reads to judge whether the routing makes sense.
    """

    mask = train_y == label_index
    if not mask.any():
        return None

    indices = np.where(mask)[0]
    sims = train_vectors[indices] @ query_vector
    best = int(indices[int(np.argmax(sims))])
    return {"text": train_texts[best], "similarity": float(sims.max())}
