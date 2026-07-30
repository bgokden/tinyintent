from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class MLPHead:
    """A small MLP over frozen embeddings.

    Only worth using when a linear head underfits (interacting features,
    many examples). Kept intentionally tiny and trained for a fixed, short
    schedule so it stays fast and portable.
    """

    name = "mlp"

    def __init__(self, hidden_dim: int = 128, epochs: int = 60, lr: float = 1e-3, seed: int = 0) -> None:
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.seed = seed
        self.n_labels = 0
        self.input_dim = 0
        self._net = None

    def _build(self, input_dim: int, n_labels: int):
        import torch
        from torch import nn

        torch.manual_seed(self.seed)
        return nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, n_labels),
        )

    def fit(self, vectors: np.ndarray, y: np.ndarray, n_labels: int) -> None:
        import torch
        from torch import nn

        self.n_labels = n_labels
        self.input_dim = vectors.shape[1]
        self._net = self._build(self.input_dim, n_labels)

        x = torch.from_numpy(vectors.astype(np.float32))
        target = torch.from_numpy(y.astype(np.int64))
        optimizer = torch.optim.AdamW(self._net.parameters(), lr=self.lr, weight_decay=1e-4)
        loss_fn = nn.CrossEntropyLoss()

        self._net.train()
        for _ in range(self.epochs):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(self._net(x), target)
            loss.backward()
            optimizer.step()
        self._net.eval()

    def scores(self, vectors: np.ndarray) -> np.ndarray:
        import torch

        with torch.no_grad():
            logits = self._net(torch.from_numpy(vectors.astype(np.float32)))
            proba = torch.softmax(logits, dim=-1).numpy()
        return proba.astype(np.float32)

    def save(self, directory: str | Path) -> None:
        import torch

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self._net.state_dict(), directory / "mlp.pt")
        (directory / "mlp.json").write_text(
            json.dumps(
                {
                    "hidden_dim": self.hidden_dim,
                    "epochs": self.epochs,
                    "lr": self.lr,
                    "seed": self.seed,
                    "n_labels": self.n_labels,
                    "input_dim": self.input_dim,
                }
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: str | Path) -> "MLPHead":
        import torch

        directory = Path(directory)
        config = json.loads((directory / "mlp.json").read_text(encoding="utf-8"))
        head = cls(
            hidden_dim=config["hidden_dim"],
            epochs=config["epochs"],
            lr=config["lr"],
            seed=config["seed"],
        )
        head.n_labels = config["n_labels"]
        head.input_dim = config["input_dim"]
        head._net = head._build(head.input_dim, head.n_labels)
        head._net.load_state_dict(torch.load(directory / "mlp.pt", weights_only=True))
        head._net.eval()
        return head
