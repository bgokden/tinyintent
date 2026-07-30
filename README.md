# tinyintent

A small, portable intent classifier. tinyintent embeds a few labelled
utterances per intent with a frozen sentence encoder and fits a linear head on
top, producing a model that maps text to the single best intent on CPU, with
no LLM in the loop and no training step.

```
utterance
  -> frozen sentence encoder (bge-large)
  -> linear classifier head
  -> top-1 intent
```

## Install

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Quickstart (CLI)

```bash
uv run tinyintent train --data intents.jsonl --out model
uv run tinyintent predict --model model "cancel my order"
uv run tinyintent evaluate --model model --data intents.jsonl
```

```
> put my motorcycle up for sale
  intent: sell  (0.87)
  runners-up: buy 0.04, rent 0.04
  nearest example: "list my bike for sale" (0.84)
```

## Quickstart (Python)

```python
from tinyintent import IntentModel, load_jsonl

data = load_jsonl("intents.jsonl")
model = IntentModel.fit(data)          # frozen encoder + linear head, no training
model.save("model")

print(model.classify("I want my money back for order 883"))   # refund

pred = model.predict("I want my money back for order 883")
print(pred.intent, pred.score)      # refund 0.83
print(pred.ranking[:3])             # ranked intents
print(pred.explanation)             # nearest labelled example
```

## Accuracy

Top-1 accuracy, frozen encoder + linear head, few-shot, averaged over seeds:

| dataset | bge-small (portable) | bge-large (default) |
|---|---:|---:|
| CLINC150, 20-shot | 0.96 | 0.97 |
| Banking77, 20-shot | 0.90 | 0.91 |

`bge-large` is the default for best accuracy; `bge-small` is ~10x lighter for
a point less. Swap the base with `IntentModel.fit(data, encoder=...)`:

```python
from tinyintent import IntentModel, SentenceEncoder
model = IntentModel.fit(data, encoder=SentenceEncoder("BAAI/bge-small-en-v1.5"))
```

Reproduce with `uv run python scripts/benchmark.py --dataset banking`.

## Fine-tuning (optional)

Fine-tuning contrastively specializes the encoder
(`MultipleNegativesRankingLoss` over same-intent pairs, the SetFit body
recipe). It is **optional and situational**: it adds about a point on a
smaller base at low shot counts, and only a marginal gain on `bge-large`,
which has little headroom left frozen.

| setup | Banking77, 20-shot |
|---|---:|
| bge-small, frozen | 0.895 |
| bge-small, fine-tuned (lr 2e-5) | 0.905 |
| bge-large, frozen | 0.909 |
| bge-large, fine-tuned (lr 1e-6) | 0.913 |

The learning rate must scale down with the base: 2e-5 suits `bge-small`, but a
335M encoder like `bge-large` needs roughly `1e-6` — the default 2e-5 slightly
*degrades* it. So reach for fine-tuning mainly when you need a small, portable
base *and* the extra point:

```bash
uv sync --extra train
uv run tinyintent train --data intents.jsonl --out model --finetune --epochs 1
# larger base: pass a lower LR via finetune_encoder(..., learning_rate=1e-6)
```

`--epochs` controls the passes (one is usually best; more overfits the pair
set). Fine-tuning needs a training step (a minute or two on a GPU). Note that
ModernBERT-based encoders collapse under this recipe at 2e-5 (they need a far
lower LR just to match their frozen accuracy), so they are best used frozen.

## Data format

JSON Lines of `{"text", "label"}`. A handful of examples per intent is enough
(10-20 works well). The reserved label `oos` marks out-of-scope examples; they
are ignored during training and evaluation.

```json
{"text": "cancel my order", "label": "cancel_order"}
{"text": "what's the weather", "label": "oos"}
```

## API

- `IntentModel.fit(examples, encoder=None, classifier="linear")` — fit the head on frozen embeddings
- `model.classify(text)` / `classify_batch(texts)` — the top-1 intent label(s)
- `model.predict(text)` — `Prediction(intent, score, ranking, explanation)`
- `model.evaluate(examples)` — top-1 accuracy report
- `model.save(dir)` / `IntentModel.load(dir)` — persist and reload
- `finetune_encoder(examples, out_dir, base_model=..., epochs=1, ...)` — optional encoder fine-tune

## How it works

- **Encoder** (`encoder.py`) — a frozen `SentenceTransformer` (`bge-large`
  base). Pluggable via the `Encoder` protocol; `bge-small`, static
  (Model2Vec), and a dependency-free hashing encoder are included.
- **Scorer** (`scorer.py`) — a logistic-regression head over the embeddings,
  stored as plain arrays so the model artifact stays portable.
- **Model** (`model.py`) — fits the head, returns the top-1 intent with a
  ranking, and attaches the nearest labelled example as an explanation.

## Layout

```
src/tinyintent/
    data.py       Example, jsonl / few-shot loaders, stratified split
    encoder.py    Encoder protocol, SentenceEncoder, StaticEncoder, HashingEncoder
    scorer.py     LinearScorer (default), ExemplarScorer
    finetune.py   optional contrastive encoder fine-tune
    model.py      IntentModel: fit / classify / predict / evaluate / save / load
    metrics.py    top-1 accuracy report
    explain.py    nearest labelled example
    cli.py        train / predict / evaluate
examples/         commerce intents
scripts/          benchmark.py, experiments.py
tests/            offline tests (hashing encoder)
```

## Not in scope

Argument/slot extraction and multi-turn context, to keep the model small and
portable.
