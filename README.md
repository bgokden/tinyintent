# tinyintent

A small, portable intent classifier. tinyintent contrastively fine-tunes a
sentence encoder on a few labelled utterances per intent and fits a linear
head on top, producing a model that maps text to the single best intent on
CPU, with no LLM in the loop.

```
utterance
  -> fine-tuned sentence encoder (bge-small)
  -> linear classifier head
  -> top-1 intent
```

## Install

Requires [uv](https://docs.astral.sh/uv/). Fine-tuning uses the `train` extra:

```bash
uv sync --extra train
```

## Quickstart (CLI)

```bash
uv run tinyintent train --data intents.jsonl --out model --finetune
uv run tinyintent predict --model model "cancel my order"
uv run tinyintent evaluate --model model --data intents.jsonl
```

```
> put my motorcycle up for sale
  intent: sell  (0.76)
  runners-up: rent 0.08, buy 0.07
  nearest example: "list my bike for sale" (0.89)
```

## Quickstart (Python)

```python
from tinyintent import IntentModel, finetune_encoder, load_jsonl

data = load_jsonl("intents.jsonl")
encoder = finetune_encoder(data, out_dir="model/encoder", epochs=1)
model = IntentModel.fit(data, encoder=encoder)
model.save("model")

print(model.classify("I want my money back for order 883"))   # refund

pred = model.predict("I want my money back for order 883")
print(pred.intent, pred.score)      # refund 0.83
print(pred.ranking[:3])             # ranked intents
print(pred.explanation)             # nearest labelled example
```

## Fine-tuning

The encoder is trained with a contrastive objective
(`MultipleNegativesRankingLoss` over same-intent pairs with in-batch
negatives, the SetFit body recipe). This specializes the embedding space so
same-intent utterances cluster and distinct intents separate, raising how
often the correct intent ranks first. The result is a plain
`SentenceTransformer`, used frozen by the classifier at inference.

`--epochs` controls the number of training passes:

```bash
uv run tinyintent train --data intents.jsonl --out model --finetune --epochs 3
```

One epoch is the default and is often best on large, well-separated datasets,
where extra epochs overfit the pair set. Small or closely-worded intent sets
can benefit from 2-3 epochs. A larger base encoder trades size for accuracy:

```python
finetune_encoder(data, out_dir="model/encoder", base_model="BAAI/bge-base-en-v1.5")
```

Fine-tuning needs a training step (a minute or two on a GPU).

## Data format

JSON Lines of `{"text", "label"}`. A handful of examples per intent is enough
(10-20 works well). The reserved label `oos` marks out-of-scope examples; they
are ignored during training and evaluation.

```json
{"text": "cancel my order", "label": "cancel_order"}
{"text": "what's the weather", "label": "oos"}
```

## Accuracy

Top-1 accuracy after fine-tuning (`bge-small` base, linear head, few-shot,
averaged over seeds):

| dataset | 10-shot | 20-shot |
|---|---:|---:|
| CLINC150 | 0.95 | 0.96 |
| Banking77 | 0.88 | 0.91 |

Reproduce with `uv run python scripts/benchmark.py --dataset banking --finetune`.

## API

- `finetune_encoder(examples, out_dir, base_model=..., epochs=1, ...)` — fine-tune and return an encoder
- `IntentModel.fit(examples, encoder=None, classifier="linear")` — fit the head
- `model.classify(text)` / `classify_batch(texts)` — the top-1 intent label(s)
- `model.predict(text)` — `Prediction(intent, score, ranking, explanation)`
- `model.evaluate(examples)` — top-1 accuracy report
- `model.save(dir)` / `IntentModel.load(dir)` — persist and reload

## How it works

- **Encoder** (`encoder.py`) — a fine-tuned `SentenceTransformer` (`bge-small`
  base). Pluggable via the `Encoder` protocol.
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
    finetune.py   contrastive encoder fine-tune
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
