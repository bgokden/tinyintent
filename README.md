# tinyintent

A small, portable intent-detection training framework. Give it labelled
utterances and it produces a tiny model that classifies intent with a
confidence score, an **abstain** option, and a plain-language
**explanation** — running on frozen sentence embeddings, no LLM at
inference.

It is deliberately minimal: a frozen encoder + a pluggable head +
calibrated abstention. Think "SetFit, stripped down, with out-of-scope
handling and explanations as first-class features."

```
utterance
  -> frozen encoder (MiniLM by default)
  -> head (prototype | logreg | mlp)
  -> per-label scores
  -> threshold + top1-top2 margin  ->  intent  or  abstain
```

## Why

Triggering workflows from short text (buy / sell / rent, cancel / refund /
track, or which tool to call) doesn't need an LLM call per message. A
frozen embedding plus a light head handles the clear cases instantly and
locally; the calibrated threshold lets it say "not sure" instead of firing
a wrong workflow. That abstention is exactly the signal to escalate the
hard cases to an LLM — cheap first pass, expensive fallback.

## Install

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Quickstart (CLI)

```bash
# train + calibrate, save to ./model
uv run tinyintent train --data examples/commerce_intents.jsonl --out model

# classify
uv run tinyintent predict --model model "put my motorcycle up for sale"
uv run tinyintent predict --model model        # interactive

# evaluate on a labelled set
uv run tinyintent evaluate --model model --data examples/commerce_intents.jsonl
```

```
> put my motorcycle up for sale
  intent: sell   score: 0.655
  ranked: sell 0.65, buy 0.39, cancel_order 0.27
  nearest example: "list my bike for sale" (0.65)

> what's the capital of France
  intent: (abstain)   score: 0.143
  ranked: rent 0.14, sell 0.11, buy 0.08
```

## Quickstart (Python)

```python
from tinyintent import IntentModel, load_jsonl, split

data = load_jsonl("examples/commerce_intents.jsonl")
train, val = split(data, test_frac=0.25)

model = IntentModel.fit(train, head="logreg")     # or "prototype" / "mlp"
model.calibrate(val, max_false_fire=0.02)         # tune the abstain threshold
model.save("model")

pred = model.predict("I want my money back for order 883")
print(pred.intent, round(pred.score, 3))          # refund 0.81
print(pred.abstained, pred.alternatives)
print(pred.explanation)                           # nearest labelled example
```

## Data format

JSON Lines, one `{"text", "label"}` per line. Use the reserved label
`oos` for out-of-scope / none-of-the-above examples — they are never
learned as a class, only used to calibrate when the model should decline.

```json
{"text": "cancel my order", "label": "cancel_order"}
{"text": "what's the weather", "label": "oos"}
```

Few-shot is fine — 8–15 examples per intent already works.

## Heads

All heads expose the same interface; pick by how much data you have.

| Head | Trains? | Use when |
|---|---|---|
| `prototype` | no | Few-shot. Scores a label by its nearest example (keeps multi-modal intents intact). Add an intent by adding examples. |
| `logreg` | yes (tiny) | You have some data per intent and want a calibrated linear boundary. Strong default. |
| `mlp` | yes | Only if a linear head underfits (many examples, interacting features). |

## Abstention and calibration

The model doesn't just take the argmax. It fires only when the top score
clears a **threshold** and beats the runner-up by a **margin**; otherwise
it abstains. `calibrate` sets both from a held-out split (with your `oos`
examples as negatives) to keep the out-of-scope false-fire rate under
`max_false_fire` while maximizing correct in-scope coverage. Evaluation
reports `accuracy`, `coverage`, `false_fire`, and `abstain_recall`.

## Encoders and portability

The default encoder is `all-MiniLM-L6-v2` via sentence-transformers. The
`Encoder` interface is pluggable — a dependency-free `HashingEncoder` is
included (offline, used in tests), and ONNX or static/Model2Vec encoders
can be dropped in for a smaller runtime footprint. The saved artifact is
just the head weights, the training vectors (for explanations), the label
names, and the calibrated thresholds — kilobytes on top of the encoder.

## Layout

```
src/tinyintent/
    data.py       Example, jsonl / few-shot loaders, stratified split
    encoder.py    Encoder protocol, SentenceEncoder (MiniLM), HashingEncoder
    heads/        prototype, logreg, mlp (+ base protocol)
    model.py      IntentModel: fit / predict / calibrate / evaluate / save / load
    calibrate.py  threshold + margin search
    metrics.py    accuracy, coverage, false_fire, abstain_recall
    explain.py    nearest labelled example
    cli.py        train / predict / evaluate
examples/
    commerce_intents.jsonl   buy/sell/rent/cancel/track/refund + oos
tests/            offline tests (hashing encoder)
```

## Not in scope

Argument/slot extraction (intent detection is step one; filling a
workflow's arguments is a separate concern) and multi-turn context are
intentionally out of scope to keep the model small and portable.
