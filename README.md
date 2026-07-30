# tinyintent

A small, portable **selective** intent classifier. Give it labelled
utterances and it produces a model that, for a new utterance, returns a
risk-controlled decision — **fire** one intent, **abstain**, or flag it
**ambiguous** — with a plain-language explanation. Frozen sentence
embeddings, no LLM at inference.

It is intentionally opinionated: one scoring method, one principled
decision layer.

```
utterance
  -> frozen encoder (MiniLM by default)
  -> exemplar similarity per intent        (max cosine to each class)
  -> conformal prediction set at risk a     (coverage >= 1 - a)
  -> |set| = 0 abstain | = 1 fire | >= 2 ambiguous
```

## Why selective, not just a classifier

If a wrong intent triggers a workflow, "always pick the top class" is the
wrong behaviour. tinyintent is built around *declining*:

- **Conformal prediction** gives a distribution-free guarantee — set the
  risk `a`, and the true intent is in the returned set at least `1 - a` of
  the time on in-scope data.
- The set is over **absolute** exemplar similarity (not softmax), so an
  utterance far from every intent produces an **empty set** and abstains.
  This is what a relative softmax cannot do — it always names a winner.
- One mechanism covers the three outcomes you actually care about: nothing
  (abstain / out-of-scope), one (fire), or several (ambiguous → clarify or
  escalate to an LLM).

## Benchmark

CLINC150, 20-shot, 130 in-scope intents with 20 intents **held out entirely
as out-of-scope** (unseen at training — the hard, near-OOS case). Frozen
`all-MiniLM-L6-v2`.

| risk `a` | coverage | fire rate | fire acc | ambiguous | abstain | OOS false-fire | OOS abstain |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 0.95 | 0.23 | 0.96 | 0.74 | 0.03 | 0.23 | 0.28 |
| 0.10 | 0.91 | 0.40 | 0.96 | 0.53 | 0.07 | 0.32 | 0.42 |
| 0.20 | 0.79 | 0.55 | 0.97 | 0.27 | 0.19 | 0.27 | 0.69 |

Read it as: **coverage tracks `1 - a`** (the guarantee holds), and a fired
single intent is right **~96%** of the time. The cost is ambiguity — with
130 close intents many sets hold 2+ candidates, which is the safe failure
mode (escalate, don't misfire). Reproduce with
`uv run python scripts/benchmark.py`.

Note: "OOS" here is held-out *real* intents, the hardest kind. Far
out-of-scope input (chit-chat) abstains far more reliably.

## Install

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Quickstart (CLI)

```bash
uv run tinyintent train --data examples/commerce_intents.jsonl --out model --risk 0.1
uv run tinyintent predict --model model "put my motorcycle up for sale"
uv run tinyintent evaluate --model model --data examples/commerce_intents.jsonl
```

```
> put my motorcycle up for sale
  decision: fire  ->  sell
  prediction set: sell 0.53
  nearest example: "list my bike for sale" (0.53)

> what's the weather tomorrow
  decision: abstain
  prediction set: (empty)
```

## Quickstart (Python)

```python
from tinyintent import IntentModel, load_jsonl

data = load_jsonl("examples/commerce_intents.jsonl")

# fit + calibrate with an internal held-out split (conformal needs one)
model = IntentModel.fit_calibrate(data, risk=0.1)
model.save("model")

pred = model.predict("I want my money back for order 883")
print(pred.decision, pred.intent)     # fire refund
print(pred.set_)                      # [('refund', 0.81), ...]
print(pred.explanation)               # nearest labelled example
```

Calibration must use held-out data — exemplars self-match at similarity
1.0, which collapses the threshold. `fit_calibrate` splits for you; if you
call `fit` and `calibrate` separately, pass disjoint sets.

## Data format

JSON Lines of `{"text", "label"}`. Use the reserved label `oos` for
out-of-scope examples — they are never a class, only used to measure false
firing. Few-shot is fine (10–20 per intent).

```json
{"text": "cancel my order", "label": "cancel_order"}
{"text": "what's the weather", "label": "oos"}
```

## How it works

- **Encoder** (`encoder.py`) — frozen `all-MiniLM-L6-v2` by default;
  pluggable via the `Encoder` protocol. A dependency-free `HashingEncoder`
  is included for offline tests. ONNX or static (Model2Vec) encoders can be
  dropped in for a smaller footprint.
- **Scorer** (`scorer.py`) — each intent is its set of example vectors; a
  query's score for an intent is the max cosine similarity to them
  (absolute, so out-of-scope stays low; multi-modal intents stay intact).
- **Conformal** (`conformal.py`) — split-conformal (LAC): the threshold
  `1 - q` is calibrated so in-scope coverage is at least `1 - a`.
- **Model** (`model.py`) — ties them together, turns set size into a
  decision, and attaches the nearest exemplar as an explanation.

## Tuning

- **`risk`** is the dial. Lower → larger sets, more coverage, more abstain /
  ambiguous. Higher → more single-intent fires, less coverage.
- **More shots or more separable intents** shrink sets and reduce
  ambiguity. If two intents are near-duplicates, expect ambiguity — that is
  the model correctly refusing to guess.

## Layout

```
src/tinyintent/
    data.py       Example, jsonl / few-shot loaders, stratified split
    encoder.py    Encoder protocol, SentenceEncoder (MiniLM), HashingEncoder
    scorer.py     ExemplarScorer (per-class max cosine similarity)
    conformal.py  split-conformal prediction sets (LAC)
    model.py      IntentModel: fit / calibrate / predict / evaluate / save / load
    metrics.py    coverage, fire rate/accuracy, ambiguous, abstain, OOS rates
    explain.py    nearest labelled example
    cli.py        train / predict / evaluate
examples/         commerce intents (+ oos)
scripts/          benchmark.py (CLINC150)
tests/            offline tests (hashing encoder)
```

## Not in scope

Argument/slot extraction (a separate step after the intent) and multi-turn
context, to keep the model small and portable.
