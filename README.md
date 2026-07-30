# tinyintent

A small, portable intent classifier. Give it labelled utterances and it
maps text to the single best intent — accurately, on CPU, with no LLM in
the loop. Frozen sentence embeddings plus a light classifier head; a few
example utterances per intent is enough.

```
utterance
  -> frozen encoder (bge-small by default)
  -> linear classifier head
  -> top-1 intent   (always decides)
```

## What it does

Return the single best intent for every input. Top-1 accuracy (20-shot,
frozen `bge-small`, averaged over seeds):

| dataset | exemplar head | **linear head (default)** | + fine-tune |
|---|---:|---:|---:|
| CLINC150 | 0.92 | **0.96** | 0.96 |
| Banking77 | 0.89 | **0.90** | 0.90 |

The linear (logistic-regression) head learns a decision boundary instead of
trusting the single nearest example, which is why it wins for pure accuracy.
No fine-tuning needed for CLINC-level results; fine-tuning helps most when
intents are close (see below).

```bash
uv run tinyintent train --data intents.jsonl --out model
uv run tinyintent predict --model model "cancel my order"
# Python: IntentModel.fit(examples).classify("cancel my order")
```

## How it compares

Few-shot top-1 accuracy against published methods on the same datasets. The
point is that a frozen modern encoder plus a linear head is competitive with
methods that fine-tune, at a fraction of the cost:

| method | CLINC 5-shot | CLINC 10-shot | Banking 5-shot | Banking 10-shot |
|---|---:|---:|---:|---:|
| **tinyintent** (frozen bge + linear) | 0.91 | 0.95 | 0.83 | 0.87 |
| DNNC | 0.91 | 0.94 | 0.80 | 0.87 |
| CPFT | 0.92 | 0.94 | 0.81 | 0.87 |
| SetFit (8-shot) | 0.86 | — | 0.78 | — |

Full-data ceilings are around 0.97 (CLINC) and 0.94-0.95 (Banking, RoBERTa /
SPACE-2.0). Most of tinyintent's few-shot strength comes from the encoder, not
a clever method — which is exactly the point: keep the method small and let a
good frozen encoder do the work.

## Install

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Quickstart (CLI)

```bash
uv run tinyintent train --data examples/commerce_intents.jsonl --out model
uv run tinyintent predict --model model "put my motorcycle up for sale"
uv run tinyintent evaluate --model model --data examples/commerce_intents.jsonl
```

```
> put my motorcycle up for sale
  intent: sell  (0.71)
  runners-up: buy 0.12, rent 0.08
  nearest example: "list my bike for sale" (0.53)
```

## Quickstart (Python)

```python
from tinyintent import IntentModel, load_jsonl

data = load_jsonl("examples/commerce_intents.jsonl")
model = IntentModel.fit(data)
model.save("model")

print(model.classify("I want my money back for order 883"))   # refund

pred = model.predict("I want my money back for order 883")
print(pred.intent, pred.score)   # refund 0.83
print(pred.ranking[:3])          # [('refund', 0.83), ('cancel_order', 0.06), ...]
print(pred.explanation)          # nearest labelled example
```

## Fine-tuning (optional)

Frozen embeddings cap how often the true intent ranks first among many close
intents. A short contrastive fine-tune (SetFit body recipe: same-intent pairs
with in-batch negatives) specializes the encoder and helps most on datasets
with near-synonym intents. It stays a plain SentenceTransformer afterwards,
used frozen by the rest of the pipeline.

```bash
uv sync --extra train
uv run tinyintent train --data intents.jsonl --out model --finetune
# or in Python: from tinyintent import finetune_encoder
```

It needs a training step (a minute or two on a GPU) and breaks the pure
zero-training story, so it is opt-in.

## Data format

JSON Lines of `{"text", "label"}`. Few-shot is fine (10-20 per intent). The
reserved label `oos` marks out-of-scope examples; they are ignored at fit
time (never a class) and skipped when measuring accuracy.

```json
{"text": "cancel my order", "label": "cancel_order"}
{"text": "what's the weather", "label": "oos"}
```

## How it works

- **Encoder** (`encoder.py`) — frozen `BAAI/bge-small-en-v1.5` by default;
  pluggable via the `Encoder` protocol. A dependency-free `HashingEncoder`
  is included for offline tests. ONNX or static (Model2Vec) encoders can be
  dropped in for a smaller footprint.
- **Scorer** (`scorer.py`) — the classifier head. `LinearScorer` (default)
  is a logistic regression over the embeddings; `ExemplarScorer` scores each
  intent by max cosine similarity to its example vectors. Both store plain
  arrays, so the artifact stays portable.
- **Model** (`model.py`) — ties them together, returns the top-1 intent with
  a ranking, and attaches the nearest labelled example as an explanation.

## Encoders

Default: `BAAI/bge-small-en-v1.5` — it beat `all-MiniLM-L6-v2` on the intent
benchmarks while staying small and frozen. Alternatives via the pluggable
`Encoder` protocol:

- `SentenceEncoder("sentence-transformers/all-MiniLM-L6-v2")` — lighter.
- `StaticEncoder` — Model2Vec static embeddings, **numpy-only, no torch**,
  for the smallest footprint (`uv sync --extra static`). Faster and tiny,
  but weaker on phrasing/negation.
- `HashingEncoder` — dependency-free stub used in tests.

## Layout

```
src/tinyintent/
    data.py       Example, jsonl / few-shot loaders, stratified split
    encoder.py    Encoder protocol, SentenceEncoder, StaticEncoder, HashingEncoder
    scorer.py     LinearScorer (default), ExemplarScorer
    model.py      IntentModel: fit / classify / predict / evaluate / save / load
    metrics.py    top-1 accuracy report
    explain.py    nearest labelled example
    finetune.py   optional contrastive encoder fine-tune
    cli.py        train / predict / evaluate
examples/         commerce intents
scripts/          benchmark.py (CLINC150, Banking77)
tests/            offline tests (hashing encoder)
```

## Not in scope

Argument/slot extraction (a separate step after the intent) and multi-turn
context, to keep the model small and portable.
