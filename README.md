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

## Primary use: accurate top-1 classification

The default is to **always decide** — return the single best intent for
every input. Top-1 accuracy (20-shot, frozen `bge-small`, averaged over
seeds):

| dataset | exemplar head | **linear head (default)** | + fine-tune |
|---|---:|---:|---:|
| CLINC150 | 0.92 | **0.96** | 0.96 |
| Banking77 | 0.89 | **0.90** | 0.90 |

The linear (logistic-regression) head learns a boundary instead of trusting
the single nearest example, which is why it wins for pure accuracy. No
fine-tuning needed for CLINC-level results; fine-tuning helps most when
intents are close (see below).

```bash
uv run tinyintent train --data intents.jsonl --out model      # linear, always decide
uv run tinyintent predict --model model "cancel my order"
# Python: IntentModel.fit(examples, classifier="linear").classify("cancel my order")
```

## Optional: abstain or reject instead of always deciding

If you would rather decline on uncertain or out-of-scope input than force a
guess, add a decision policy (`--method`): `gate` (fire top-1 or reject),
`aps` (safety-first prediction sets), or `lac` (a single threshold). These
trade some coverage for restraint and are documented below; the default is
to skip them and always decide.

## Benchmark

Real datasets (CLINC150, Banking77), few-shot, with some intents **held out
entirely as out-of-scope** (unseen — the hard, near-OOS case) and a slice of
those used as calibration negatives. Frozen `all-MiniLM-L6-v2`, averaged over
seeds. Reproduce with
`uv run python scripts/benchmark.py --dataset {clinc,banking} --seeds 3`.

Banking77, 20-shot, 65 in-scope + 12 OOS intents, risk 0.2:

| policy | coverage | fire rate | fire acc | ambiguous | OOS false-fire |
|---|---:|---:|---:|---:|---:|
| **APS** (default) | 0.74 | 0.19 | **1.00** | 0.55 | **0.02** |
| **LAC + floor** | 0.71 | **0.60** | 0.96 | 0.14 | 0.20 |

Two policies, two operating profiles — a trade, not a winner:

- **APS is safety-first.** It almost never fires the wrong intent (OOS
  false-fire ~0, fire accuracy ~100%) by turning uncertain or out-of-scope
  inputs into *ambiguous* instead of a confident guess. The price is
  decisiveness — it fires less and escalates more.
- **LAC + floor is decisive.** It resolves most inputs itself (~60% fire),
  at the cost of more near-OOS false fires (~20%). It also improves with
  more shots, where APS stays conservative.

Pick by the cost of a wrong workflow versus the cost of escalating. The same
pattern holds on CLINC150.

The decisiveness ceiling is the frozen-encoder ranking, not the set method:
regularized APS (RAPS, `reg_lambda`) and an LDA transform were both tried and
did not help. The one thing that does is **fine-tuning the encoder** (see
below), which lifts the decisive LAC policy substantially:

| setup | fire rate | fire acc | ambiguous | coverage |
|---|---:|---:|---:|---:|
| frozen bge + LAC (CLINC) | 0.60 | 0.98 | 0.15 | 0.74 |
| **fine-tuned bge + LAC (CLINC)** | **0.79** | **0.99** | **0.02** | **0.81** |

## Fine-tuning (optional)

Frozen embeddings cap how often the true intent ranks first among many close
intents. A short contrastive fine-tune (SetFit body recipe: same-intent pairs
with in-batch negatives) specializes the encoder and fixes that ranking. It
is the one lever that meaningfully raises decisiveness — validated on CLINC150
and Banking77 — and it stays a plain SentenceTransformer afterwards, used
frozen by the rest of the pipeline.

```bash
uv sync --extra train
uv run tinyintent train --data intents.jsonl --out model --finetune --method lac
# or in Python: from tinyintent import finetune_encoder
```

It needs a training step (a minute or two on a GPU) and breaks the pure
zero-training story, so it is opt-in. APS stays conservative either way; pair
fine-tuning with the decisive LAC policy.

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
- **Decision policy** — two options, both giving `1 - a` coverage:
  - **APS** (`aps.py`, default) — two stage: an absolute-similarity *gate*
    rejects out-of-scope before any softmax, then Adaptive Prediction Sets
    build the set over temperature-scaled probabilities. Safety-first.
  - **LAC** (`conformal.py`) — a single absolute-similarity threshold
    `1 - q`. Simpler and more decisive.
- **Model** (`model.py`) — ties them together, turns set size into a
  decision, and attaches the nearest exemplar as an explanation.

## Tuning

- **`risk`** is the dial. Lower → larger sets, more coverage, more abstain /
  ambiguous. Higher → more single-intent fires, less coverage.
- **More shots or more separable intents** shrink sets and reduce
  ambiguity. If two intents are near-duplicates, expect ambiguity — that is
  the model correctly refusing to guess.
- **Provide negatives.** Put `oos` examples in the calibration data and they
  raise an absolute abstain floor (`calibrate(..., use_oos=True)`, on by
  default). On the benchmark this lifted OOS abstention from 0.42 to 0.73
  and cut ambiguity, trading some in-scope coverage — a knob worth having
  when false firing is costly.
- **`method`** picks the profile: `aps` (default, safety-first — rarely
  misfires, escalates more) or `lac` (decisive — fires more, misfires more
  on near-OOS). See the benchmark.
- **`mondrian=True`** (LAC only) calibrates a threshold per intent. It helps
  only with plenty of calibration data per class; at few-shot it inflates
  sets, so it is off by default.

## Encoders

Default: `BAAI/bge-small-en-v1.5` — it beat `all-MiniLM-L6-v2` on the intent
benchmarks (higher coverage and fire accuracy, lower near-OOS false-fire)
while staying small and frozen. Alternatives via the pluggable `Encoder`
protocol:

- `SentenceEncoder("sentence-transformers/all-MiniLM-L6-v2")` — lighter.
- `StaticEncoder` — Model2Vec static embeddings, **numpy-only, no torch**,
  for the smallest footprint (`uv sync --extra static`). Faster and tiny,
  but weaker on phrasing/negation.
- `HashingEncoder` — dependency-free stub used in tests.

An optional LDA metric-learning transform (`fit(..., transform="lda")`)
sharpens class separation but lowered coverage in testing, so it is off by
default.

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
