# tinyintent

A small, portable intent classifier. Give it a few labelled utterances per
intent; it maps text to the single best intent on CPU, with no LLM in the loop.
One opinionated pipeline, no knobs to turn.

```
utterance
  -> frozen sentence encoder (bge-large)
  -> linear classifier head        (top-k candidates)
  -> trained cross-encoder reranker (picks the best)
  -> intent
```

## Install

From PyPI, with [uv](https://docs.astral.sh/uv/) or pip:

```bash
uv add tinyintent
# or
pip install tinyintent
```

Latest from git:

```bash
uv add "git+https://github.com/bgokden/tinyintent"
```

Or clone and set up for development:

```bash
uv sync
uv run pytest -q -m "not slow"   # add -m slow for the end-to-end training tests
```

Python 3.11+. Everything runs on CPU — no GPU, no API key, no LLM. Installing
pulls in torch, sentence-transformers, datasets and accelerate (training the
reranker needs the last two), so expect a few hundred MB of wheels.

The models are downloaded on first `fit()`, not at install time:

| model | role | size |
|---|---|---|
| `BAAI/bge-large-en-v1.5` | frozen sentence encoder | ~1.2 GB |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | reranker starting point | ~88 MB |

They are cached in `~/.cache/huggingface`, so only the first run pays for it. In
CI or a container, cache that directory or the download repeats on every build.

## Quickstart (CLI)

```bash
uv run tinyintent train --data intents.jsonl --out model
uv run tinyintent predict --model model "cancel my order"
uv run tinyintent evaluate --model model --data intents.jsonl
```

```
> put my motorcycle up for sale
  intent: sell  (score 0.87, confidence 0.79)
  runners-up: buy 0.04, rent 0.04
  nearest example: "list my bike for sale" (0.84)

> what's the weather in berlin
  intent: rent  (score 0.71, confidence 0.04)  ABSTAIN (below threshold)
  runners-up: buy 0.16, sell 0.08
  nearest example: "how much to rent a van" (0.31)
```

## Quickstart (Python)

```python
from tinyintent import IntentModel, load_jsonl

data = load_jsonl("intents.jsonl")
model = IntentModel.fit(data)          # trains the whole pipeline
model.save("model")

print(model.classify("I want my money back for order 883"))   # refund

pred = model.predict("I want my money back for order 883")
print(pred.intent, pred.confidence)  # refund 0.74  <- gate on this
print(pred.score)                    # 0.80         <- ranks, does not calibrate
print(pred.ranking[:3])              # ranked intents
print(pred.explanation)              # nearest labelled example

if pred.abstain:                     # set when trained with `oos` examples
    ask_for_clarification()
else:
    route(pred.intent)
```

## How it works

`IntentModel.fit(data)` trains three parts, and `classify`/`predict` run them
in order. There are no options — this is the configuration that measured best.

- **Encoder** — a frozen `bge-large` sentence encoder. It won an encoder sweep
  on the intent benchmarks; nothing smaller matched it and fine-tuning it did
  not help.
- **Linear head** — a logistic-regression classifier over the embeddings. It
  beats nearest-exemplar for top-1 accuracy and produces the top-k candidates.
- **Cross-encoder reranker** — a cross-encoder *trained on your data* (same-
  intent vs different-intent pairs, with hard negatives) reads each candidate
  together with the query and re-ranks them, ensembled with the head's scores.
  Off-the-shelf cross-encoders hurt; the win comes from training it on your
  intents, which is why it is always trained, never bundled pretrained.

## Accuracy

Top-1 accuracy, few-shot (20 examples/intent), averaged over seeds:

| dataset | accuracy |
|---|---:|
| CLINC150 | 0.975 |
| Banking77 | 0.915 |

Banking77's intents overlap heavily, so it is the harder ceiling; CLINC150 is
near-saturated. Reproduce with `uv run python scripts/benchmark.py`.

## How long training takes

Training is short because only the reranker is trained — the encoder is frozen
and the linear head is a logistic regression that fits in well under a second.
Cost scales with the number of *examples*, not the number of intents.

Measured on an **Apple M5 (10 cores, 32 GB, macOS 26.2)**, CPU/MPS only, models
already cached:

| intents | examples | linear head | reranker | **total `fit()`** |
|---:|---:|---:|---:|---:|
| 2 | 20 | 0.3 s | 5.8 s | **6.1 s** |
| 4 | 60 | 0.2 s | 9.6 s | **9.8 s** |
| 8 | 120 | 0.4 s | 15.9 s | **16.3 s** |

Add roughly 5-10 s the first time in a process for loading `bge-large`, and a
one-off download of ~1.3 GB the very first time on a machine.

Inference, same machine:

| operation | latency |
|---|---:|
| `predict` — one utterance | ~55 ms |
| `predict_batch` — per utterance, batched | ~35 ms |
| `predict_batch` — per utterance, reranker disabled | ~3 ms |
| `IntentModel.load` | ~3 s |
| saved model on disk | 92 MB |

The reranker dominates inference: it runs the query against each candidate's
exemplars, so cost grows with exemplars per intent. If you need sub-10 ms
routing and can accept slightly weaker ranking, set `model.reranker = None`
after loading — the linear head alone answers in ~3 ms.

A CPU-only Linux box without MPS will be slower, roughly 2-3x on training, so
treat these as a floor rather than a guarantee.

## Agent tool routing

The classic use case: decide which tool an agent should call. Label each intent
with a tool name, and the predicted intent is the tool to invoke (or to inject
into an LLM prompt). `examples/agent_tools.jsonl` is a toy dataset for this
(web_search, calculator, weather, calendar, email, ...).

`examples/graph_agent.py` builds a small **state-machine agent** on top: each
state allows a subset of intents as edges, and the agent follows the
highest-ranked *allowed* edge — so one classifier drives both tool selection
and control flow. Most tools return to the router; email is a two-step
draft → confirm/cancel path.

```
uv run python examples/graph_agent.py

[ROUTER] user: 'send an email to Sam about lunch'
    -> intent=email (0.87)  (runner-up reminder 0.06) | drafted the email... | next=EMAIL_CONFIRM
[EMAIL_CONFIRM] user: 'yes go ahead'
    -> intent=confirm (0.91)  (runner-up cancel 0.03) | email sent | next=ROUTER
```

The ranking matters here: in `EMAIL_CONFIRM` the agent only accepts `confirm` or
`cancel`, so it picks the top-ranked intent among those rather than the global
best.

## Conversational flow

The same graph pattern drives a conversational agent, where nodes are call
*phases* rather than tools. `examples/sales_flow.jsonl` labels **conversational
intents** (interested, question, objection, commit, handover, not_interested,
goodbye), and `examples/conversation_agent.py` walks a call graph:

```
INTRODUCTION -> PITCH -> QUESTION -> OBJECTION -> CLOSE -> BOOKED
             \-> EXIT   \-> HANDOVER
```

Each node has a line the agent says and intent-keyed edges; tinyintent
classifies the caller's reply and the agent follows the valid edge.

```
uv run python examples/conversation_agent.py

agent [PITCH]: We help homeowners cut their electric bill with rooftop solar...
  caller: 'we already use another provider'   ->  [objection 0.90]
agent [OBJECTION]: I hear you -- a lot of our customers felt the same...
  caller: 'okay that sounds interesting'   ->  [interested 0.91]
agent [CLOSE]: I'd love to book you a free 15-minute assessment. Shall I set that up?
  caller: "yes let's do it"   ->  [commit 0.90]
agent [BOOKED]: Fantastic, you're all set...
```

`predict` returns two different numbers, and they answer different questions.

| field | what it measures | gate on it for |
|---|---|---|
| `score` | the reranked softmax over the top candidates; `ranking` is ordered by it, so the top is always the decision and the margin to the runner-up is non-negative | **which** intent, and how close the call was between candidates |
| `confidence` | the linear head's unnormalised probability for the chosen intent | **whether any** intent fits at all |

`score` is normalised across the candidates, so it always sums to 1 over them.
That makes it a good relative signal and a poor absolute one: out-of-scope input
still produces a peaked `score`. On an 8-intent support model, *"what time do you
close on sundays"* scores **0.89** — indistinguishable from a real request. The
same utterance has a `confidence` of **0.38**. Threshold `confidence`; compare
`score` only against the other candidates.

`conversation_agent.py` uses the relative signal as a gate (`MIN_SCORE` /
`MIN_MARGIN`): when the best edge is too weak *against its rivals*, the agent
**stays in the node** (a self-loop -- a normal FSM choice) and asks the caller to
clarify, then routes cleanly next turn. That is in-domain ambiguity, which is
what `score` is good at. For "this caller is talking about something else
entirely", use `confidence` or `abstain`.

```
caller: 'well, it depends'        ->  [uncertain: question 0.43, margin 0.10]  STAY + clarify
agent [INTRODUCTION]: Sorry, I didn't quite catch that -- could you say a bit more?
caller: 'yeah okay, tell me more' ->  [interested 0.91]
agent [PITCH]: We help homeowners cut their electric bill...
```

Because the model always decides, this policy — stay/self-loop, ask again, or a
dedicated clarify node — lives in your graph, not the classifier, which is the
right place for it when you build the agent yourself.

### Measuring routing quality

`examples/eval_flow.py` holds out part of the flow data, trains on the rest, and
scores the routing on unseen utterances — accuracy, macro / weighted F1, and a
per-intent breakdown (pooled over splits):

```
uv run python examples/eval_flow.py

  accuracy 0.847 | macro-F1 0.833 | weighted-F1 0.835   (n=72)

intent            precision  recall    f1
greeting              1.000   1.000  1.000
question              1.000   1.000  1.000
commit                0.900   1.000  0.947
handover              0.900   1.000  0.947
interested            1.000   0.667  0.800
objection             0.600   0.333  0.429   <- the hard class
```

The per-intent F1 shows exactly which transitions are reliable and which need
work: here `objection` is weakest (objections are diverse and overlap with
questions and rejections), so it is the intent to add more examples for. The
same report backs `tinyintent evaluate`, and `IntentModel.evaluate(data)`
returns it as a `Report`.

## Data format

JSON Lines of `{"text", "label"}`. A handful of examples per intent is enough
(10-20 works well).

```json
{"text": "cancel my order", "label": "cancel_order"}
{"text": "what's the weather", "label": "oos"}
```

The reserved label `oos` marks out-of-scope examples. They never become an
intent — `fit` holds them out of the classifier and uses them to fit an
abstention threshold on `confidence`, stored on the model as `oos_threshold`
and saved with it. Predictions then come back with `abstain=True` when they
fall below it:

```python
pred = model.predict("what time do you close on sundays")
pred.intent      # still the best in-scope guess -- the model always decides
pred.abstain     # True: below the fitted threshold, so don't act on it
```

Abstention is advisory: `predict` always returns an intent and a ranking, and
the caller decides what to do. Without `oos` examples in the training data no
threshold is fitted, `oos_threshold` is `None`, and `abstain` is always `False`.

`evaluate()` reports in-scope accuracy only and never counts an out-of-scope
mistake; pair it with `model.oos_rejection_rate(data)`, which is the fraction of
`oos` examples the threshold catches.

## Using it well

### Write examples the way your users actually type

The encoder generalises across wording, so you do not need to enumerate
phrasings — you need to cover the *ways of asking*. Ten examples spanning
direct requests, complaints, and questions beat forty rewordings of one
sentence. Copy real utterances from logs where you can; invented data drifts
toward how you write, not how your users do.

Match the register too. If users type `where's my stuff`, do not train only on
`I would like to enquire about my delivery`.

### 10-20 examples per intent, roughly balanced

Below ~8 the linear head gets unstable; past ~30 the returns flatten and
training slows. Keep intents within about 3x of each other in size — a class
with 60 examples against classes with 10 will absorb the ambiguous cases.

### Always include `oos` examples

They cost one line each and are the difference between a router that says "I
don't know" and one that confidently sends a weather question to your refunds
tool. 20-50 works well. Make them *realistic* misses — the things people
actually type at your bot — not absurdities:

```json
{"text": "do you ship to australia", "label": "oos"}
{"text": "can i pay in monthly installments", "label": "oos"}
```

Near-misses like these teach the threshold where the real boundary is. Only
`what's the weather` teaches it nothing, because that was never going to be
confused.

### Gate on `confidence`, rank on `score`

The two answer different questions, and using the wrong one is the most common
way to get burned:

```python
pred = model.predict(text)

if pred.abstain:                      # or: pred.confidence < your_threshold
    clarify()                         # nothing in scope fits
elif pred.score - pred.ranking[1][1] < 0.15:
    disambiguate(pred.ranking[:2])    # in scope, but two intents are close
else:
    route(pred.intent)
```

`score` is normalised across candidates, so it stays high even when nothing
fits — it tells you *which* intent, never *whether*. `confidence` is
unnormalised and drops for unfamiliar input.

### Let the confusion report drive your data

`model.evaluate(data)` gives per-intent precision/recall/F1. Low recall on one
intent means it needs more examples; a pair that keeps swapping means the
intents overlap conceptually. Two intents that persistently confuse are usually
one intent plus a parameter — merge them and extract the difference downstream.

Hold data out rather than scoring on the training set:

```python
from tinyintent import split
train, test = split(data, test_frac=0.2, seed=0)
model = IntentModel.fit(train)
print(model.evaluate(test))
print(model.oos_rejection_rate(test))
```

### Retrain when intents change

The reranker is trained on *your* intents, so there is no incremental update:
adding or renaming an intent means calling `fit` again. At these sizes that is
seconds, so treat the model as a build artefact — retrain in CI when the data
file changes and ship `model/` alongside your app.

### Serving

Load once at startup, not per request (`load` costs ~3 s). `predict_batch` is
substantially cheaper per utterance than looping over `predict`. If you need
sub-10 ms and can accept slightly weaker ranking, drop the reranker with
`model.reranker = None`.

### What it is not for

Single-label, single-sentence routing is the whole design. It will not extract
entities, handle "cancel my order and also update my address" as two intents,
or classify long documents. It has no notion of conversation history — pass the
turn you want classified, and keep state in your own graph.

## Layout

```
src/tinyintent/
    data.py       Example, jsonl / few-shot loaders, stratified split
    encoder.py    frozen bge-large encoder (hashing stub for offline tests)
    scorer.py     linear classifier head
    reranker.py   trained cross-encoder reranker
    model.py      IntentModel: fit / classify / predict / evaluate / save / load
    metrics.py    top-1 accuracy report
    explain.py    nearest labelled example
    cli.py        train / predict / evaluate
examples/         commerce, agent tools, sales flow (+ graph/conversation agents, eval_flow.py)
scripts/          benchmark.py
tests/            offline tests (hashing encoder)
```

## Not in scope

Argument/slot extraction and multi-turn context, to keep the model small and
portable.

## License

MIT — see [LICENSE](LICENSE).
