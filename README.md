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
model = IntentModel.fit(data)          # trains the whole pipeline
model.save("model")

print(model.classify("I want my money back for order 883"))   # refund

pred = model.predict("I want my money back for order 883")
print(pred.intent, pred.score)      # refund 0.80
print(pred.ranking[:3])             # ranked intents
print(pred.explanation)             # nearest labelled example
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

`predict` returns a **single confidence** (`Prediction.score`) — the reranked
softmax over the top candidates — and `ranking` is ordered by that same number,
so the top is always the decision and the margin to the runner-up is
non-negative. That is what you gate on: confident transitions land around
0.85-0.91; a genuinely ambiguous reply drops well below.

`conversation_agent.py` uses it as a gate (`MIN_SCORE` / `MIN_MARGIN`): when the
best edge is too weak, the agent **stays in the node** (a self-loop -- a normal
FSM choice) and asks the caller to clarify, then routes cleanly next turn.

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
(10-20 works well). The reserved label `oos` marks out-of-scope examples; they
are ignored during training and evaluation.

```json
{"text": "cancel my order", "label": "cancel_order"}
{"text": "what's the weather", "label": "oos"}
```

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
