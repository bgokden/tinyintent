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
print(pred.intent, pred.score)      # refund 0.83
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
    -> intent=email (0.72)  (runner-up reminder 0.08) | drafted the email... | next=EMAIL_CONFIRM
[EMAIL_CONFIRM] user: 'yes go ahead'
    -> intent=confirm (0.88)  (runner-up cancel 0.01) | email sent | next=ROUTER
```

The ranking matters here: in `EMAIL_CONFIRM` the agent only accepts `confirm` or
`cancel`, so it picks the top-ranked intent among those rather than the global
best.

## Conversational flow (Retell-style)

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
  caller: 'we already use another provider'   ->  [objection 0.60, runner-up question 0.11]
agent [OBJECTION]: I hear you -- a lot of our customers felt the same...
  caller: 'okay that sounds interesting'   ->  [interested 0.82, runner-up commit 0.03]
agent [CLOSE]: I'd love to book you a free 15-minute assessment. Shall I set that up?
  caller: "yes let's do it"   ->  [commit 0.85, runner-up goodbye 0.04]
agent [BOOKED]: Fantastic, you're all set...
```

Both agents print the chosen intent's **score** and the **runner-up** (the
margin), so you can gate on confidence — e.g. re-prompt, confirm, or hand over
to a human when the top score is low or the margin is thin. Here the objection
lands at 0.60, a genuinely closer call than the 0.85 commit.

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
examples/         commerce, agent tools (graph_agent.py), sales flow (conversation_agent.py)
scripts/          benchmark.py
tests/            offline tests (hashing encoder)
```

## Not in scope

Argument/slot extraction and multi-turn context, to keep the model small and
portable.
