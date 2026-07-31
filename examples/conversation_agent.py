"""A conversational-flow agent (Retell-style) driven by tinyintent.

The conversation is a graph. Each node is a phase of the call with a line the
agent says and a set of edges keyed by *conversational* intent (interested,
objection, commit, handover, ...). tinyintent classifies the caller's reply and
the agent follows the highest-ranked edge that is valid in the current node, so
one intent model drives the whole call flow:

    INTRODUCTION -> PITCH -> QUESTION -> OBJECTION -> CLOSE -> BOOKED
                 \-> EXIT   \-> HANDOVER

    uv run python examples/conversation_agent.py
"""

from __future__ import annotations

from tinyintent import IntentModel, load_jsonl

# node -> what the agent says + {intent: next node}
NODES: dict[str, dict] = {
    "INTRODUCTION": {
        "say": "Hi, this is Ava from Acme Solar. Do you have a quick minute?",
        "edges": {"greeting": "INTRODUCTION", "interested": "PITCH", "question": "QUESTION",
                  "objection": "OBJECTION", "not_interested": "EXIT",
                  "handover": "HANDOVER", "goodbye": "EXIT"},
    },
    "PITCH": {
        "say": "We help homeowners cut their electric bill with rooftop solar -- "
               "often around 30% savings, with nothing upfront.",
        "edges": {"interested": "CLOSE", "commit": "CLOSE", "question": "QUESTION",
                  "objection": "OBJECTION", "not_interested": "EXIT",
                  "handover": "HANDOVER", "goodbye": "EXIT"},
    },
    "QUESTION": {
        "say": "Good question -- most homeowners pay nothing upfront and finance it "
               "straight out of the monthly savings.",
        "edges": {"interested": "PITCH", "commit": "CLOSE", "question": "QUESTION",
                  "objection": "OBJECTION", "not_interested": "EXIT",
                  "handover": "HANDOVER", "goodbye": "EXIT"},
    },
    "OBJECTION": {
        "say": "I hear you -- a lot of our customers felt the same, then saved more "
               "after switching. Could I show you a quick comparison?",
        "edges": {"interested": "CLOSE", "commit": "CLOSE", "question": "QUESTION",
                  "objection": "OBJECTION", "not_interested": "EXIT",
                  "handover": "HANDOVER", "goodbye": "EXIT"},
    },
    "CLOSE": {
        "say": "I'd love to book you a free 15-minute assessment. Shall I set that up?",
        "edges": {"commit": "BOOKED", "interested": "CLOSE", "question": "QUESTION",
                  "objection": "OBJECTION", "not_interested": "EXIT",
                  "handover": "HANDOVER", "goodbye": "EXIT"},
    },
    "BOOKED": {"say": "Fantastic, you're all set -- we'll email the details. Have a great day!",
               "edges": {}},
    "HANDOVER": {"say": "Of course, let me connect you with a specialist right now.",
                 "edges": {}},
    "EXIT": {"say": "No problem at all -- thanks for your time, take care!", "edges": {}},
}


# Confidence gate: if the best edge is weak or too close to the runner-up, the
# agent does NOT transition -- it stays in the node and asks the caller to
# clarify. The model always decides; the agent decides whether to trust it.
MIN_SCORE = 0.45
MIN_MARGIN = 0.10


def rank_allowed(model: IntentModel, node: str, utterance: str) -> list[tuple[str, float]]:
    """Intents that are valid edges out of the node, with scores, best first."""

    edges = NODES[node]["edges"]
    ranked = [(intent, score) for intent, score in model.predict(utterance).ranking
              if intent in edges]
    return ranked or [(next(iter(edges)), 0.0)]


def run(model: IntentModel, caller_turns: list[str]) -> None:
    node = "INTRODUCTION"
    print(f"agent [{node}]: {NODES[node]['say']}")
    for utterance in caller_turns:
        ranked = rank_allowed(model, node, utterance)
        intent, score = ranked[0]
        runner, runner_score = ranked[1] if len(ranked) > 1 else (None, 0.0)
        margin = score - runner_score

        if score < MIN_SCORE or margin < MIN_MARGIN:
            # too uncertain: no transition, stay put and ask to clarify
            print(f"  caller: {utterance!r}   ->  [uncertain: {intent} {score:.2f}, "
                  f"margin {margin:.2f}] -- no transition")
            print(f"agent [{node}]: Sorry, I didn't quite catch that -- could you say a bit more?")
            continue

        node = NODES[node]["edges"][intent]
        extra = f", runner-up {runner} {runner_score:.2f}" if runner else ""
        print(f"  caller: {utterance!r}   ->  [{intent} {score:.2f}{extra}]")
        print(f"agent [{node}]: {NODES[node]['say']}")
        if not NODES[node]["edges"]:            # terminal node ends the call
            break
    print()


def main() -> None:
    model = IntentModel.fit(load_jsonl("examples/sales_flow.jsonl"))

    print("=== call 1: full journey to a booking ===")
    run(model, [
        "sure, tell me more",
        "how much does it cost",
        "we already use another provider",
        "okay that sounds interesting",
        "yes let's do it",
    ])

    print("=== call 2: not interested ===")
    run(model, ["not interested, please remove me from your list"])

    print("=== call 3: hands off to a human ===")
    run(model, ["sure, tell me more", "can I talk to a real person"])

    print("=== call 4: vague reply -> clarify, no transition, then resolves ===")
    run(model, [
        "well, it depends",          # too vague to route -> agent clarifies, stays
        "yeah okay, tell me more",   # now clear -> interested -> PITCH
    ])


if __name__ == "__main__":
    main()
