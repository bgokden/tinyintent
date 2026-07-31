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


def route(model: IntentModel, node: str, utterance: str) -> str:
    """Highest-ranked intent that is a valid edge out of the current node."""

    edges = NODES[node]["edges"]
    for intent, _score in model.predict(utterance).ranking:
        if intent in edges:
            return intent
    return next(iter(edges))


def run(model: IntentModel, caller_turns: list[str]) -> None:
    node = "INTRODUCTION"
    print(f"agent [{node}]: {NODES[node]['say']}")
    for utterance in caller_turns:
        intent = route(model, node, utterance)
        node = NODES[node]["edges"][intent]
        print(f"  caller: {utterance!r}   ->  [{intent}]")
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


if __name__ == "__main__":
    main()
