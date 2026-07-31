"""A state-machine (graph) agent that routes with tinyintent.

The agent is a graph: each node (state) allows a subset of intents, and each
allowed intent is an edge to the next node. tinyintent ranks the user's
utterance and the agent follows the highest-ranked *allowed* edge -- so the
same classifier drives tool selection and the conversation's control flow.

Most tools run and return to the router; email is a two-step flow
(draft -> confirm/cancel) to show a genuine multi-node path. The routed intent
is the tool you would actually call, or inject into an LLM prompt.

    uv run python examples/graph_agent.py
"""

from __future__ import annotations

from tinyintent import IntentModel, load_jsonl

# state -> {allowed intent: next state}
GRAPH: dict[str, dict[str, str]] = {
    "ROUTER": {
        "web_search": "ROUTER",
        "calculator": "ROUTER",
        "weather": "ROUTER",
        "calendar": "ROUTER",
        "translate": "ROUTER",
        "reminder": "ROUTER",
        "code_run": "ROUTER",
        "email": "EMAIL_CONFIRM",
        "exit": "END",
    },
    "EMAIL_CONFIRM": {
        "confirm": "ROUTER",
        "cancel": "ROUTER",
    },
}

ACTIONS = {
    "web_search": "searching the web",
    "calculator": "computing the result",
    "weather": "fetching the forecast",
    "calendar": "updating your calendar",
    "translate": "translating the text",
    "reminder": "setting a reminder",
    "code_run": "running the code",
    "email": "drafted the email -- confirm to send?",
    "confirm": "email sent",
    "cancel": "email discarded",
    "exit": "goodbye",
}


def route(model: IntentModel, state: str, utterance: str) -> str:
    """Highest-ranked intent that is a valid edge out of the current state."""

    allowed = GRAPH[state]
    for intent, _score in model.predict(utterance).ranking:
        if intent in allowed:
            return intent
    return next(iter(allowed))          # fallback: first allowed edge


def run(model: IntentModel, messages: list[str]) -> None:
    state = "ROUTER"
    for msg in messages:
        intent = route(model, state, msg)
        nxt = GRAPH[state][intent]
        print(f"[{state}] user: {msg!r}")
        print(f"    -> intent={intent} | {ACTIONS[intent]} | next={nxt}")
        state = nxt
        if state == "END":
            break


def main() -> None:
    model = IntentModel.fit(load_jsonl("examples/agent_tools.jsonl"))
    conversation = [
        "what's the weather in Paris tomorrow",
        "calculate 15% of 240",
        "search the web for the tallest mountain",
        "send an email to Sam about lunch",   # enters the email flow
        "yes go ahead",                        # confirm -> back to router
        "remind me to call the dentist at 3pm",
        "that's all, thanks",                  # exit -> END
    ]
    run(model, conversation)


if __name__ == "__main__":
    main()
