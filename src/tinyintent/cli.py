from __future__ import annotations

import argparse

from tinyintent.data import load_jsonl
from tinyintent.model import IntentModel


def cmd_train(args: argparse.Namespace) -> None:
    data = load_jsonl(args.data)
    model = IntentModel.fit(data)
    print(f"Trained on {len(data)} examples, {len(model.label_names)} intents")
    model.save(args.out)
    print(f"Saved model to {args.out}")


def cmd_evaluate(args: argparse.Namespace) -> None:
    model = IntentModel.load(args.model)
    for key, value in model.evaluate(load_jsonl(args.data)).as_dict().items():
        print(f"  {key}: {value}")


def _show(text: str, model: IntentModel) -> None:
    p = model.predict(text)
    print(f"\n> {text}")
    print(f"  intent: {p.intent}  ({p.score:.2f})")
    runners = ", ".join(f"{l} {s:.2f}" for l, s in p.ranking[1:3])
    if runners:
        print(f"  runners-up: {runners}")
    if p.explanation:
        print(f"  nearest example: \"{p.explanation['text']}\" "
              f"({p.explanation['similarity']:.2f})")


def cmd_predict(args: argparse.Namespace) -> None:
    model = IntentModel.load(args.model)
    if args.text:
        _show(" ".join(args.text), model)
        return
    print("Enter an utterance (empty line or Ctrl-D to quit).")
    while True:
        try:
            text = input("\nutterance> ").strip()
        except EOFError:
            print()
            break
        if not text:
            break
        _show(text, model)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tinyintent")
    sub = parser.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="train an intent model")
    t.add_argument("--data", required=True)
    t.add_argument("--out", required=True)
    t.set_defaults(func=cmd_train)

    e = sub.add_parser("evaluate", help="evaluate a saved model on a dataset")
    e.add_argument("--model", required=True)
    e.add_argument("--data", required=True)
    e.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("predict", help="predict intent for text (or interactive)")
    p.add_argument("--model", required=True)
    p.add_argument("text", nargs="*")
    p.set_defaults(func=cmd_predict)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
