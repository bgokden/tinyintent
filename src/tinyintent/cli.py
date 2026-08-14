from __future__ import annotations

import argparse

from tinyintent.data import load_jsonl
from tinyintent.model import IntentModel


def cmd_train(args: argparse.Namespace) -> None:
    data = load_jsonl(args.data)
    model = IntentModel.fit(data, device=args.device,
                            reranker=not args.no_reranker)
    stage = "encoder + head" if model.reranker is None else "encoder + head + reranker"
    print(f"Trained on {len(data)} examples, {len(model.label_names)} intents "
          f"({stage})")
    if model.oos_threshold is None:
        print("No 'oos' examples: the model will always decide. Add some to "
              "enable abstention.")
    else:
        print(f"Abstains below confidence {model.oos_threshold:.3f}")
    model.save(args.out)
    print(f"Saved model to {args.out}")


def cmd_evaluate(args: argparse.Namespace) -> None:
    model = IntentModel.load(args.model)
    data = load_jsonl(args.data)
    report = model.evaluate(data)
    for key, value in report.as_dict().items():
        print(f"  {key}: {value}")
    rejected = model.oos_rejection_rate(data)
    if rejected is not None:
        # evaluate() scores in-scope only, so this is the half it cannot see.
        print(f"  oos_rejection_rate: {rejected:.3f}")
    print()
    print(report.table())


def _show(text: str, model: IntentModel) -> None:
    p = model.predict(text)
    print(f"\n> {text}")
    flag = "  ABSTAIN (below threshold)" if p.abstain else ""
    print(f"  intent: {p.intent}  (score {p.score:.2f}, "
          f"confidence {p.confidence:.2f}){flag}")
    runners = ", ".join(f"{l} {s:.2f}" for l, s in p.ranking[1:3])
    if runners:
        print(f"  runners-up: {runners}")
    if p.explanation:
        print(f"  nearest example: \"{p.explanation['text']}\" "
              f"({p.explanation['similarity']:.2f})")


def cmd_predict(args: argparse.Namespace) -> None:
    model = IntentModel.load(args.model, device=args.device)
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
    t.add_argument("--device", default=None,
                   help="torch device, e.g. cuda / cpu / mps (default: auto)")
    t.add_argument("--no-reranker", action="store_true",
                   help="train the linear head only: much faster to train and "
                        "~15x faster to predict, for ~0.004 less top-1 accuracy")
    t.set_defaults(func=cmd_train)

    e = sub.add_parser("evaluate", help="evaluate a saved model on a dataset")
    e.add_argument("--model", required=True)
    e.add_argument("--data", required=True)
    e.add_argument("--device", default=None)
    e.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("predict", help="predict intent for text (or interactive)")
    p.add_argument("--model", required=True)
    p.add_argument("text", nargs="*")
    p.add_argument("--device", default=None)
    p.set_defaults(func=cmd_predict)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
