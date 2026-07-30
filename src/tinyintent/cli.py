from __future__ import annotations

import argparse
from pathlib import Path

from tinyintent.data import load_jsonl, split
from tinyintent.encoder import HashingEncoder, SentenceEncoder
from tinyintent.model import IntentModel


def _make_encoder(name: str):
    if name == "minilm":
        return SentenceEncoder()
    if name == "hashing":
        return HashingEncoder()
    raise SystemExit(f"unknown encoder: {name} (choose minilm or hashing)")


def cmd_train(args: argparse.Namespace) -> None:
    data = load_jsonl(args.data)
    fit_set, cal_set = split(data, test_frac=args.calibrate_frac, seed=args.seed)

    encoder = _make_encoder(args.encoder)
    model = IntentModel.fit(fit_set, head=args.head, encoder=encoder)
    policy = model.calibrate(cal_set, max_false_fire=args.max_false_fire)
    model.save(args.out)

    print(f"Trained head '{args.head}' on {len(fit_set)} examples, "
          f"{len(model.label_names)} intents: {model.label_names}")
    print(f"Calibrated on {len(cal_set)}: threshold={policy.threshold:.3f} "
          f"margin={policy.margin:.3f}")
    print(f"Saved model to {args.out}")


def cmd_evaluate(args: argparse.Namespace) -> None:
    model = IntentModel.load(args.model)
    report = model.evaluate(load_jsonl(args.data))
    for key, value in report.as_dict().items():
        print(f"  {key}: {value}")


def _show(text: str, model: IntentModel) -> None:
    p = model.predict(text)
    label = p.intent if p.intent is not None else "(abstain)"
    print(f"\n> {text}")
    print(f"  intent: {label}   score: {p.score:.3f}")
    alts = ", ".join(f"{l} {s:.2f}" for l, s in p.alternatives)
    print(f"  ranked: {alts}")
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

    t = sub.add_parser("train", help="train and calibrate an intent model")
    t.add_argument("--data", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--head", default="prototype", choices=["prototype", "logreg", "mlp"])
    t.add_argument("--encoder", default="minilm", choices=["minilm", "hashing"])
    t.add_argument("--calibrate-frac", type=float, default=0.25)
    t.add_argument("--max-false-fire", type=float, default=0.02)
    t.add_argument("--seed", type=int, default=0)
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
