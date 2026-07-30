from __future__ import annotations

import argparse

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

    if args.finetune:
        from tinyintent.finetune import finetune_encoder
        from tinyintent.encoder import DEFAULT_MODEL

        encoder = finetune_encoder(fit_set, out_dir=f"{args.out}/encoder", base_model=DEFAULT_MODEL)
    else:
        encoder = _make_encoder(args.encoder)

    model = IntentModel.fit(fit_set, encoder=encoder, classifier=args.classifier)
    print(f"Trained on {len(fit_set)} examples, {len(model.label_names)} intents "
          f"(classifier={args.classifier})")
    if args.method == "none":
        print("No policy: predict always decides the single best intent")
    else:
        policy = model.calibrate(cal_set, risk=args.risk, method=args.method)
        print(f"Calibrated on {len(cal_set)} using policy '{policy.name}'")
    model.save(args.out)
    print(f"Saved model to {args.out}")


def cmd_evaluate(args: argparse.Namespace) -> None:
    model = IntentModel.load(args.model)
    for key, value in model.evaluate(load_jsonl(args.data)).as_dict().items():
        print(f"  {key}: {value}")


def _show(text: str, model: IntentModel) -> None:
    p = model.predict(text)
    print(f"\n> {text}")
    print(f"  decision: {p.decision}" + (f"  ->  {p.intent}" if p.intent else ""))
    set_str = ", ".join(f"{l} {s:.2f}" for l, s in p.set_) or "(empty)"
    print(f"  prediction set: {set_str}")
    print(f"  top: {p.top[0]} {p.top[1]:.2f}")
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
    t.add_argument("--encoder", default="minilm", choices=["minilm", "hashing"])
    t.add_argument("--method", default="none", choices=["none", "aps", "lac", "gate"],
                   help="none = always decide top-1; others add abstain/reject")
    t.add_argument("--classifier", default="linear", choices=["linear", "exemplar"])
    t.add_argument("--finetune", action="store_true", help="contrastively fine-tune the encoder")
    t.add_argument("--risk", type=float, default=0.1, help="conformal alpha")
    t.add_argument("--calibrate-frac", type=float, default=0.25)
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
