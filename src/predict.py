"""Command-line prediction with the best saved pipeline.

Usage:
    python -m src.predict "Congratulations! You won a prize, call 09061234567"
    python -m src.predict "msg one" "msg two"
"""
import html
import json
import sys

import joblib

from . import config as C


def load_best():
    with open(C.RESULTS_DIR / "metrics.json") as f:
        m = json.load(f)
    vec, clf = m["best_model"].split("_", 1)
    return joblib.load(C.model_file(vec, clf)), m


def predict(messages):
    pipe, m = load_best()
    texts = [html.unescape(t).strip() for t in messages]
    preds = pipe.predict(texts)
    clf = pipe.named_steps["clf"]
    if hasattr(clf, "predict_proba"):
        scores, kind = pipe.predict_proba(texts)[:, 1], "P(spam)"
    else:
        scores, kind = pipe.decision_function(texts), "decision score"
    return [(t, "Spam (1)" if p == 1 else "Ham (0)", float(s), kind) for t, p, s in zip(messages, preds, scores)], m


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print(__doc__)
        return 1
    results, m = predict(argv)
    print(f"Model: {m['best_model_pretty']} (outputs/models)")
    for text, label, s, kind in results:
        print(f"Message : {text}")
        print(f"Label   : {label}   {kind} = {s:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
