"""Appendix B: preprocessing ablation, learning curve and top-10 CV grid per model."""
import json
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score, learning_curve

from . import config as C
from .data import get_split
from .eda import savefig, set_style
from .evaluate import pretty
from .preprocess import STOP_WORDS, clean_text, clean_text_no_norm, lowercase_only, stem_tokenize
from .train import cv_splitter

warnings.filterwarnings("ignore", message="Your stop_words may be inconsistent")
warnings.filterwarnings("ignore", message="The parameter 'token_pattern' will not be used")

VARIANTS = [
    ("a", "Full preprocessing (main pipeline)",
     dict(preprocessor=clean_text, token_pattern=C.TOKEN_PATTERN, stop_words=STOP_WORDS)),
    ("b", "No stop-word removal",
     dict(preprocessor=clean_text, token_pattern=C.TOKEN_PATTERN, stop_words=None)),
    ("c", "No number/URL/currency normalisation",
     dict(preprocessor=clean_text_no_norm, token_pattern=C.TOKEN_PATTERN, stop_words=STOP_WORDS)),
    ("d", "Full preprocessing + Porter stemming",
     dict(preprocessor=clean_text, tokenizer=stem_tokenize, token_pattern=None, stop_words=None)),
    ("e", "Raw text, lowercasing only",
     dict(preprocessor=lowercase_only, token_pattern=r"(?u)\b\w\w+\b", stop_words=None)),
]


def run_ablation():
    C.ensure_dirs()
    set_style()
    with open(C.RESULTS_DIR / "metrics.json") as f:
        metrics = json.load(f)
    best = metrics["best_model"]
    vec_name, clf_name = best.split("_", 1)
    base = joblib.load(C.model_file(vec_name, clf_name))
    X_train, X_test, y_train, y_test = get_split()

    rows = []
    for key, desc, params in VARIANTS:
        pipe = clone(base)
        pipe.set_params(**{f"vec__{k}": v for k, v in params.items()})
        cv = cross_val_score(pipe, X_train, y_train, scoring=C.SCORING, cv=cv_splitter(), n_jobs=-1)
        pipe.fit(X_train, y_train)
        test_f1 = f1_score(y_test, pipe.predict(X_test))
        rows.append({"Variant": f"({key})", "Description": desc, "CV F1 mean": cv.mean(), "CV F1 std": cv.std(),
                     "Test F1": test_f1, "Vocabulary": len(pipe.named_steps["vec"].vocabulary_)})
        print(f"  ({key}) {desc:42s} CV {cv.mean():.4f} ± {cv.std():.4f} | test {test_f1:.4f}")
    tab = pd.DataFrame(rows).round(4)
    tab.to_csv(C.TABLE_DIR / "table_b_1_ablation.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    y = np.arange(len(tab))[::-1]
    ax.errorbar(tab["CV F1 mean"], y + 0.12, xerr=tab["CV F1 std"], fmt="o", color="#9a9893",
                ecolor="#9a9893", capsize=3, ms=6, label="5-fold CV F1 (mean ± std)")
    ax.scatter(tab["Test F1"], y - 0.12, marker="D", s=46, color=C.INK, zorder=3,
               label="Test F1", edgecolor="white")
    ax.set_yticks(y, [f"{v} {d}" for v, d in zip(tab.Variant, tab.Description)])
    ax.set_xlabel("F1 (spam); axis does not start at 0")
    ax.set_title(f"Preprocessing ablation ({pretty(best)}, tuned hyperparameters)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    savefig(fig, "fig_b_1_ablation.png")

    # Learning curve of the best model
    sizes, tr, va = learning_curve(clone(base), X_train, y_train, train_sizes=np.linspace(0.1, 1.0, 8),
                                   cv=cv_splitter(), scoring=C.SCORING, n_jobs=-1, shuffle=True,
                                   random_state=C.RANDOM_STATE)
    lc = pd.DataFrame({"Training messages": sizes, "Train F1 mean": tr.mean(1), "Train F1 std": tr.std(1),
                       "CV F1 mean": va.mean(1), "CV F1 std": va.std(1)}).round(4)
    lc.to_csv(C.TABLE_DIR / "table_b_3_learning_curve.csv", index=False)
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    for col, lab, color, mk, ls in [("Train", "Training F1", "#9a9893", "o", "--"), ("CV", "Cross-validation F1", C.INK, "D", "-")]:
        m, s = lc[f"{col} F1 mean"], lc[f"{col} F1 std"]
        ax.plot(lc["Training messages"], m, marker=mk, color=color, lw=2, ms=5, label=lab, ls=ls)
        ax.fill_between(lc["Training messages"], m - s, m + s, color=color, alpha=0.15, lw=0)
    ax.set_xlabel("Number of training messages")
    ax.set_ylabel("F1 (spam)")
    ax.set_title(f"Learning curve ({pretty(best)})")
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    savefig(fig, "fig_b_2_learning_curve.png")

    # Top-10 CV grid configurations per model
    top = []
    for vec in C.VECTORIZERS:
        for clf in C.CLASSIFIERS:
            name = C.model_name(vec, clf)
            res = pd.read_csv(C.RESULTS_DIR / f"cv_results_{name}.csv")
            res = res.sort_values(["rank_test_score", "mean_test_score"], ascending=[True, False]).head(10)
            for _, r in res.iterrows():
                p = {k.replace("param_", ""): r[k] for k in res.columns if k.startswith("param_")}
                top.append({"Model": pretty(name), "Rank": int(r["rank_test_score"]),
                            "Parameters": "; ".join(f"{k.split('__')[1]}={v}" for k, v in p.items()),
                            "CV F1 mean": round(r["mean_test_score"], 4), "CV F1 std": round(r["std_test_score"], 4)})
    pd.DataFrame(top).to_csv(C.TABLE_DIR / "table_b_2_cv_top10.csv", index=False)
    return tab, lc


if __name__ == "__main__":
    run_ablation()
