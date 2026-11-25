"""Model interpretation: most spam- and ham-indicative tokens (report Fig. 6.7, §6.4)."""
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C
from .eda import savefig, set_style
from .evaluate import pretty


def _best_of(metrics, clf):
    """Best model of one classifier family by the same rule as the overall best model."""
    names = [n for n in metrics["ranking_by_rule"] if n.endswith("_" + clf)]
    return names[0]


def linear_weights(pipe):
    vec, clf = pipe.named_steps["vec"], pipe.named_steps["clf"]
    return np.asarray(vec.get_feature_names_out()), clf.coef_.ravel()


def nb_log_ratio(pipe):
    vec, clf = pipe.named_steps["vec"], pipe.named_steps["clf"]
    # log P(w | spam) - log P(w | ham)
    return np.asarray(vec.get_feature_names_out()), clf.feature_log_prob_[1] - clf.feature_log_prob_[0]


def top_k(names, w, k=20):
    order = np.argsort(w)
    spam = [(names[i], float(w[i])) for i in order[::-1][:k]]
    ham = [(names[i], float(w[i])) for i in order[:k]]
    return spam, ham


def fig_top_features(lr, nb, lr_model, nb_model):
    """2 x 2: rows = (LR coefficients, NB log-probability ratio), cols = (spam, ham)."""
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 9.4))
    rows = [(lr, f"(a) Logistic Regression coefficients - {pretty(lr_model)}", "coefficient"),
            (nb, f"(b) Naive Bayes log P(w|spam) - log P(w|ham) - {pretty(nb_model)}", "log-ratio")]
    for r, ((spam, ham), title, unit) in enumerate(rows):
        k = len(spam)
        y = np.arange(k)[::-1]
        a1, a2 = axes[r]
        a1.barh(y, [v for _, v in spam], color=C.SPAM_COLOR, height=0.72)
        a1.set_yticks(y, [t for t, _ in spam])
        a1.set_title(f"Top {k} spam-indicative tokens", fontsize=10)
        a1.set_xlabel(f"{unit} (positive = spam)")
        a2.barh(y, [abs(v) for _, v in ham], color=C.HAM_COLOR, height=0.72)
        a2.set_yticks(y, [t for t, _ in ham])
        a2.set_title(f"Top {k} ham-indicative tokens", fontsize=10)
        a2.set_xlabel(f"|{unit}| (negative = ham)")
        for a in (a1, a2):
            a.grid(axis="y", visible=False)
            a.tick_params(axis="y", labelsize=8.5)
        a1.text(0.0, 1.10, title, transform=a1.transAxes, fontsize=10.5, fontweight="bold", color=C.INK)
    fig.tight_layout(h_pad=3.2)
    return savefig(fig, "fig_6_7_top_features.png")


def run_interpretation():
    set_style()
    with open(C.RESULTS_DIR / "metrics.json") as f:
        metrics = json.load(f)

    def load(name):
        vec, clf = name.split("_", 1)
        return joblib.load(C.model_file(vec, clf))

    lr_name = _best_of(metrics, "LogisticRegression")
    nb_name = _best_of(metrics, "MultinomialNB")
    svm_name = _best_of(metrics, "LinearSVM")
    best = metrics["best_model"]

    lr_spam, lr_ham = top_k(*linear_weights(load(lr_name)))
    nb_spam, nb_ham = top_k(*nb_log_ratio(load(nb_name)))
    svm_spam, svm_ham = top_k(*linear_weights(load(svm_name)))
    fig_top_features((lr_spam, lr_ham), (nb_spam, nb_ham), lr_name, nb_name)

    rows = []
    for i in range(20):
        rows.append({
            "Rank": i + 1,
            "LR spam token": lr_spam[i][0], "LR spam coef": round(lr_spam[i][1], 4),
            "LR ham token": lr_ham[i][0], "LR ham coef": round(lr_ham[i][1], 4),
            "NB spam token": nb_spam[i][0], "NB spam log-ratio": round(nb_spam[i][1], 4),
            "NB ham token": nb_ham[i][0], "NB ham log-ratio": round(nb_ham[i][1], 4),
            "SVM spam token": svm_spam[i][0], "SVM spam weight": round(svm_spam[i][1], 4),
            "SVM ham token": svm_ham[i][0], "SVM ham weight": round(svm_ham[i][1], 4),
        })
    pd.DataFrame(rows).to_csv(C.TABLE_DIR / "table_6_6_top_features.csv", index=False)

    fam = {"LogisticRegression": (lr_spam, lr_ham), "MultinomialNB": (nb_spam, nb_ham),
           "LinearSVM": (svm_spam, svm_ham)}
    b_spam, b_ham = fam[best.split("_", 1)[1]]
    out = {"lr_model": lr_name, "nb_model": nb_name, "svm_model": svm_name, "best_model": best,
           "best_top_spam": [t for t, _ in b_spam[:10]], "best_top_ham": [t for t, _ in b_ham[:10]],
           "lr_top_spam": [t for t, _ in lr_spam[:10]], "nb_top_spam": [t for t, _ in nb_spam[:10]],
           "svm_top_spam": [t for t, _ in svm_spam[:10]],
           "lr_top_ham": [t for t, _ in lr_ham[:10]], "nb_top_ham": [t for t, _ in nb_ham[:10]],
           "svm_top_ham": [t for t, _ in svm_ham[:10]]}
    with open(C.RESULTS_DIR / "interpretation.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Top spam tokens (best model):", out["best_top_spam"][:5])
    return out


if __name__ == "__main__":
    run_interpretation()
