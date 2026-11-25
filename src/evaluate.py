"""Test-set evaluation of the 6 tuned pipelines: metrics, curves, confusion matrices,
error analysis, sample predictions and McNemar test (report Chapter 6)."""
import json
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import binomtest
from sklearn.metrics import (accuracy_score, average_precision_score, classification_report,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score, roc_curve)

from . import config as C
from .data import get_split
from .eda import savefig, set_style

LABELS = ["Ham", "Spam"]

# New messages written by the team (NOT from the dataset) for Table 6.5.
CUSTOM_MESSAGES = [
    ("Congratulations! You won a £1000 voucher, call 09061234567 now to claim", "Prize / premium-rate lure"),
    ("Hey, are we still meeting at 5?", "Casual personal message"),
    ("Your OTP for login is 482913. Do not share it with anyone.", "One-time password notice"),
    ("Your parcel is out for delivery today. Track it at www.example-courier.com/track", "Delivery notice with URL"),
    ("I got a free ticket for the match tonight, want to come with me?", "Casual message containing 'free'"),
]


def pretty(name):
    vec, clf = name.split("_", 1)
    return f"{vec} + {C.CLF_SHORT[clf]}"


def load_models():
    models = {}
    for vec in C.VECTORIZERS:
        for clf in C.CLASSIFIERS:
            models[C.model_name(vec, clf)] = joblib.load(C.model_file(vec, clf))
    return models


def score(pipe, X):
    """Continuous score for ROC/PR: P(spam) for NB/LR, signed margin for LinearSVC."""
    clf = pipe.named_steps["clf"]
    if hasattr(clf, "predict_proba"):
        return pipe.predict_proba(X)[:, 1], "P(spam)"
    return pipe.decision_function(X), "SVM decision score"


def inference_ms_per_1000(pipe, X, repeats=7):
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        pipe.predict(X)
        times.append(time.perf_counter() - t0)
    return 1000 * float(np.median(times)) * 1000 / len(X)


def compute_metrics(y, pred, s):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred),
        "f1": f1_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro"),
        "roc_auc": roc_auc_score(y, s),
        "pr_auc": average_precision_score(y, s),
        "mcc": matthews_corrcoef(y, pred),
        "fpr": fp / (fp + tn),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


def rank_models(metrics):
    """Best-model rule: highest test F1 (spam); tie-break lower FPR, then higher ROC-AUC."""
    return sorted(metrics, key=lambda m: (-round(metrics[m]["f1"], 10),
                                          round(metrics[m]["fpr"], 10),
                                          -round(metrics[m]["roc_auc"], 10)))


def mcnemar(y, pred_a, pred_b):
    a_ok, b_ok = pred_a == y, pred_b == y
    b = int(np.sum(a_ok & ~b_ok))   # A right, B wrong
    c = int(np.sum(~a_ok & b_ok))   # A wrong, B right
    n = b + c
    p = 1.0 if n == 0 else float(binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue)
    return {"a_correct_b_wrong": b, "a_wrong_b_correct": c, "n_discordant": n,
            "p_value_exact": p, "test": "McNemar exact (binomial) test, two-sided"}


# ----------------------------------------------------------------- figures
def _cm_panel(ax, cm, title, annot_pct=False, fontsize=10):
    row_pct = cm / cm.sum(axis=1, keepdims=True)
    if annot_pct:
        annot = np.array([[f"{cm[i, j]}\n({100 * row_pct[i, j]:.2f}%)" for j in range(2)] for i in range(2)])
    else:
        annot = cm.astype(str)
    sns.heatmap(row_pct, annot=annot, fmt="", cmap="Blues", vmin=0, vmax=1, cbar=False,
                linewidths=2, linecolor="white", square=True, ax=ax,
                annot_kws={"fontsize": fontsize})
    ax.set_xticklabels(LABELS)
    ax.set_yticklabels(LABELS, rotation=0)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)


def fig_confusion_grid(cms):
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.0))
    for r, vec in enumerate(C.VECTORIZERS):
        for c, clf in enumerate(C.CLASSIFIERS):
            name = C.model_name(vec, clf)
            _cm_panel(axes[r, c], cms[name], pretty(name))
            if c > 0:
                axes[r, c].set_ylabel("")
            if r == 0:
                axes[r, c].set_xlabel("")
    fig.tight_layout()
    return savefig(fig, "fig_6_1_confusion_matrices.png")


def _style_for(name):
    vec, clf = name.split("_", 1)
    return dict(color=C.CLF_COLORS[clf], ls=C.VEC_LINESTYLE[vec], lw=1.6)


def fig_roc(y, scores_, metrics):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.6))
    for name, s in scores_.items():
        fpr, tpr, _ = roc_curve(y, s)
        for ax in (ax1, ax2):
            ax.plot(fpr, tpr, label=f"{pretty(name)} (AUC {metrics[name]['roc_auc']:.4f})", **_style_for(name))
    ax1.plot([0, 1], [0, 1], color="#b5b3ad", lw=1, ls=":")
    ax1.set(xlim=(0, 1), ylim=(0, 1.01), xlabel="False positive rate", ylabel="True positive rate (recall)",
            title="ROC curves (full range)")
    ax2.set(xlim=(0, 0.10), ylim=(0.75, 1.005), xlabel="False positive rate", title="Zoom: FPR 0-0.10")
    ax2.legend(loc="lower right", frameon=True, framealpha=0.95, fontsize=7.5)
    fig.tight_layout()
    return savefig(fig, "fig_6_2_roc_curves.png")


def fig_pr(y, scores_, metrics):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.6))
    for name, s in scores_.items():
        p, r, _ = precision_recall_curve(y, s)
        for ax in (ax1, ax2):
            ax.plot(r, p, label=f"{pretty(name)} (AP {metrics[name]['pr_auc']:.4f})", **_style_for(name))
    prev = float(np.mean(y))
    ax1.axhline(prev, color="#b5b3ad", lw=1, ls=":")
    ax1.text(0.03, prev + 0.03, f"No-skill baseline = spam share ({prev:.3f})", fontsize=8, color=C.INK_2)
    ax1.set(xlim=(0, 1.0), ylim=(0, 1.02), xlabel="Recall (spam)", ylabel="Precision (spam)",
            title="Precision-Recall curves (full range)")
    ax2.set(xlim=(0.80, 1.0), ylim=(0.50, 1.01), xlabel="Recall (spam)", title="Zoom: recall 0.80-1.00")
    ax2.legend(loc="lower left", frameon=True, framealpha=0.95, fontsize=7.5)
    fig.tight_layout()
    return savefig(fig, "fig_6_3_pr_curves.png")


def fig_metric_comparison(metrics, order):
    """Dot plot of Precision / Recall / F1 per model (axis zoomed; bars would hide the differences)."""
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    y = np.arange(len(order))[::-1]
    markers = {"Precision": "o", "Recall": "s", "F1": "D"}
    keys = {"Precision": "precision", "Recall": "recall", "F1": "f1"}
    offs = {"Precision": 0.18, "Recall": 0.0, "F1": -0.18}
    for metric, col in C.METRIC_COLORS.items():
        vals = [metrics[m][keys[metric]] for m in order]
        ax.scatter(vals, y + offs[metric], color=col, marker=markers[metric], s=42, label=metric,
                   zorder=3, edgecolor="white", linewidth=0.8)
    lo = min(min(metrics[m][k] for k in ("precision", "recall", "f1")) for m in order)
    ax.set_xlim(np.floor(lo * 50) / 50 - 0.01, 1.005)
    ax.set_yticks(y, [pretty(m) for m in order])
    ax.set_xlabel("Score on the held-out test set (spam = positive class); axis does not start at 0")
    ax.set_title("Precision, Recall and F1 by model")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=False)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return savefig(fig, "fig_6_4_metric_comparison.png")


def fig_bow_vs_tfidf(metrics):
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    y = np.arange(len(C.CLASSIFIERS))[::-1]
    for yi, clf in zip(y, C.CLASSIFIERS):
        b = metrics[C.model_name("BoW", clf)]["f1"]
        t = metrics[C.model_name("TF-IDF", clf)]["f1"]
        ax.plot([b, t], [yi, yi], color="#c9c8c3", lw=2.5, zorder=1)
        ax.scatter([b], [yi], s=70, color=C.VEC_COLORS["BoW"], marker="o", zorder=3,
                   label="BoW" if yi == y[0] else None, edgecolor="white")
        ax.scatter([t], [yi], s=70, color=C.VEC_COLORS["TF-IDF"], marker="D", zorder=3,
                   label="TF-IDF" if yi == y[0] else None, edgecolor="white")
        left, right = (b, t) if b <= t else (t, b)
        ax.text(left - 0.0015, yi + 0.18, f"{left:.4f}", ha="right", fontsize=8, color=C.INK)
        ax.text(right + 0.0015, yi + 0.18, f"{right:.4f}", ha="left", fontsize=8, color=C.INK)
    vals = [metrics[m]["f1"] for m in metrics]
    ax.set_xlim(min(vals) - 0.02, max(vals) + 0.02)
    ax.set_ylim(-0.6, len(C.CLASSIFIERS) - 0.4)
    ax.set_yticks(y, ["Multinomial NB", "Logistic Regression", "Linear SVM"])
    ax.set_xlabel("Test F1 (spam); axis does not start at 0")
    ax.set_title("BoW vs TF-IDF: test F1 per classifier")
    ax.legend(loc="upper right", frameon=False)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return savefig(fig, "fig_6_5_bow_vs_tfidf.png")


def fig_cv_vs_test(cv, metrics, order):
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    y = np.arange(len(order))[::-1]
    cvm = cv.set_index("model")
    ax.errorbar([cvm.loc[m, "cv_f1_mean"] for m in order], y + 0.12,
                xerr=[cvm.loc[m, "cv_f1_std"] for m in order], fmt="o", color="#9a9893",
                ecolor="#9a9893", capsize=3, ms=6, label="5-fold CV F1 (mean ± std)")
    ax.scatter([metrics[m]["f1"] for m in order], y - 0.12, marker="D", s=46, color=C.INK,
               zorder=3, label="Test F1", edgecolor="white")
    ax.set_yticks(y, [pretty(m) for m in order])
    ax.set_xlabel("F1 (spam); axis does not start at 0")
    ax.set_title("Cross-validation F1 vs held-out test F1")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return savefig(fig, "fig_6_6_cv_vs_test.png")


def fig_best_cm(cm, name):
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    _cm_panel(ax, cm, f"Best model: {pretty(name)}", annot_pct=True, fontsize=12)
    return savefig(fig, "fig_6_8_best_confusion_matrix.png")


# ----------------------------------------------------------------- main
def run_evaluation():
    C.ensure_dirs()
    set_style()
    X_train, X_test, y_train, y_test = get_split()
    y = y_test.to_numpy()
    models = load_models()
    cv = pd.read_csv(C.RESULTS_DIR / "cv_summary.csv")

    metrics, preds, scores_, score_type, cms = {}, {}, {}, {}, {}
    for name, pipe in models.items():
        pred = pipe.predict(X_test)
        s, stype = score(pipe, X_test)
        m = compute_metrics(y, pred, s)
        m["inference_ms_per_1000"] = inference_ms_per_1000(pipe, X_test)
        m["score_type"] = stype
        metrics[name], preds[name], scores_[name], score_type[name] = m, pred, s, stype
        cms[name] = confusion_matrix(y, pred, labels=[0, 1])
        with open(C.RESULTS_DIR / f"clf_report_{name}.txt", "w") as f:
            f.write(f"{pretty(name)} - classification report on the test set (n={len(y)})\n\n")
            f.write(classification_report(y, pred, target_names=["ham", "spam"], digits=4))

    order = rank_models(metrics)
    best, second = order[0], order[1]
    cv_order = cv.sort_values("cv_f1_mean", ascending=False)["model"].tolist()
    mc = mcnemar(y, preds[best], preds[second])
    mc.update({"model_a": best, "model_b": second})

    # Table 6.1 (test) and Table 6.2 (CV)
    t61 = pd.DataFrame([{
        "Model": pretty(n), "Vectorizer": n.split("_")[0], "Classifier": n.split("_", 1)[1],
        "Accuracy": metrics[n]["accuracy"], "Precision": metrics[n]["precision"],
        "Recall": metrics[n]["recall"], "F1": metrics[n]["f1"], "Macro-F1": metrics[n]["macro_f1"],
        "ROC-AUC": metrics[n]["roc_auc"], "PR-AUC": metrics[n]["pr_auc"], "MCC": metrics[n]["mcc"],
        "FPR": metrics[n]["fpr"], "FP": metrics[n]["fp"], "FN": metrics[n]["fn"],
        "Inference ms per 1000": metrics[n]["inference_ms_per_1000"],
    } for n in metrics]).round(4)
    t61.to_csv(C.TABLE_DIR / "table_6_1_results.csv", index=False)

    cvm = cv.set_index("model")
    t62 = pd.DataFrame([{
        "Model": pretty(n), "CV F1 mean": cvm.loc[n, "cv_f1_mean"], "CV F1 std": cvm.loc[n, "cv_f1_std"],
        "CV train F1": cvm.loc[n, "cv_train_f1_mean"], "Test F1": metrics[n]["f1"],
        "Refit time (s)": cvm.loc[n, "refit_time_s"], "Grid search time (s)": cvm.loc[n, "search_time_s"],
        "Candidates": int(cvm.loc[n, "n_candidates"]), "Vocabulary": int(cvm.loc[n, "vocab_size"]),
    } for n in metrics]).round(4)
    t62.to_csv(C.TABLE_DIR / "table_6_2_cv_results.csv", index=False)

    # Figures
    print("Figures:")
    fig_confusion_grid(cms)
    fig_roc(y, scores_, metrics)
    fig_pr(y, scores_, metrics)
    fig_metric_comparison(metrics, list(metrics))
    fig_bow_vs_tfidf(metrics)
    fig_cv_vs_test(cv, metrics, list(metrics))
    fig_best_cm(cms[best], best)

    # Error analysis (best model)
    best_pipe = models[best]
    thr = 0.5 if score_type[best] == "P(spam)" else 0.0
    df_test = pd.DataFrame({"text": X_test.to_numpy(), "actual": y, "predicted": preds[best],
                            "score": scores_[best]})
    err = df_test[df_test.actual != df_test.predicted].copy()
    err["error_type"] = np.where(err.predicted == 1, "FP (ham marked spam)", "FN (spam missed)")
    err["actual"] = err.actual.map({0: "ham", 1: "spam"})
    err["predicted"] = err.predicted.map({0: "ham", 1: "spam"})
    err["distance_from_threshold"] = (err.score - thr).abs()
    err = err.sort_values(["error_type", "distance_from_threshold"], ascending=[True, False])
    err[["error_type", "text", "actual", "predicted", "score"]].round(4).to_csv(
        C.TABLE_DIR / "table_6_3_misclassified.csv", index=False)

    # Sample predictions from the test set: 2 confident ham, 2 confident spam, 1 hard case
    correct = df_test[df_test.actual == df_test.predicted]
    rng = np.random.RandomState(C.RANDOM_STATE)
    picks = []
    for cls in (0, 1):
        pool = correct[(correct.actual == cls) & (correct.text.str.len().between(25, 140))]
        picks += list(pool.sample(2, random_state=rng).index)
    hard_pool = err.index if len(err) else df_test.index
    hard = (df_test.loc[hard_pool, "score"] - thr).abs().idxmin()
    picks.append(hard)
    samp = df_test.loc[picks].copy()
    samp["case"] = ["Typical ham", "Typical ham", "Typical spam", "Typical spam", "Hard case (closest to threshold)"]
    samp["actual"] = samp.actual.map({0: "ham", 1: "spam"})
    samp["predicted"] = samp.predicted.map({0: "ham", 1: "spam"})
    samp["correct"] = np.where(samp.actual == samp.predicted, "Yes", "No")
    samp[["case", "text", "actual", "predicted", "score", "correct"]].round(4).to_csv(
        C.TABLE_DIR / "table_6_4_sample_predictions.csv", index=False)

    # Custom messages (not from the dataset)
    texts = [t for t, _ in CUSTOM_MESSAGES]
    cs, _ = score(best_pipe, pd.Series(texts))
    cp = best_pipe.predict(pd.Series(texts))
    pd.DataFrame({"source": "Written by the team (not from the dataset)",
                  "description": [d for _, d in CUSTOM_MESSAGES], "text": texts,
                  "predicted": np.where(cp == 1, "spam", "ham"), "score": np.round(cs, 4)}).to_csv(
        C.TABLE_DIR / "table_6_5_custom_predictions.csv", index=False)

    out = {
        "n_test": int(len(y)), "n_test_spam": int(y.sum()), "n_test_ham": int((y == 0).sum()),
        "models": metrics,
        "ranking_by_rule": order,
        "ranking_by_cv_f1": cv_order,
        "best_model": best, "best_model_pretty": pretty(best),
        "best_model_score_type": score_type[best], "best_model_threshold": thr,
        "best_params": json.loads(cvm.loc[best, "best_params"]),
        "selection_rule": "Highest test F1 (spam); tie-break by lower FPR, then higher ROC-AUC",
        "mcnemar_top2": mc,
        "n_errors_best": int(len(err)),
        "n_fp_best": int((err.error_type.str.startswith("FP")).sum()),
        "n_fn_best": int((err.error_type.str.startswith("FN")).sum()),
    }
    with open(C.RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(out, f, indent=2)
    print(t61[["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC", "FPR", "FP", "FN"]].to_string(index=False))
    print(f"Best model: {pretty(best)}; McNemar vs {pretty(second)}: p = {mc['p_value_exact']:.4f}")
    return out


if __name__ == "__main__":
    run_evaluation()
