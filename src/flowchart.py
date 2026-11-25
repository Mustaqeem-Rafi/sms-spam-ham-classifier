"""Draw the project workflow (report Fig. 5.1) with matplotlib."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from . import config as C
from .eda import savefig, set_style

BOX_FILL = "#eef3fb"
BOX_EDGE = "#2a78d6"
PIPE_FILL = "#fdf0ea"
PIPE_EDGE = "#eb6834"


def _box(ax, x, y, w, h, title, sub, fill=BOX_FILL, edge=BOX_EDGE):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fill, ec=edge, lw=1.3, zorder=2))
    ax.text(x, y + (0.13 if sub else 0), title, ha="center", va="center", fontsize=9.5,
            fontweight="bold", color=C.INK, zorder=3)
    if sub:
        ax.text(x, y - 0.17, sub, ha="center", va="center", fontsize=7.6, color=C.INK_2, zorder=3,
                linespacing=1.15)


def _arrow(ax, p, q):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=12, lw=1.2,
                                 color="#52514e", zorder=1, shrinkA=0, shrinkB=0))


def draw_workflow():
    set_style()
    fig, ax = plt.subplots(figsize=(7.2, 7.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10.8)
    ax.axis("off")
    w, h = 3.6, 0.78
    xl, xr = 2.6, 7.4

    with open(C.RESULTS_DIR / "data_log.json") as f:
        n_raw = json.load(f)["n_raw"]
    # Left column (top to bottom): data side
    left = [
        (9.8, "1. Dataset", f"UCI SMS Spam Collection\n{n_raw:,} labelled messages"),
        (8.5, "2. Data cleaning", "ham=0 / spam=1, HTML unescape,\nremove exact duplicates"),
        (7.2, "3. Exploratory data analysis", "class balance, length, digits,\ntop words, correlations"),
        (5.9, "4. Train/Test split", "stratified 80/20, random_state=42"),
    ]
    for y, t, s in left:
        _box(ax, xl, y, w, h, t, s)
    for (y1, *_), (y2, *_) in zip(left, left[1:]):
        _arrow(ax, (xl, y1 - h / 2), (xl, y2 + h / 2))

    # Pipeline group (right column)
    ax.add_patch(FancyBboxPatch((5.35, 3.65), 4.1, 6.55, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc="none", ec=PIPE_EDGE, lw=1.2, ls=(0, (4, 3)), zorder=0))
    ax.text(7.4, 10.32, "scikit-learn Pipeline (fitted on training data only)", ha="center", va="bottom",
            fontsize=8.2, color="#b54a1c", fontweight="bold")
    right = [
        (9.5, "5. Text preprocessing", "lowercase, URL/e-mail/money/number\ntokens, regex tokens, stop words"),
        (8.0, "6. Feature engineering", "Bag of Words  |  TF-IDF\nn-grams (1,1)/(1,2), min_df"),
        (6.5, "7. Model training", "Multinomial NB, Logistic Regression,\nLinear SVM"),
        (5.0, "8. Hyperparameter tuning", "GridSearchCV, 5-fold stratified CV,\nscoring = F1 (spam)"),
    ]
    for y, t, s in right:
        _box(ax, xr, y, w, h, t, s, fill=PIPE_FILL, edge=PIPE_EDGE)
    for (y1, *_), (y2, *_) in zip(right, right[1:]):
        _arrow(ax, (xr, y1 - h / 2), (xr, y2 + h / 2))
    # split -> pipeline (training data)
    _arrow(ax, (xl + w / 2, 5.9), (5.35, 5.9))
    ax.text(4.83, 6.02, "training set", ha="center", va="bottom", fontsize=7.5, color=C.INK_2)

    # Bottom row
    bottom = [
        (2.6, 2.6, "9. Evaluation", "held-out test set: P, R, F1,\nROC-AUC, PR-AUC, MCC, FPR"),
        (7.4, 2.6, "10. Model comparison", "6 models; best = highest test F1,\nthen lower FPR, higher ROC-AUC"),
        (7.4, 1.0, "11. Prediction", "CLI: python -m src.predict \"message\"\n-> Spam (1) / Ham (0)"),
    ]
    for x, y, t, s in bottom:
        _box(ax, x, y, w, h, t, s)
    _arrow(ax, (xr, 5.0 - h / 2), (xr, 3.65))
    ax.annotate("", xy=(xl, 2.6 + h / 2), xytext=(xr - 0.4, 3.65),
                arrowprops=dict(arrowstyle="-|>", color="#52514e", lw=1.2, mutation_scale=12,
                                connectionstyle="arc3,rad=0.0"))
    ax.text(4.3, 3.45, "tuned pipelines", fontsize=7.5, color=C.INK_2, rotation=0)
    # test set from split straight to evaluation
    _arrow(ax, (xl - 1.2, 5.9 - h / 2), (xl - 1.2, 2.6 + h / 2))
    ax.text(xl - 1.12, 4.3, "test set\n(used once)", fontsize=7.5, color=C.INK_2, ha="left")
    _arrow(ax, (xl + w / 2, 2.6), (xr - w / 2, 2.6))
    _arrow(ax, (xr, 2.6 - h / 2), (xr, 1.0 + h / 2))
    return savefig(fig, "fig_5_1_workflow.png")


if __name__ == "__main__":
    draw_workflow()
