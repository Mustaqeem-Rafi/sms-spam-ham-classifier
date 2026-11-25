"""Exploratory data analysis: statistics, Figs 3.1-3.8, Tables 3.1-3.5 (report Chapter 3)."""
import json
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud

from . import config as C
from .data import load_clean, load_raw
from .preprocess import tokens_no_stop

ORDER = ["ham", "spam"]


# ----------------------------------------------------------------- style helpers
def set_style():
    """One style for every figure in the report."""
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
        "axes.titleweight": "bold", "axes.edgecolor": "#c9c8c3",
        "grid.color": "#e6e5e1", "grid.linewidth": 0.6,
        "text.color": C.INK, "axes.labelcolor": C.INK,
        "xtick.color": C.INK_2, "ytick.color": C.INK_2,
        "figure.dpi": 100, "savefig.dpi": C.FIG_DPI, "savefig.bbox": "tight",
    })


def savefig(fig, name):
    path = C.FIG_DIR / name
    fig.savefig(path, dpi=C.FIG_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {path.relative_to(C.ROOT)}")
    return path


def _bar_labels(ax, bars, labels, pad=3):
    for b, lab in zip(bars, labels):
        ax.annotate(lab, (b.get_x() + b.get_width() / 2, b.get_height()),
                    xytext=(0, pad), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8.5, color=C.INK)


# ----------------------------------------------------------------- figures
def fig_class_distribution(raw_counts, clean_counts):
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    x = np.arange(2)
    w = 0.36
    for i, (title, counts) in enumerate([("Raw (before deduplication)", raw_counts),
                                         ("After deduplication", clean_counts)]):
        total = sum(counts.values())
        vals = [counts[c] for c in ORDER]
        bars = ax.bar(x[i] + np.array([-w / 2, w / 2]), vals, width=w - 0.02,
                      color=[C.HAM_COLOR, C.SPAM_COLOR], edgecolor="white", linewidth=1.5)
        _bar_labels(ax, bars, [f"{v:,}\n({100 * v / total:.2f}%)" for v in vals])
    ax.set_xticks(x, ["Raw (before deduplication)", "After deduplication"])
    ax.set_ylabel("Number of messages")
    ax.set_ylim(0, max(raw_counts.values()) * 1.22)
    handles = [plt.Rectangle((0, 0), 1, 1, color=C.CLASS_PALETTE[c]) for c in ORDER]
    ax.legend(handles, ["Ham (0)", "Spam (1)"], loc="upper right", frameon=False)
    ax.set_title("Class distribution")
    ax.grid(axis="x", visible=False)
    return savefig(fig, "fig_3_1_class_distribution.png")


def fig_length_chars(df):
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for c in ORDER:
        s = df.loc[df.label == c, "n_chars"]
        ax.hist(s, bins=np.arange(0, 451, 10), density=True, alpha=0.55,
                color=C.CLASS_PALETTE[c], label=f"{c.capitalize()} (median {s.median():.0f})",
                edgecolor="white", linewidth=0.4)
    ax.axvline(160, color=C.INK_2, lw=1, ls=":")
    ax.text(163, ax.get_ylim()[1] * 0.92, "160-character\nSMS limit", fontsize=8, color=C.INK_2, va="top")
    ax.set_xlim(0, 450)
    ax.set_xlabel("Message length (characters); x-axis clipped at 450")
    ax.set_ylabel("Density (per class)")
    ax.set_title("Message length in characters by class")
    ax.legend(frameon=False)
    return savefig(fig, "fig_3_2_length_chars.png")


def _box(ax, df, col, ylabel):
    sns.boxplot(data=df, x="label", y=col, order=ORDER, hue="label", hue_order=ORDER,
                palette=C.CLASS_PALETTE, width=0.5, fliersize=2, linewidth=1, ax=ax, legend=False)
    ax.set_xticks([0, 1], ["Ham", "Spam"])
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)


def fig_length_words(df):
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    _box(ax, df, "n_words", "Words per message")
    for i, c in enumerate(ORDER):
        med = df.loc[df.label == c, "n_words"].median()
        ax.annotate(f"median {med:.0f}", (i + 0.27, med), xytext=(4, 0), textcoords="offset points",
                    va="center", ha="left", fontsize=8.5, color=C.INK)
    ax.set_xlim(-0.5, 1.95)
    ax.set_title("Word count by class")
    return savefig(fig, "fig_3_3_length_words.png")


def fig_digits(df):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.4), gridspec_kw={"width_ratios": [1, 1.25]})
    _box(ax1, df, "n_digits", "Digits per message")
    ax1.set_title("Digit count by class")
    feats = [("Any digit", df.n_digits > 0), ("URL", df.has_url == 1),
             ("Currency symbol", df.has_currency == 1)]
    x = np.arange(len(feats))
    w = 0.38
    for j, c in enumerate(ORDER):
        m = df.label == c
        vals = [100 * f[m].mean() for _, f in feats]
        bars = ax2.bar(x + (j - 0.5) * w, vals, width=w - 0.02, color=C.CLASS_PALETTE[c],
                       label=c.capitalize(), edgecolor="white", linewidth=1.5)
        _bar_labels(ax2, bars, [f"{v:.1f}%" for v in vals], pad=2)
    ax2.set_xticks(x, [f for f, _ in feats])
    ax2.set_ylim(0, 110)
    ax2.set_ylabel("% of messages in class")
    ax2.set_title("Messages containing digits, URLs, currency")
    ax2.legend(frameon=False, loc="upper right")
    ax2.grid(axis="x", visible=False)
    fig.tight_layout()
    return savefig(fig, "fig_3_4_digits_by_class.png")


def fig_top_words(counter, cls, fname, n=20):
    top = counter.most_common(n)
    words, counts = zip(*top)
    fig, ax = plt.subplots(figsize=(3.5, 4.7))
    y = np.arange(n)[::-1]
    bars = ax.barh(y, counts, color=C.CLASS_PALETTE[cls], height=0.72)
    ax.set_yticks(y, words)
    for b, v in zip(bars, counts):
        ax.text(b.get_width(), b.get_y() + b.get_height() / 2, f" {v}", va="center", fontsize=7.5, color=C.INK)
    ax.set_xlabel("Occurrences after preprocessing")
    ax.set_title(f"Top {n} tokens in {cls} messages")
    ax.set_xlim(0, max(counts) * 1.2)
    ax.grid(axis="y", visible=False)
    return savefig(fig, fname)


def fig_wordclouds(counters):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    for ax, c in zip(axes, ORDER):
        color = C.CLASS_PALETTE[c]
        wc = WordCloud(width=900, height=600, background_color="white", max_words=120,
                       random_state=C.RANDOM_STATE, prefer_horizontal=0.95,
                       color_func=lambda *a, _c=color, **k: _c)
        wc.generate_from_frequencies(dict(counters[c]))
        ax.imshow(wc, interpolation="bilinear")
        ax.set_title(f"{c.capitalize()} messages")
        ax.axis("off")
    fig.tight_layout()
    return savefig(fig, "fig_3_7_wordclouds.png")


def fig_correlation(df):
    cols = ["n_chars", "n_words", "n_digits", "n_upper", "has_url", "has_currency", "target"]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    cmap = sns.diverging_palette(250, 25, s=80, l=55, center="light", as_cmap=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap=cmap, vmin=-1, vmax=1, square=True,
                linewidths=1, linecolor="white", cbar_kws={"shrink": 0.75, "label": "Pearson r"},
                annot_kws={"fontsize": 8.5}, ax=ax)
    ax.set_title("Correlation of engineered numeric features")
    ax.tick_params(axis="x", rotation=40)
    return savefig(fig, "fig_3_8_feature_correlation.png"), corr


# ----------------------------------------------------------------- tables
def _short(s, n=80):
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[: n - 3].rstrip() + "..."


def run_eda():
    C.ensure_dirs()
    set_style()
    df = load_clean()
    raw = load_raw()
    with open(C.RESULTS_DIR / "data_log.json") as f:
        log = json.load(f)

    raw_counts = raw["label"].value_counts().to_dict()
    clean_counts = df["label"].value_counts().to_dict()

    # token counts per class (after preprocessing + stop-word removal)
    counters = {c: Counter() for c in ORDER}
    for lab, txt in zip(df.label, df.text):
        counters[lab].update(tokens_no_stop(txt))
    vocab_all = set(counters["ham"]) | set(counters["spam"])

    print("Figures:")
    fig_class_distribution(raw_counts, clean_counts)
    fig_length_chars(df)
    fig_length_words(df)
    fig_digits(df)
    fig_top_words(counters["ham"], "ham", "fig_3_5_top_words_ham.png")
    fig_top_words(counters["spam"], "spam", "fig_3_6_top_words_spam.png")
    fig_wordclouds(counters)
    _, corr = fig_correlation(df)

    # Table 3.1 overview
    overview = [
        ("Dataset name", "SMS Spam Collection"),
        ("Source", "UCI Machine Learning Repository"),
        ("URL", C.DATASET_PAGE),
        ("Records (raw)", f"{len(raw):,}"),
        ("Records (after removing duplicates)", f"{len(df):,}"),
        ("Class counts (raw)", f"ham {raw_counts['ham']:,}; spam {raw_counts['spam']:,}"),
        ("Class counts (after removing duplicates)", f"ham {clean_counts['ham']:,}; spam {clean_counts['spam']:,}"),
        ("Features", "1 text input (message) + 1 target"),
        ("Target variable", "label (ham / spam), encoded ham = 0, spam = 1"),
        ("Task", "Binary text classification"),
        ("File format", "Tab-separated text file (label<TAB>message), no header, UTF-8"),
        ("Missing values", str(log["missing_values"])),
        ("DOI", C.DATASET_INFO["doi"]),
        ("Licence", C.DATASET_INFO["licence"]),
        ("Date accessed", C.DATE_ACCESSED),
    ]
    info = dict(C.DATASET_INFO, date_accessed=C.DATE_ACCESSED, url=C.DATASET_PAGE)
    comp_ham = sum(c["ham"] for c in info["composition"])
    comp_spam = sum(c["spam"] for c in info["composition"])
    info["composition_matches_file"] = (comp_ham == raw_counts["ham"] and comp_spam == raw_counts["spam"])
    with open(C.RESULTS_DIR / "dataset_info.json", "w") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    pd.DataFrame(overview, columns=["Attribute", "Value"]).to_csv(
        C.TABLE_DIR / "table_3_1_overview.csv", index=False)

    attrs = [
        ("label", "Categorical (string)", "Class of the message: 'ham' (legitimate) or 'spam'", "Original - target"),
        ("text", "Text (string)", "Raw SMS message content", "Original - input"),
        ("target", "Integer (0/1)", "Encoded label: ham = 0, spam = 1", "Derived - target"),
        ("n_chars", "Integer", "Number of characters in the message", "Derived (EDA only)"),
        ("n_words", "Integer", "Number of whitespace-separated words", "Derived (EDA only)"),
        ("n_digits", "Integer", "Number of digit characters (0-9)", "Derived (EDA only)"),
        ("n_upper", "Integer", "Number of uppercase letters", "Derived (EDA only)"),
        ("has_url", "Binary (0/1)", "1 if the message contains a URL (http... or www.)", "Derived (EDA only)"),
        ("has_currency", "Binary (0/1)", "1 if the message contains a currency symbol (£, $, €)", "Derived (EDA only)"),
    ]
    pd.DataFrame(attrs, columns=["Attribute", "Data type", "Description", "Role"]).to_csv(
        C.TABLE_DIR / "table_3_2_attributes.csv", index=False)

    samples = pd.concat([
        df[df.label == c].sample(5, random_state=C.RANDOM_STATE) for c in ORDER])
    samples = samples.assign(text=samples.text.map(_short))[["label", "text", "n_chars", "n_words"]]
    samples.to_csv(C.TABLE_DIR / "table_3_3_sample_records.csv", index=False)

    rows = []
    for c in ORDER:
        sub = df[df.label == c]
        for col, name in [("n_chars", "Characters"), ("n_words", "Words"), ("n_digits", "Digits")]:
            s = sub[col]
            rows.append({"Class": c, "Feature": name, "Mean": round(s.mean(), 2),
                         "Median": round(s.median(), 2), "Std": round(s.std(), 2),
                         "Min": int(s.min()), "Max": int(s.max())})
    stats = pd.DataFrame(rows)
    stats.to_csv(C.TABLE_DIR / "table_3_4_summary_stats.csv", index=False)

    dup_rows = []
    for c in ORDER + ["total"]:
        before = len(raw) if c == "total" else raw_counts[c]
        after = len(df) if c == "total" else clean_counts[c]
        dup_rows.append({"Class": c, "Before": before, "Duplicates removed": before - after,
                         "After": after, "Percent removed": round(100 * (before - after) / before, 2)})
    pd.DataFrame(dup_rows).to_csv(C.TABLE_DIR / "table_3_5_duplicates.csv", index=False)

    # dtypes table (for §3.4)
    pd.DataFrame({"Column": df.columns, "dtype": [str(t) for t in df.dtypes]}).to_csv(
        C.TABLE_DIR / "table_3_x_dtypes.csv", index=False)

    def pct(mask):
        return round(100 * float(mask.mean()), 2)

    ham, spam = df[df.label == "ham"], df[df.label == "spam"]
    obs = {
        "n_raw": len(raw), "n_clean": len(df),
        "raw_class_counts": raw_counts, "clean_class_counts": clean_counts,
        "raw_spam_pct": round(100 * raw_counts["spam"] / len(raw), 2),
        "spam_pct": pct(df.target == 1), "ham_pct": pct(df.target == 0),
        "imbalance_ratio_ham_per_spam": round(clean_counts["ham"] / clean_counts["spam"], 2),
        "missing_values": log["missing_values"],
        "n_duplicates_removed": log["n_duplicates_removed"],
        "duplicates_by_class": log["duplicates_by_class"],
        "duplicates_pct_of_raw": round(100 * log["n_duplicates_removed"] / len(raw), 2),
        "n_html_entity_messages": log["n_html_entity_messages"],
        "n_texts_with_conflicting_labels": log["n_texts_with_conflicting_labels"],
        "mean_chars": {"ham": round(ham.n_chars.mean(), 2), "spam": round(spam.n_chars.mean(), 2)},
        "median_chars": {"ham": float(ham.n_chars.median()), "spam": float(spam.n_chars.median())},
        "mean_words": {"ham": round(ham.n_words.mean(), 2), "spam": round(spam.n_words.mean(), 2)},
        "median_words": {"ham": float(ham.n_words.median()), "spam": float(spam.n_words.median())},
        "mean_digits": {"ham": round(ham.n_digits.mean(), 2), "spam": round(spam.n_digits.mean(), 2)},
        "median_digits": {"ham": float(ham.n_digits.median()), "spam": float(spam.n_digits.median())},
        "mean_upper": {"ham": round(ham.n_upper.mean(), 2), "spam": round(spam.n_upper.mean(), 2)},
        "max_chars": int(df.n_chars.max()),
        "pct_with_digits": {"ham": pct(ham.n_digits > 0), "spam": pct(spam.n_digits > 0)},
        "pct_with_url": {"ham": pct(ham.has_url == 1), "spam": pct(spam.has_url == 1)},
        "pct_with_currency": {"ham": pct(ham.has_currency == 1), "spam": pct(spam.has_currency == 1)},
        "pct_over_160_chars": {"ham": pct(ham.n_chars > 160), "spam": pct(spam.n_chars > 160)},
        "pct_over_200_chars": {"ham": pct(ham.n_chars > 200), "spam": pct(spam.n_chars > 200)},
        "pct_5plus_digit_run": {"ham": pct(ham.text.str.contains(r"\d{5,}")),
                                "spam": pct(spam.text.str.contains(r"\d{5,}"))},
        "vocab_size_eda": len(vocab_all),
        "vocab_size_ham": len(counters["ham"]), "vocab_size_spam": len(counters["spam"]),
        "vocab_spam_only": len(set(counters["spam"]) - set(counters["ham"])),
        "total_tokens": {c: int(sum(counters[c].values())) for c in ORDER},
        "mean_tokens_per_msg": {c: round(sum(counters[c].values()) / clean_counts[c], 2) for c in ORDER},
        "top20_ham": counters["ham"].most_common(20),
        "top20_spam": counters["spam"].most_common(20),
        "corr_with_target": {k: round(float(v), 4) for k, v in corr["target"].drop("target").items()},
        "corr_chars_words": round(float(corr.loc["n_chars", "n_words"]), 4),
        "empty_after_cleaning": int((df.text_clean.str.len() == 0).sum()),
    }
    # Which frequent spam tokens does the scikit-learn stop list remove?
    from .preprocess import STOP_WORDS, clean_text, tokenize
    raw_spam = Counter(t for txt in spam.text for t in tokenize(clean_text(txt)))
    top_raw = [t for t, _ in raw_spam.most_common(60)]
    obs["n_stop_words"] = len(STOP_WORDS)
    obs["top60_spam_tokens_in_stoplist"] = [t for t in top_raw if t in set(STOP_WORDS)]
    obs["call_in_stoplist"] = "call" in set(STOP_WORDS)
    obs["call_count_spam"] = int(raw_spam.get("call", 0))
    obs["call_count_ham"] = int(sum(tokenize(clean_text(t)).count("call") for t in ham.text))
    with open(C.RESULTS_DIR / "eda_observations.json", "w") as f:
        json.dump(obs, f, indent=2)
    print("EDA done:", {k: obs[k] for k in ["n_raw", "n_clean", "spam_pct", "n_duplicates_removed", "vocab_size_eda"]})
    return obs


if __name__ == "__main__":
    run_eda()
