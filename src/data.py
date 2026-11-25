"""Download, load, clean, deduplicate and split the UCI SMS Spam Collection (report §3.1, §4.1, §4.4)."""
import html
import io
import json
import re
import urllib.request
import zipfile

import pandas as pd
from sklearn.model_selection import train_test_split

from . import config as C
from .preprocess import clean_text

EXPECTED_RAW_ROWS = 5574  # stated on the UCI page


def download(force=False):
    """Download and unzip the dataset into data/raw/. Returns the raw file path."""
    C.ensure_dirs()
    if C.RAW_FILE.exists() and not force:
        return C.RAW_FILE
    print(f"Downloading {C.DATASET_URL}")
    try:
        with urllib.request.urlopen(C.DATASET_URL, timeout=60) as r:
            payload = r.read()
    except Exception as exc:  # blocker §13.1
        raise SystemExit(
            f"Download failed ({exc}). Please download the zip from {C.DATASET_PAGE} "
            f"and place 'SMSSpamCollection' in {C.DATA_RAW}"
        )
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        z.extract("SMSSpamCollection", C.DATA_RAW)
    return C.RAW_FILE


def load_raw():
    """Load the tab-separated raw file. quoting=3 (QUOTE_NONE) because messages contain quotes."""
    df = pd.read_csv(C.RAW_FILE, sep="\t", header=None, names=["label", "text"],
                     quoting=3, encoding="utf-8")
    print(f"Loaded raw file: {len(df)} rows")
    assert len(df) == EXPECTED_RAW_ROWS, f"expected {EXPECTED_RAW_ROWS} rows, got {len(df)}"
    assert set(df["label"]) == {"ham", "spam"}
    return df


URL_ANY = re.compile(r"http\S+|www\.\S+", re.I)
CURRENCY_ANY = re.compile(r"[£$€]")


def add_features(df):
    """Derived numeric features used for EDA only (never as model inputs)."""
    t = df["text"]
    df["n_chars"] = t.str.len()
    df["n_words"] = t.str.split().str.len()
    df["n_digits"] = t.str.count(r"\d")
    df["n_upper"] = t.str.count(r"[A-Z]")
    df["has_url"] = t.str.contains(URL_ANY).astype(int)
    df["has_currency"] = t.str.contains(CURRENCY_ANY).astype(int)
    return df


def clean(df_raw):
    """Map labels, check missing, strip/unescape, drop exact duplicates. Returns (df, log dict)."""
    df = df_raw.copy()
    log = {"n_raw": int(len(df)),
           "raw_class_counts": df["label"].value_counts().to_dict()}
    log["missing_values"] = int(df.isna().sum().sum())
    df = df.dropna()
    df["target"] = df["label"].map({"ham": 0, "spam": 1}).astype(int)

    n_html = int(df["text"].str.contains(r"&(?:lt|gt|amp|quot|#\d+);").sum())
    df["text"] = df["text"].map(html.unescape).str.strip()
    log["n_html_entity_messages"] = n_html

    # Exact duplicate messages (after strip/unescape); keep first occurrence.
    dup_mask = df.duplicated(subset="text", keep="first")
    log["n_duplicates_removed"] = int(dup_mask.sum())
    log["duplicates_by_class"] = df.loc[dup_mask, "label"].value_counts().to_dict()
    conflicting = df.groupby("text")["label"].nunique()
    log["n_texts_with_conflicting_labels"] = int((conflicting > 1).sum())
    df = df.loc[~dup_mask].reset_index(drop=True)
    log["n_clean"] = int(len(df))
    log["clean_class_counts"] = df["label"].value_counts().to_dict()

    df = add_features(df)
    df["text_clean"] = df["text"].map(clean_text)
    cols = ["label", "target", "text", "text_clean", "n_chars", "n_words",
            "n_digits", "n_upper", "has_url", "has_currency"]
    return df[cols], log


def build_clean_dataset():
    """Full data step: download -> load -> clean -> save CSV + log."""
    download()
    df, log = clean(load_raw())
    C.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_csv(C.CLEAN_FILE, index=False)
    with open(C.RESULTS_DIR / "data_log.json", "w") as f:
        json.dump(log, f, indent=2)
    print(f"Duplicates removed: {log['n_duplicates_removed']} {log['duplicates_by_class']}")
    print(f"Clean rows: {log['n_clean']} {log['clean_class_counts']}")
    return df, log


def load_clean():
    if not C.CLEAN_FILE.exists():
        build_clean_dataset()
    return pd.read_csv(C.CLEAN_FILE, keep_default_na=False)


def get_split(df=None):
    """Stratified 80/20 split of the deduplicated data (random_state=42)."""
    if df is None:
        df = load_clean()
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["target"], test_size=C.TEST_SIZE,
        stratify=df["target"], random_state=C.RANDOM_STATE)
    return X_train, X_test, y_train, y_test


def save_split_table(df=None):
    """Table 4.1: split sizes and class counts per subset."""
    if df is None:
        df = load_clean()
    from sklearn.model_selection import StratifiedKFold
    X_train, X_test, y_train, y_test = get_split(df)
    n = len(df)
    rows = []
    for name, y, note in [
        ("Training", y_train, "Used to fit the final model (vectorizer + classifier)"),
        ("Testing", y_test, "Held-out; used once for final evaluation"),
    ]:
        rows.append({"Subset": name, "Records": str(len(y)),
                     "Percent": round(100 * len(y) / n, 2),
                     "Ham": str(int((y == 0).sum())), "Spam": str(int((y == 1).sum())),
                     "Spam_percent": round(100 * y.mean(), 2), "Note": note})
    # Validation = 5-fold stratified CV inside the training set (not a separate split)
    skf = StratifiedKFold(n_splits=C.CV_FOLDS, shuffle=True, random_state=C.RANDOM_STATE)
    folds = [y_train.iloc[va] for _, va in skf.split(X_train, y_train)]
    rng = lambda vals: f"{min(vals)}" if min(vals) == max(vals) else f"{min(vals)}-{max(vals)}"
    rows.insert(1, {
        "Subset": "Validation",
        "Records": rng([len(f) for f in folds]) + " per fold",
        "Percent": round(100 * len(folds[0]) / n, 2),
        "Ham": rng([int((f == 0).sum()) for f in folds]),
        "Spam": rng([int((f == 1).sum()) for f in folds]),
        "Spam_percent": round(100 * float(pd.concat(folds).mean()), 2),
        "Note": f"Validation: {C.CV_FOLDS}-fold CV within training set (each fold held out once)"})
    rows.append({"Subset": "Total", "Records": str(n), "Percent": 100.0,
                 "Ham": str(int((df.target == 0).sum())), "Spam": str(int((df.target == 1).sum())),
                 "Spam_percent": round(100 * df.target.mean(), 2), "Note": "After deduplication"})
    out = pd.DataFrame(rows)
    out.to_csv(C.TABLE_DIR / "table_4_1_split.csv", index=False)
    return out


if __name__ == "__main__":
    C.ensure_dirs()
    df, log = build_clean_dataset()
    print(save_split_table(df).to_string(index=False))
