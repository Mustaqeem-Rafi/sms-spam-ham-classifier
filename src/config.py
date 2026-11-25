"""Central configuration: paths, seeds, split size, hyperparameter grids, colours."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
RAW_FILE = DATA_RAW / "SMSSpamCollection"
CLEAN_FILE = DATA_PROCESSED / "sms_clean.csv"

OUTPUTS = ROOT / "outputs"
FIG_DIR = OUTPUTS / "figures"
TABLE_DIR = OUTPUTS / "tables"
MODEL_DIR = OUTPUTS / "models"
RESULTS_DIR = OUTPUTS / "results"
REPORT_DIR = ROOT / "report"

DATASET_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
DATASET_PAGE = "https://archive.ics.uci.edu/dataset/228/sms+spam+collection"
DATE_ACCESSED = "25 November 2026"

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
SCORING = "f1"  # F1 of the positive (spam = 1) class

TOKEN_PATTERN = r"\b[a-z][a-z]+\b"

# Shared vectorizer grid (TF-IDF additionally tunes sublinear_tf)
VEC_GRID = {
    "vec__ngram_range": [(1, 1), (1, 2)],
    "vec__min_df": [1, 2],
}
TFIDF_EXTRA_GRID = {"vec__sublinear_tf": [False, True]}

CLF_GRIDS = {
    "MultinomialNB": {"clf__alpha": [0.01, 0.05, 0.1, 0.5, 1.0]},
    "LogisticRegression": {"clf__C": [0.1, 1, 10, 100],
                           "clf__class_weight": [None, "balanced"]},
    "LinearSVM": {"clf__C": [0.01, 0.1, 1, 10],
                  "clf__class_weight": [None, "balanced"]},
}

VECTORIZERS = ["BoW", "TF-IDF"]
CLASSIFIERS = ["MultinomialNB", "LogisticRegression", "LinearSVM"]
CLF_SHORT = {"MultinomialNB": "NB", "LogisticRegression": "LR", "LinearSVM": "SVM"}

# Consistent colours in every figure (checked with a colour-vision-deficiency
# validator: every pair below stays distinguishable for protan/deutan/tritan).
HAM_COLOR = "#2a78d6"    # blue
SPAM_COLOR = "#eb6834"   # orange
CLASS_PALETTE = {"ham": HAM_COLOR, "spam": SPAM_COLOR}
CLF_COLORS = {"MultinomialNB": "#1baf7a", "LogisticRegression": "#6250d6", "LinearSVM": "#eda100"}
VEC_LINESTYLE = {"BoW": "--", "TF-IDF": "-"}          # secondary encoding for vectorizer
VEC_HATCH = {"BoW": "//", "TF-IDF": ""}
VEC_COLORS = {"BoW": "#a8a6a0", "TF-IDF": "#3d3c39"}  # neutral pair for BoW-vs-TF-IDF bars
METRIC_COLORS = {"Precision": "#6250d6", "Recall": "#008300", "F1": "#e87ba4"}
INK = "#0b0b0b"
INK_2 = "#52514e"
FIG_DPI = 300


def model_name(vec: str, clf: str) -> str:
    """Canonical model id, e.g. 'TF-IDF_LinearSVM'."""
    return f"{vec}_{clf}"


def model_file(vec: str, clf: str) -> Path:
    return MODEL_DIR / f"{vec.replace('-', '')}_{clf}.joblib"


def ensure_dirs() -> None:
    for d in (DATA_RAW, DATA_PROCESSED, FIG_DIR, TABLE_DIR, MODEL_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
