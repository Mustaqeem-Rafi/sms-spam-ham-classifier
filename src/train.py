"""Train and tune the 6 pipelines (2 vectorizers x 3 classifiers) with GridSearchCV (report Chapter 5)."""
import json
import time
import warnings

import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from . import config as C
from .data import get_split
from .preprocess import STOP_WORDS, clean_text

# scikit-learn checks the stop list against the preprocessor and warns that
# tokens such as "don't" would be split; the list is applied to our tokens, so
# the warning is expected and harmless.
warnings.filterwarnings("ignore", message="Your stop_words may be inconsistent")


def make_vectorizer(name, **overrides):
    kw = dict(preprocessor=clean_text, token_pattern=C.TOKEN_PATTERN, stop_words=STOP_WORDS)
    kw.update(overrides)
    if name == "BoW":
        return CountVectorizer(**kw)
    if name == "TF-IDF":
        return TfidfVectorizer(**kw)
    raise ValueError(name)


def make_classifier(name):
    if name == "MultinomialNB":
        return MultinomialNB()
    if name == "LogisticRegression":
        return LogisticRegression(solver="liblinear", max_iter=2000, random_state=C.RANDOM_STATE)
    if name == "LinearSVM":
        return LinearSVC(dual="auto", random_state=C.RANDOM_STATE, max_iter=10000)
    raise ValueError(name)


def make_pipeline(vec, clf):
    return Pipeline([("vec", make_vectorizer(vec)), ("clf", make_classifier(clf))])


def param_grid(vec, clf):
    grid = dict(C.VEC_GRID)
    if vec == "TF-IDF":
        grid.update(C.TFIDF_EXTRA_GRID)
    grid.update(C.CLF_GRIDS[clf])
    return grid


def cv_splitter():
    return StratifiedKFold(n_splits=C.CV_FOLDS, shuffle=True, random_state=C.RANDOM_STATE)


def _jsonable(params):
    out = {}
    for k, v in params.items():
        out[k] = list(v) if isinstance(v, tuple) else v
    return out


def train_all():
    C.ensure_dirs()
    X_train, X_test, y_train, y_test = get_split()
    print(f"Training set: {len(X_train)} messages; test set: {len(X_test)} (test is not touched here)")
    rows = []
    for vec in C.VECTORIZERS:
        for clf in C.CLASSIFIERS:
            name = C.model_name(vec, clf)
            grid = param_grid(vec, clf)
            gs = GridSearchCV(make_pipeline(vec, clf), grid, scoring=C.SCORING, cv=cv_splitter(),
                              n_jobs=-1, refit=True, return_train_score=True)
            t0 = time.perf_counter()
            gs.fit(X_train, y_train)
            search_time = time.perf_counter() - t0
            i = gs.best_index_
            res = pd.DataFrame(gs.cv_results_)
            res.to_csv(C.RESULTS_DIR / f"cv_results_{name}.csv", index=False)
            joblib.dump(gs.best_estimator_, C.model_file(vec, clf))
            vocab = len(gs.best_estimator_.named_steps["vec"].vocabulary_)
            rows.append({
                "model": name, "vectorizer": vec, "classifier": clf,
                "best_params": json.dumps(_jsonable(gs.best_params_)),
                "cv_f1_mean": gs.cv_results_["mean_test_score"][i],
                "cv_f1_std": gs.cv_results_["std_test_score"][i],
                "cv_train_f1_mean": gs.cv_results_["mean_train_score"][i],
                "n_candidates": len(res), "n_fits": len(res) * C.CV_FOLDS,
                "search_time_s": search_time, "refit_time_s": gs.refit_time_,
                "vocab_size": vocab,
            })
            print(f"  {name:28s} CV F1 = {rows[-1]['cv_f1_mean']:.4f} ± {rows[-1]['cv_f1_std']:.4f} "
                  f"| {gs.best_params_} | {search_time:.1f}s")
    summary = pd.DataFrame(rows)
    summary.to_csv(C.RESULTS_DIR / "cv_summary.csv", index=False)
    return summary


if __name__ == "__main__":
    train_all()
