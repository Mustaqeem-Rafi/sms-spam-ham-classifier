# DECISIONS.md — rulings made while building the project

Format: **Ruling** — why — cost if wrong.

1. **Project root = this folder (`sms_spam_detection_filter/`) instead of a new `sms-spam-ml/` subfolder.** — HANDOFF.md lives here and the tree in §4 starts with HANDOFF.md at the root, so the folder *is* the project root. — Cosmetic only; rename the folder if the name matters.
2. **Python 3.12 in a local `.venv`.** — The system `python3` is 3.9 (spec needs 3.10+); 3.12 was available. — None.
3. **Report built with python-docx (as HANDOFF §11 says), not docx-js.** — The spec names `report/build_report.py` with python-docx. — None.
4. **Self-checks for `clean_text` live in `src/preprocess.py` (`python -m src.preprocess`), no separate `tests/` folder.** — §4 says "create exactly"; §12 asks only for unit checks that print 5 cleaned examples. — None.
5. **Duplicates are detected after HTML unescaping and whitespace stripping.** — Two messages that differ only by `&lt;` vs `<` or trailing spaces are the same message. — Count of duplicates could differ by a handful from a raw-text dedupe.
6. **Table 4.1 "Validation" row reports the per-fold size (825–826) rather than a separate split.** — Validation is 5-fold CV inside the training set (spec §5.4); listing it as 80 % would double-count. — None.
7. **LinearSVC uses `dual="auto"` and `max_iter=10000`.** — `dual="auto"` per spec note for sklearn ≥ 1.3; higher max_iter avoids convergence warnings at C = 10. — None.
8. **Workflow figure (Fig. 5.1) places the train/test split before feature extraction**, and groups text preprocessing + BoW/TF-IDF + classifier inside one "scikit-learn Pipeline (fitted on training data only)" box. — The spec's linear order (Feature Engineering → Split) would visually contradict the no-leakage design that the report claims. All stages named in the spec are kept. — None.
9. **Colours:** ham `#2a78d6` (blue), spam `#eb6834` (orange); classifiers NB `#1baf7a`, LR `#6250d6`, SVM `#eda100`; BoW drawn dashed / hatched, TF-IDF solid. — Palette checked with a colour-vision-deficiency validator (all pairs distinguishable); vectorizer has a secondary encoding so colour is never the only cue. — None.
10. **Two messages become empty after cleaning (pure emoticons ":)", ":-) :-)").** They are kept (a zero vector is classified by the prior/intercept). — Removing real data would be an undocumented filter. — Negligible.
11. **Literature:** all 12 sources were verified (see `report/literature_verification.md`). Two were corrected: ref. Gómez Hidalgo 2006 uses full Spanish double surnames; Sahami 1998 gets pages 55–62 and AAAI Tech. Rep. WS-98-05. Porter (1980) is added because the ablation uses the Porter stemmer. — Citation accuracy. — None.
