# HANDOFF — ML Case Study 8: SMS Spam Message Detection

> **Deadline: TODAY (submit before end of day).** Work fast. Do not stop to ask for approval, except for the blockers in §13.
> **Final deliverables:** (1) one PDF report in the department format, (2) the editable `.docx`, (3) source code + executed notebook, all reproducible with one command.

---

## 0. Starter prompt (paste this into Claude Code)

```
Read HANDOFF.md in this folder completely. Then build the whole project end to end:
code, executed notebook, figures, results, and the final report (.docx + .pdf) exactly
as HANDOFF.md specifies. Follow the phases in §12 in order. Use a task list. Do not
invent any number: every value in the report must come from files in outputs/.
Leave every unknown personal detail as a yellow-highlighted [FILL: ...] placeholder.
Only stop and ask me if you hit a blocker listed in §13.
```

---

## 1. Project facts (use these exactly)

| Item | Value |
|---|---|
| Case study no. | 8 |
| Title (cover page) | **SMS Spam Message Detection Using Machine Learning: A Comparative Study of Naive Bayes, Logistic Regression and Support Vector Machine with Bag-of-Words and TF-IDF Features** |
| Short title (headers / certificate) | SMS Spam Message Detection Using Machine Learning |
| College | BMS Institute of Technology and Management, Bengaluru — **[FILL: confirm exact name as on college letterhead]** |
| Department | Department of Artificial Intelligence and Machine Learning |
| Programme | V Semester Bachelor of Engineering in Artificial Intelligence and Machine Learning |
| Academic year | 2026–2027 — **[FILL: confirm]** |
| Guide | **[FILL: Guide name and designation]** |
| Team (keep this order) | 1. Javeria Taj – **[FILL: USN]** <br> 2. Shaik Mohammed Mustaqeem Rafi – **[FILL: USN]** <br> 3. Mohammed Mohsin – **[FILL: USN]** <br> 4. Varun Kalmane – **[FILL: USN]** |
| Repo link (Appendix A) | **[FILL: GitHub URL]** |

**Assigned problem statement (copy verbatim into §1.2 of the report):**
> Classify SMS messages as Spam or Ham. Apply text preprocessing such as lowercasing, tokenization and stop-word removal, represent messages using Bag of Words and TF-IDF, and compare classifiers.
> Dataset: UCI SMS Spam Collection — https://archive.ics.uci.edu/dataset/228/sms+spam+collection
> Suggested ML techniques: Naive Bayes, Logistic Regression, SVM, TF-IDF

---

## 2. What "done" means (acceptance criteria)

- [ ] `python run_all.py` runs from a clean clone and regenerates every figure, table, metric, model and the report `.docx`.
- [ ] Notebook `notebooks/SMS_Spam_Detection.ipynb` is executed, with all outputs saved in it.
- [ ] 2 feature representations (BoW, TF-IDF) × 3 classifiers (Multinomial Naive Bayes, Logistic Regression, Linear SVM) = **6 models**, all tuned and evaluated on the **same held-out test set**.
- [ ] No data leakage: deduplicate before split; all vectorizers fitted only on training data (inside a `Pipeline`).
- [ ] Report follows the format in §10 and §11: every chapter, every table, numbered figure/table captions, IEEE references, 25–35 main-body pages.
- [ ] Every number in the report is read programmatically from `outputs/` (no typing numbers by hand).
- [ ] `report/SMS_Spam_Report.pdf` exists and the TOC, List of Figures and List of Tables show real page numbers.

---

## 3. Environment

- Python 3.10+ (record the exact version in report §5.3).
- `requirements.txt`:
  ```
  numpy
  pandas
  scikit-learn>=1.3
  matplotlib
  seaborn
  wordcloud
  nltk
  joblib
  python-docx
  jupyter
  nbconvert
  ipykernel
  ```
- Pin the installed versions after install: `pip freeze | grep -iE "numpy|pandas|scikit|matplotlib|seaborn|wordcloud|nltk|docx" > outputs/results/versions.txt`. Put these versions in report §5.3.
- **Avoid NLTK data downloads** (they can fail offline). Use:
  - Tokenization: regex (see §5).
  - Stop words: `sklearn.feature_extraction.text.ENGLISH_STOP_WORDS`.
  - Stemming (ablation only): `nltk.stem.PorterStemmer` (needs no download).
- For PDF conversion you need LibreOffice (`soffice`). Check with `which soffice`. If missing, see §11.4 fallback.
- Record hardware (CPU, RAM, OS) with `platform` / `psutil` or `lscpu` for report §5.3.
- Global seed: `RANDOM_STATE = 42` everywhere.

---

## 4. Folder structure (create exactly)

```
sms-spam-ml/
├── HANDOFF.md
├── README.md                  # install, get data, run, reproduce
├── requirements.txt
├── run_all.py                 # runs every phase in order
├── data/
│   ├── raw/                   # SMSSpamCollection (tab-separated) — git-ignored
│   └── processed/             # sms_clean.csv
├── src/
│   ├── config.py              # paths, RANDOM_STATE, TEST_SIZE, grids
│   ├── data.py                # download, load, dedupe, split
│   ├── preprocess.py          # clean_text(), tokenizer, stop words
│   ├── eda.py                 # all EDA stats + figures
│   ├── train.py               # 6 GridSearchCV pipelines, save models
│   ├── evaluate.py            # test metrics, curves, confusion matrices, error analysis
│   ├── interpret.py           # top spam/ham tokens from NB and LR
│   ├── ablation.py            # preprocessing ablation (Appendix B)
│   ├── predict.py             # CLI: python -m src.predict "your message"
│   └── flowchart.py           # draws Fig 5.1 workflow with matplotlib
├── notebooks/
│   └── SMS_Spam_Detection.ipynb   # calls src/ functions, shows outputs, executed
├── outputs/
│   ├── figures/               # PNG, 300 dpi
│   ├── tables/                # CSV for every report table
│   ├── models/                # joblib pipelines
│   └── results/               # metrics.json, cv_results.csv, versions.txt, env.json
└── report/
    ├── build_report.py        # python-docx report generator
    ├── SMS_Spam_Report.docx
    └── SMS_Spam_Report.pdf
```

---

## 5. Data pipeline

### 5.1 Get the data
1. Try download: `https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip` → unzip → `data/raw/SMSSpamCollection`.
2. If download fails, stop and ask the user to download the zip from the UCI page and put it in `data/raw/` (blocker §13).
3. Format: one message per line, `label<TAB>text`. Labels: `ham` / `spam`.
4. Load with `pd.read_csv(path, sep="\t", header=None, names=["label","text"], quoting=3, encoding="utf-8")`. `quoting=3` (QUOTE_NONE) is required, because messages contain quote characters. Assert row count after load and log it.
5. Record today's date as "Date accessed" (25 November 2026). Licence: **CC BY 4.0** (check the UCI page text and use what it says).

### 5.2 Cleaning (report §4.1)
- Map target: `ham → 0`, `spam → 1`. **Spam is the positive class** everywhere.
- Check missing values (expect none). Report the count.
- **Remove exact duplicate messages** (keep first). Duplicates are common in this dataset (several hundred). Reason to report: a duplicate in both train and test leaks information and inflates scores. Report counts before/after, per class.
- Strip leading/trailing whitespace. Unescape HTML entities (`&lt;`, `&gt;`, `&amp;` appear in the raw text) with `html.unescape`.
- Do **not** remove outliers (long messages are real data). Say this in §4.1 with the reason.

### 5.3 Text preprocessing (`src/preprocess.py`) — report §4.2/§4.3
Implement `clean_text(s, normalize=True)`, applied as the vectorizer `preprocessor`:

1. Lowercase.
2. Normalize (feature engineering, justify with EDA — spam has more digits, URLs, currency):
   - URLs (`http\S+|www\.\S+`) → ` urltoken `
   - E-mail addresses → ` emailtoken `
   - Currency symbols `£ $ €` → ` moneytoken `
   - Digit runs with ≥ 5 digits (phone / short codes) → ` phonetoken `
   - Other digit runs → ` numtoken `
3. Remove all characters except `a–z`, spaces and the tokens above.
4. Tokenize with regex `r"\b[a-z][a-z]+\b"` (tokens of ≥ 2 letters) via `token_pattern` or a custom tokenizer.
5. Stop-word removal: `ENGLISH_STOP_WORDS` (pass `stop_words=list(ENGLISH_STOP_WORDS)` to the vectorizer).
6. Stemming: **off** in the main pipeline (keep tokens readable). Test it only in the ablation.

Save `data/processed/sms_clean.csv` with columns `label, target, text, text_clean, n_chars, n_words, n_digits, n_upper, has_url, has_currency` for EDA and the report.

### 5.4 Split (report §4.4)
- `train_test_split(test_size=0.20, stratify=y, random_state=42)` on the **deduplicated** data.
- Validation = **5-fold StratifiedKFold CV inside the training set** (no separate validation split). State this in Table 4.1 as "Validation: 5-fold CV within training set".
- Save split sizes and class counts per subset to `outputs/tables/table_4_1_split.csv`.
- Leakage rule: the vectorizer is a step of the `Pipeline`, so it is fitted only on the training folds in CV and only on the training set for the final model. Say this clearly in the report.

---

## 6. EDA (`src/eda.py`) — report Chapter 3

Save every figure to `outputs/figures/` at 300 dpi with the file names below. Use one consistent style (seaborn `whitegrid`, ham = blue, spam = red/orange, same colors in every plot). Save every stat table as CSV.

| Fig no. | File | Content |
|---|---|---|
| 3.1 | `fig_3_1_class_distribution.png` | Bar chart of ham vs spam counts with % labels (after dedupe; optionally before/after side by side) |
| 3.2 | `fig_3_2_length_chars.png` | Histogram (or KDE) of message length in characters, by class |
| 3.3 | `fig_3_3_length_words.png` | Box plot of word count by class |
| 3.4 | `fig_3_4_digits_by_class.png` | Box/bar plot of digit count per message by class |
| 3.5 | `fig_3_5_top_words_ham.png` | Top 20 tokens in ham (after preprocessing) |
| 3.6 | `fig_3_6_top_words_spam.png` | Top 20 tokens in spam (after preprocessing) |
| 3.7 | `fig_3_7_wordclouds.png` | Two word clouds side by side (ham, spam) |
| 3.8 | `fig_3_8_feature_correlation.png` | Correlation heatmap of the engineered numeric features (`n_chars, n_words, n_digits, n_upper, has_url, has_currency, target`) |

Tables (CSV):
- `table_3_1_overview.csv` — dataset overview (fills report Table 3.1, see §10).
- `table_3_2_attributes.csv` — attribute description (label, text, plus derived features marked "Derived (EDA only)").
- `table_3_3_sample_records.csv` — 5 ham + 5 spam sample rows (shorten long text to ~80 chars).
- `table_3_4_summary_stats.csv` — mean/median/std/min/max of `n_chars, n_words, n_digits` by class.
- `table_3_5_duplicates.csv` — rows before/after dedupe per class.

Also write `outputs/results/eda_observations.json` with computed facts the report text will quote (class %, mean length per class, % of spam with digits/URLs/currency, duplicate count, vocabulary size). The report writer reads this file.

---

## 7. Modelling (`src/train.py`) — report Chapter 5

### 7.1 The 6 pipelines
```python
vectorizers = {
    "BoW":    CountVectorizer(preprocessor=clean_text, token_pattern=r"\b[a-z][a-z]+\b",
                              stop_words=list(ENGLISH_STOP_WORDS)),
    "TF-IDF": TfidfVectorizer(preprocessor=clean_text, token_pattern=r"\b[a-z][a-z]+\b",
                              stop_words=list(ENGLISH_STOP_WORDS)),
}
classifiers = {
    "MultinomialNB":      MultinomialNB(),
    "LogisticRegression": LogisticRegression(solver="liblinear", max_iter=2000, random_state=42),
    "LinearSVM":          LinearSVC(random_state=42),   # dual="auto" if sklearn>=1.3
}
# Pipeline([("vec", vectorizer), ("clf", classifier)]) for each of 6 combinations
```
Note: the custom `preprocessor` overrides `lowercase`, so `clean_text` must lowercase itself (it does). Put `clean_text` in an importable module (not a lambda) so joblib can pickle the pipeline.

### 7.2 Hyperparameter grids (report Table 5.1)
Shared vectorizer grid:
- `vec__ngram_range`: `[(1,1), (1,2)]`
- `vec__min_df`: `[1, 2]`
- TF-IDF only: `vec__sublinear_tf`: `[False, True]`

Classifier grids:
- MultinomialNB: `clf__alpha`: `[0.01, 0.05, 0.1, 0.5, 1.0]`
- LogisticRegression: `clf__C`: `[0.1, 1, 10, 100]`, `clf__class_weight`: `[None, "balanced"]`
- LinearSVM: `clf__C`: `[0.01, 0.1, 1, 10]`, `clf__class_weight`: `[None, "balanced"]`

`GridSearchCV(pipe, grid, scoring="f1", cv=StratifiedKFold(5, shuffle=True, random_state=42), n_jobs=-1, refit=True)`. Scoring = F1 of the spam class (justify: data is imbalanced, accuracy alone misleads).

Save for each model:
- best params, best CV F1 mean and std → `outputs/results/cv_summary.csv`
- full `cv_results_` → `outputs/results/cv_results_<model>.csv`
- fitted pipeline → `outputs/models/<vec>_<clf>.joblib`
- fit time (wall clock) of the refit on the full training set.

Expect total runtime of a few minutes on a laptop.

---

## 8. Evaluation (`src/evaluate.py`) — report Chapter 6

### 8.1 Metrics on the test set, for all 6 models
Accuracy, Precision (spam), Recall (spam), F1 (spam), Macro-F1, ROC-AUC, PR-AUC (average precision), MCC, False Positive Rate (ham wrongly marked spam), number of FP and FN, test inference time per 1,000 messages.
- Scores for curves: `predict_proba[:,1]` for NB and LR; `decision_function` for LinearSVC (valid for ROC-AUC/PR-AUC; say so in the report).
- Save `outputs/results/metrics.json` and `outputs/tables/table_6_1_results.csv` (one row per model, rounded to 4 decimals).
- Also save the `classification_report` text for each model to `outputs/results/clf_report_<model>.txt`.

**Best model rule (write it in the report):** highest test F1 (spam); tie-break by lower FPR, then higher ROC-AUC. Reason: a false positive hides a real message from the user, so it costs more than a false negative.

### 8.2 Figures
| Fig no. | File | Content |
|---|---|---|
| 5.1 | `fig_5_1_workflow.png` | Flowchart (from `src/flowchart.py`): Dataset → EDA → Preprocessing → Feature Engineering (BoW / TF-IDF) → Train/Test Split → Model Training (NB, LR, SVM + GridSearchCV) → Evaluation → Model Comparison → Prediction |
| 6.1 | `fig_6_1_confusion_matrices.png` | 2×3 grid of confusion matrices (rows = BoW/TF-IDF, cols = NB/LR/SVM), counts annotated |
| 6.2 | `fig_6_2_roc_curves.png` | ROC curves of all 6 models on one plot, AUC in legend |
| 6.3 | `fig_6_3_pr_curves.png` | Precision–Recall curves of all 6, AP in legend (more useful than ROC under imbalance — say so) |
| 6.4 | `fig_6_4_metric_comparison.png` | Grouped bar chart: Precision, Recall, F1 for 6 models |
| 6.5 | `fig_6_5_bow_vs_tfidf.png` | F1 per classifier, BoW vs TF-IDF side by side |
| 6.6 | `fig_6_6_cv_vs_test.png` | CV F1 (mean ± std error bars) vs test F1 per model (shows over/under-fitting) |
| 6.7 | `fig_6_7_top_features.png` (`src/interpret.py`) | Top 20 spam-indicative and top 20 ham-indicative tokens: LR coefficients (best LR model) and/or NB log-probability ratio |
| 6.8 | `fig_6_8_best_confusion_matrix.png` | Larger confusion matrix of the best model, counts + row-normalized % |

### 8.3 Error analysis + sample predictions (report §6.4, §6.5)
- `table_6_3_misclassified.csv`: all FP and FN of the best model (text, actual, predicted, score). Report shows up to 5 of each and explains *why* they failed (e.g. short spam with no money/number words; ham with "free", "call", numbers).
- `table_6_4_sample_predictions.csv`: 5 representative **test-set** messages (mix of ham/spam, include one hard case) with actual, predicted (best model) and probability/score.
- `table_6_5_custom_predictions.csv`: 5 new messages the team writes (e.g. "Congratulations! You won a £1000 voucher, call 09061234567 now", "Hey, are we still meeting at 5?", an OTP-style message, a delivery notice, a casual message with "free"). Clearly label these as "not from the dataset".
- `src/predict.py`: CLI that loads the best model and prints label + score. Show one CLI run (text output) in §6.5.

### 8.4 Optional but fast (do if time allows, put in Appendix B)
- **Ablation** (`src/ablation.py`) on the best (vectorizer, classifier) combination with its best params, 5-fold CV F1 on train + test F1:
  (a) full preprocessing, (b) no stop-word removal, (c) no number/URL/currency normalization, (d) + Porter stemming, (e) raw text with only lowercasing. → `table_b_1_ablation.csv` + `fig_b_1_ablation.png`.
- **McNemar test** between the top 2 models on the test set → one-line result in §6.4 (state p-value; do not over-claim).
- Learning curve of the best model → `fig_b_2_learning_curve.png`.

---

## 9. Notebook

`notebooks/SMS_Spam_Detection.ipynb` must be readable by the examiner as a standalone walkthrough:
1. Title, team names, problem statement (markdown).
2. Imports from `src/` (no logic duplicated in the notebook).
3. Sections that mirror report chapters 3–6, each with a short markdown explanation and the displayed figure/table.
4. Final cell: demo prediction on custom messages.

Execute it: `jupyter nbconvert --to notebook --execute --inplace notebooks/SMS_Spam_Detection.ipynb`. Commit with outputs.

---

## 10. Report content — chapter by chapter

Writing rules:
- Formal, third person ("This study…", "The model…"). Clear, short sentences.
- Every figure and table: numbered by chapter, captioned, **referred to in text and interpreted** ("Fig. 3.2 shows that spam messages are longer…"). No figure without discussion.
- Every number comes from `outputs/` via `build_report.py`. Use f-strings with loaded values.
- Do not claim deployment, real-time use, or novelty. Use "demonstrates", "on this dataset".
- Target **25–35 pages** for Chapters 1–7 (figures help). Approximate word budget below.

### Preliminary pages (Roman numerals)
1. **Cover page** — exactly the layout of the template: college name, department, "MACHINE LEARNING CASE STUDY REPORT", title, "Submitted in partial fulfilment…V Semester Bachelor of Engineering in Artificial Intelligence and Machine Learning", "Submitted by" + 4 names – USNs, "Under the guidance of" + guide, academic year. Add college logo only if a file `report/assets/logo.png` exists. No page number on cover.
2. **Certificate** — template text with title and names/USNs filled. Signature lines for Guide and HOD, Date, Place (Place: Bengaluru).
3. **Declaration** — template text, numbered list of 4 names and USNs.
4. **Acknowledgement** — ~half page. Thank guide [FILL], HOD [FILL: name or "the Head of the Department"], Principal, BMSIT&M, department faculty, UCI repository / dataset authors, family and friends.
5. **Abstract** — 200–300 words: problem, objective, dataset (size after dedupe, imbalance), preprocessing, BoW vs TF-IDF, NB/LR/SVM with GridSearchCV, metrics, **best model and its test F1/precision/recall/ROC-AUC**, main conclusion. **Keywords:** SMS spam detection; text classification; TF-IDF; Bag of Words; Naive Bayes; Support Vector Machine.
6. **Table of Contents** (Word TOC field, Heading 1–3).
7. **List of Figures** (caption-based TOC field on "Figure").
8. **List of Tables** (caption-based TOC field on "Table").
9. **List of Abbreviations** (two-column table): ML, EDA, NLP, SMS, UCI, BoW, TF-IDF, TF, IDF, NB, MNB, LR, SVM, CV, ROC, AUC, PR, AP, TP, TN, FP, FN, FPR, MCC, CSV, IEEE.

### Chapter 1 – Introduction (~1,200 words)
- 1.1 Background: SMS still widely used (OTPs, banking, delivery); spam causes fraud/phishing ("smishing"), cost, annoyance; short informal text with slang/abbreviations makes filtering hard compared to e-mail. No country-specific statistics unless found and cited.
- 1.2 Problem Statement: verbatim from §1, then one line: "The system takes the raw text of one SMS message as input and classifies it as Spam (1) or Ham (0)."
- 1.3 Objectives (5): analyse the dataset with EDA; preprocess text (lowercasing, tokenization, stop-word removal, normalization); represent text with BoW and TF-IDF; train and tune NB, LR, SVM with cross-validation; compare the 6 models on a held-out test set with precision, recall, F1, ROC-AUC; identify limitations and improvements.
- 1.4 Scope: included — UCI English SMS dataset, binary classification, 6 classical models, offline evaluation, CLI demo. Excluded — deep learning/transformers, non-English messages, live deployment on phones, MMS/images, sender metadata.
- 1.5 Organization of the Report: one line per chapter.

### Chapter 2 – Literature Survey (~1,200 words + table)
Table 2.1 (Sl. No., Author(s) and Year, Title/Source, Method Used, Key Findings/Limitations). Use these sources. **Verify each with a quick web search before use; drop any you cannot verify. Do not invent numeric results from papers — quote a paper's numbers only if you read them in the paper.**
1. T. A. Almeida, J. M. Gómez Hidalgo, A. Yamakami, "Contributions to the study of SMS spam filtering: New collection and results," *Proc. 11th ACM Symp. Document Engineering (DocEng '11)*, 2011, pp. 259–262. — introduces this dataset; compares many classifiers; linear SVM was among the best.
2. J. M. Gómez Hidalgo, G. C. Bringas, E. P. Sánz, F. C. García, "Content based SMS spam filtering," *Proc. ACM Symp. Document Engineering (DocEng '06)*, 2006, pp. 107–114. — early SMS spam work, Bayesian filters and SVM on small SMS corpora.
3. S. J. Delany, M. Buckley, D. Greene, "SMS spam filtering: Methods and data," *Expert Systems with Applications*, vol. 39, no. 10, pp. 9899–9908, 2012. — survey + comparison; content-based features; dataset issues.
4. M. Sahami, S. Dumais, D. Heckerman, E. Horvitz, "A Bayesian approach to filtering junk e-mail," *AAAI-98 Workshop on Learning for Text Categorization*, 1998. — Naive Bayes for spam; basis for content filters.
5. A. McCallum, K. Nigam, "A comparison of event models for Naive Bayes text classification," *AAAI-98 Workshop on Learning for Text Categorization*, 1998, pp. 41–48. — multinomial NB beats Bernoulli on larger vocabularies.
6. T. Joachims, "Text categorization with support vector machines: Learning with many relevant features," *Proc. ECML*, LNCS 1398, 1998, pp. 137–142. — SVMs suit high-dimensional sparse text.
7. G. Salton, C. Buckley, "Term-weighting approaches in automatic text retrieval," *Information Processing & Management*, vol. 24, no. 5, pp. 513–523, 1988. — TF-IDF weighting.

Paragraph per source (2–4 sentences) + 2.1 Research Gap / Motivation: many studies report one representation or one metric; this case study gives a controlled comparison of BoW vs TF-IDF across three classifiers under the same preprocessing, deduplicated split and tuning protocol, and reports imbalance-aware metrics (precision, recall, F1, PR-AUC, FPR). No novelty claim.

### Chapter 3 – Dataset Description and EDA (~1,800 words + figs 3.1–3.8)
- 3.1 Dataset Source: name, providers (Almeida & Gómez Hidalgo), URL, DOI if shown on UCI page, date accessed 25 November 2026, licence, file format (tab-separated text, no header), original composition per the UCI page.
- 3.2 Table 3.1 Overview: name, URL, records (raw and after dedupe), features (1 text input + 1 target), target `label` (ham/spam), task "Binary text classification", file format.
- 3.3 Table 3.2 Attribute description.
- 3.4 EDA: shape + sample records (Table 3.3), data types, summary stats (Table 3.4), class distribution (Fig 3.1), length distributions (Figs 3.2–3.3), digits (Fig 3.4), top words (Figs 3.5–3.6), word clouds (Fig 3.7), correlation (Fig 3.8), duplicates (Table 3.5). Interpret each.
- 3.5 Observations: imbalance → use F1/PR-AUC, stratified split, test `class_weight`; spam longer and digit-heavy → normalization tokens; duplicates → dedupe; frequent stop words carry little meaning → removal; strong lexical cues ("free", "call", "txt", "claim", "prize") → linear models on word features should work well.

### Chapter 4 – Preprocessing and Feature Engineering (~1,500 words)
- 4.1 Data Cleaning: missing, duplicates, HTML entities, label mapping, no outlier removal (with reasons).
- 4.2 Encoding and Feature Scaling: label encoding of target; text → BoW counts and TF-IDF weights. Give formulas: tf(t,d); idf(t) = ln((1+n)/(1+df(t))) + 1 (scikit-learn smooth default); tf-idf = tf × idf, then L2 normalization. No extra scaling needed (TF-IDF is already L2-normalized; NB needs non-negative counts). Include a tiny worked example: 3 short messages → BoW matrix and TF-IDF matrix (generate it with code, Table 4.2).
- 4.3 Feature Engineering: lowercasing, normalization tokens, regex tokenization, stop-word removal, n-grams (1,1)/(1,2) and min_df tuned by CV; resulting vocabulary size (from fitted best vectorizers). Derived numeric features were used only for EDA, not in models (keeps the comparison purely about text representation) — say this.
- 4.4 Split: Table 4.1 (Training, Validation = 5-fold CV inside training, Testing; N and %), random_state 42, stratified, leakage prevention via Pipeline.

### Chapter 5 – Model Development (~2,000 words)
- 5.1 Workflow: Fig 5.1 + short walkthrough.
- 5.2.1 Multinomial Naive Bayes: Bayes rule, conditional independence, P(c|d) ∝ P(c) Π P(w|c)^count, Laplace/Lidstone smoothing α. Why: fast, strong baseline for text. Limitations: independence assumption.
- 5.2.2 Logistic Regression: sigmoid σ(wᵀx+b), log-loss with L2 penalty, C = inverse regularization. Why: probabilistic output, interpretable coefficients. Limitations: linear boundary.
- 5.2.3 Linear SVM: max-margin hyperplane, hinge loss with L2, C trade-off. Why: works well in high-dimensional sparse text (Joachims). Limitations: no native probabilities.
- 5.3 Implementation Details: Python version, library versions (from `versions.txt`), environment (VS Code/Jupyter), hardware (from `env.json`), module list from §4 with one line each. One short code snippet of the pipeline + GridSearchCV.
- 5.4 Table 5.1 Hyperparameter Settings: Model | Hyperparameter | Search range | Selected value | Method (Grid Search, 5-fold CV, F1). One row per model × parameter, selected values from `cv_summary.csv`.

### Chapter 6 – Results and Evaluation (~2,500 words + figs)
- 6.1 Metrics: define Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, MCC, FPR with formulas from TP/TN/FP/FN. Justify: imbalance; cost of FP.
- 6.2 Table 6.1 test results (6 rows) + Table 6.2 CV results (best CV F1 mean ± std, fit time). Bold the best value in each column.
- 6.3 Visual results: Figs 6.1–6.8, each interpreted.
- 6.4 Comparison and Discussion: BoW vs TF-IDF per classifier; NB vs LR vs SVM; precision–recall trade-off; effect of `class_weight`; CV vs test agreement; error analysis with misclassified examples (Table 6.3); McNemar result if done. Explain *why* the best model wins, using results, not assumptions.
- 6.5 Sample Predictions: Table 6.4 (test set), Table 6.5 (custom messages), CLI output snippet.

### Chapter 7 – Conclusion and Future Enhancements (~800 words)
- 7.1 Conclusion: what was done; best model with its key test metrics; whether each objective in §1.3 was met (one line each).
- 7.2 Limitations: one dataset, mostly UK/Singapore SMS from ~2011 (check the UCI description for the exact sources); English only; small spam count; spam tactics change over time (concept drift); no sender/metadata; bag-of-words ignores word order and context; dedupe changes comparability with papers that did not dedupe.
- 7.3 Future: Indian/multilingual/code-mixed (Hinglish/Kannada-English) SMS data; character n-grams for obfuscated words; transformer models (e.g. DistilBERT) and compare cost; threshold tuning for a target FPR; probability calibration; explainability (LIME/SHAP); periodic retraining; a simple web/mobile demo.

### References (IEEE, numbered in order of first citation)
Must include: the dataset itself — T. A. Almeida and J. M. Gómez Hidalgo, "SMS Spam Collection," UCI Machine Learning Repository, 2011. [Online]. Available: https://archive.ics.uci.edu/dataset/228/sms+spam+collection. Accessed: 25 Nov. 2026. (add the DOI shown on the UCI page); the 7 papers in Ch. 2; F. Pedregosa et al., "Scikit-learn: Machine learning in Python," *JMLR*, vol. 12, pp. 2825–2830, 2011; C. Cortes and V. Vapnik, "Support-vector networks," *Machine Learning*, vol. 20, no. 3, pp. 273–297, 1995; pandas, matplotlib (J. D. Hunter, "Matplotlib: A 2D graphics environment," *Computing in Science & Engineering*, vol. 9, no. 3, pp. 90–95, 2007), seaborn, NLTK docs with access dates. Cite in text as [1], [2]…

### Appendix A – Source Code and Reproducibility
Repo link [FILL]; folder tree; install steps; where to put data; `python run_all.py`; expected runtime; seed; how to run `src/predict.py`. No keys or private data.

### Appendix B – Additional Results
Ablation table/figure, learning curve, full CV grid top-10 per model, classification reports.

---

## 11. Report build (`report/build_report.py`, python-docx)

### 11.1 Formatting (from the department guidelines — mandatory)
| Item | Setting |
|---|---|
| Paper | A4 |
| Body | Times New Roman 12 pt, line spacing 1.5, justified |
| Chapter heading (Heading 1) | 16 pt, bold, UPPERCASE, e.g. "CHAPTER 1 – INTRODUCTION"; start each chapter on a new page |
| Section heading (Heading 2) | 14 pt, bold, e.g. "1.1 Background" |
| Subsection heading (Heading 3) | 12 pt, bold, e.g. "5.2.1 Algorithm 1: Multinomial Naive Bayes" |
| Margins | Left 1.0 in; Top, Bottom, Right 0.8 in |
| Page numbers | Footer, centered. Cover: none. Preliminary pages: lower-roman (i, ii, …). Chapter 1 onward: Arabic starting at 1 |
| Figure captions | **Below** figure, centered, "Fig. 3.1: Class distribution…" |
| Table captions | **Above** table, centered, "Table 3.1: Dataset overview" |
| Tables | Grid style, header row bold + light grey shading, 10–11 pt text allowed inside tables, repeat header row |
| Figures | Width ~6.0 in (fit text width), centered |
| References | IEEE |

Set the Heading 1/2/3 and Normal styles' fonts (including `w:eastAsia` rFonts) to Times New Roman so the TOC picks them up correctly.

### 11.2 Implementation notes
- Section breaks: section 1 = cover (no footer number), section 2 = preliminary pages (`<w:pgNumType w:fmt="lowerRoman" w:start="1"/>`), section 3 = chapters (`<w:pgNumType w:fmt="decimal" w:start="1"/>`). Unlink footers between sections (`is_linked_to_previous = False`). Insert a `PAGE` field in each footer.
- TOC: insert field `TOC \o "1-3" \h \z \u`. List of Figures: `TOC \h \z \c "Figure"`. List of Tables: `TOC \h \z \c "Table"`.
- Captions: build with a `SEQ Figure` / `SEQ Table` field so Word lists them. Display text format "Fig. 3.1" / "Table 3.1": simplest is caption text "Figure 3.1: …" using `STYLEREF 1 \s`-free manual chapter prefix + `SEQ` field with `\s 1` reset — or write the number as plain text and keep a `SEQ` field hidden. Choose what works reliably; check the PDF lists.
- Placeholders: every `[FILL: …]` run gets yellow highlight (`WD_COLOR_INDEX.YELLOW`) so the team can find them.
- All numbers: load `metrics.json`, `cv_summary.csv`, `eda_observations.json`, table CSVs; format with 2–4 decimals or % consistently (pick one: e.g. percentages with 2 decimals in text, 4-decimal fractions in tables).
- Bold the best value per column in Table 6.1.

### 11.3 PDF conversion + updating the TOC
1. Preferred: LibreOffice headless with a small UNO/Basic macro that updates all indexes (`doc.getDocumentIndexes()` → `.update()` for each) and then exports PDF. Then check the PDF: TOC, List of Figures and List of Tables show real page numbers.
2. If that fails: produce the `.docx` and tell the user to open it in MS Word → Ctrl+A → F9 (update fields, "entire table") → File → Save As → PDF. This takes 1 minute. Put this step in the final message.

### 11.4 If LibreOffice is missing
Deliver `.docx` and give the user the Word steps in 11.3.2. Still render page images in some other way for your own visual QA if possible (e.g. `docx2pdf` on Windows/macOS with Word installed).

### 11.5 Visual QA (mandatory)
Convert the PDF to page images (`pdftoppm -r 60 -png`) and look at every page: no overflowing tables, no blank pages, figures readable, captions in the right place, headings on new pages, page numbering correct (cover none, roman, then 1…). Count main-body pages (Ch. 1–7) and confirm 25–35. If short, deepen discussion in Ch. 3, 4 and 6 (more interpretation, not filler). If long, shrink figures to 5.5 in.

---

## 12. Execution order (time-boxed)

| Phase | Time | Work | Checkpoint |
|---|---|---|---|
| 0 | 10 min | Folder tree, venv, requirements, config, README skeleton | `pip install` ok |
| 1 | 20 min | `data.py` download/load/dedupe/split; `preprocess.py` + unit checks (print 5 cleaned examples) | Counts logged; split CSV saved |
| 2 | 25 min | `eda.py` all Figs 3.x + tables + `eda_observations.json` | 8 figures exist |
| 3 | 30 min | `train.py` 6 GridSearchCV runs, save models + CV results | `cv_summary.csv` has 6 rows |
| 4 | 30 min | `evaluate.py`, `interpret.py`, `flowchart.py`, `predict.py`; optional `ablation.py` | `metrics.json`, Figs 5.1, 6.x exist |
| 5 | 15 min | Notebook build + execute | Notebook runs end to end |
| 6 | 75 min | `build_report.py` all chapters + formatting | `.docx` builds |
| 7 | 25 min | PDF + TOC update + visual QA + fixes | PDF passes §11.5 |
| 8 | 10 min | `run_all.py` clean re-run, README final, zip `submission/` | §2 checklist all ticked |

Send the user the PDF as soon as the first full draft exists (end of Phase 7), then finalize.

Final package `submission/` (and a zip of it):
- `SMS_Spam_Report.pdf`, `SMS_Spam_Report.docx`
- `SMS_Spam_Detection.ipynb` (executed)
- `source_code.zip` (src/, run_all.py, requirements.txt, README.md — no data/raw, no venv)

---

## 13. Stop and ask the user only if

1. The dataset cannot be downloaded (ask them to drop the UCI zip into `data/raw/`).
2. A package cannot be installed and there is no offline alternative.
3. The guide/department gave a different page limit or format (they did not, so far).

Everything else: decide, note the decision in `DECISIONS.md`, and continue.

---

## 14. Before submission — the team must do these by hand

- [ ] Fill 4 USNs, guide name + designation, HOD name (if used), repo URL, confirm college name and academic year (search the `.docx` for yellow `[FILL:` highlights).
- [ ] Update fields in Word (Ctrl+A → F9) if the PDF lists show no page numbers; re-export PDF.
- [ ] Read the abstract and conclusion once — the numbers must match Table 6.1.
- [ ] Push code to GitHub; paste the link in Appendix A.
- [ ] Get Guide and HOD signatures on the Certificate page if the department needs a signed copy.
- [ ] Run through the §FINAL SUBMISSION CHECKLIST on the last page of the department format.

---

## 15. Quick facts for the viva (Claude Code: fill the blanks from results at the end and print this section in the final message)

- Records raw / after dedupe: ___ / ___; spam share: ___%.
- Best model: ___ + ___; test F1 = ___, precision = ___, recall = ___, ROC-AUC = ___, FPR = ___.
- BoW vs TF-IDF: which was better for each classifier, and by how much.
- Why F1/PR-AUC, not accuracy: ~87% of messages are ham, so "always ham" already gets ~87% accuracy and catches 0 spam.
- Why dedupe before split: identical messages in train and test inflate scores.
- Why Pipeline: the vectorizer vocabulary and IDF are learned from training data only.
- Top 5 spam words from the model: ___.
