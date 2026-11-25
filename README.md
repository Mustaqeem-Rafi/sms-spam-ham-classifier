# SMS Spam Message Detection Using Machine Learning

ML Case Study 8 — Department of Artificial Intelligence and Machine Learning, B M S Institute of Technology and Management, Bengaluru.

A comparison of **Bag of Words** and **TF-IDF** features with **Multinomial Naive Bayes**, **Logistic Regression** and a **linear SVM** (6 tuned pipelines) on the UCI SMS Spam Collection. Every figure, table and number in the report is regenerated from code.

## Quick start

```bash
git clone https://github.com/Mustaqeem-Rafi/sms-spam-ham-classifier.git
cd sms-spam-ham-classifier
python3 -m venv .venv                     # Python 3.10+
source .venv/bin/activate                 # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_all.py                         # data -> EDA -> train -> evaluate -> notebook -> report (.docx + .pdf)
```

Options: `python run_all.py --no-notebook --no-pdf` skips the notebook execution and the PDF export.

## Data

`src/data.py` downloads `sms+spam+collection.zip` from the UCI repository and extracts `SMSSpamCollection` into `data/raw/` (not committed). Without internet access, download the zip from <https://archive.ics.uci.edu/dataset/228/sms+spam+collection> and put `SMSSpamCollection` in `data/raw/`.

Dataset: T. A. Almeida and J. M. Gómez Hidalgo, *SMS Spam Collection*, UCI Machine Learning Repository, 2011, DOI 10.24432/C5CC84, licence CC BY 4.0.

## Pipeline

| Step | Module | Output |
|---|---|---|
| Load, clean, deduplicate, split | `src/data.py` | `data/processed/sms_clean.csv`, `outputs/tables/table_4_1_split.csv` |
| Text preprocessing | `src/preprocess.py` | `clean_text()` used inside every vectorizer (`python -m src.preprocess` runs self-checks) |
| EDA | `src/eda.py` | Figs 3.1–3.8, Tables 3.1–3.5, `outputs/results/eda_observations.json` |
| Training (6 × GridSearchCV, 5-fold CV, F1) | `src/train.py` | `outputs/models/*.joblib`, `outputs/results/cv_summary.csv`, `cv_results_*.csv` |
| Test evaluation | `src/evaluate.py` | `outputs/results/metrics.json`, Tables 6.1–6.5, Figs 6.1–6.6, 6.8 |
| Interpretation | `src/interpret.py` | Fig 6.7, Table 6.6 |
| Workflow figure | `src/flowchart.py` | Fig 5.1 |
| Ablation, learning curve | `src/ablation.py` | Table B.1–B.3, Figs B.1–B.2 |
| Prediction CLI | `src/predict.py` | `python -m src.predict "your message"` |
| Report | `report/build_report.py` | `report/SMS_Spam_Report.docx`, `.pdf` |

All randomness uses `random_state = 42`; results are identical on every run except measured timings.

## Leakage prevention

Exact duplicates are removed **before** the stratified 80/20 split, and the vectorizer is the first step of a scikit-learn `Pipeline`, so vocabulary and IDF weights are learned only from training folds (during CV) or the training set (final model).

## Report

`report/build_report.py` reads every value from `outputs/` (JSON/CSV) and writes the `.docx` with python-docx. `report/lo_export.py` opens it in LibreOffice (headless), updates the Table of Contents, List of Figures and List of Tables, and exports the PDF.

Without LibreOffice: open the `.docx` in MS Word → Ctrl+A → F9 (choose "Update entire table") → File → Save As → PDF.

The remaining yellow-highlighted `[FILL: …]` placeholder in the report (guide name and designation) must be completed by the team. **Fill it in the `GUIDE` constant at the top of `report/build_report.py`** (team details live there too) and re-run `python report/build_report.py`; hand edits to the `.docx` are overwritten whenever the report is rebuilt.

## Prediction

```bash
python -m src.predict "Congratulations! You won a £1000 voucher, call 09061234567 now"
```

## Files

- `notebooks/SMS_Spam_Detection.ipynb` — executed walkthrough of Chapters 3–6 (calls `src/`).
- `DECISIONS.md` — design decisions taken while building the project.
- `report/literature_verification.md` — how each reference was verified.
