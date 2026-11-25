"""Build the case-study report (.docx) from the files in outputs/ and export it to PDF.

Every number in the report is read from outputs/ (JSON/CSV) and inserted with f-strings.
Qualitative claims that depend on results are guarded by `claim(...)`, so the build
fails loudly instead of writing a wrong sentence if the results ever change.

    python report/build_report.py          # .docx + .pdf
    python report/build_report.py --no-pdf
"""
import json
import sys
from pathlib import Path

import pandas as pd
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from docx_helpers import ReportDoc  # noqa: E402

OUT = ROOT / "outputs"
FIG, TAB, RES = OUT / "figures", OUT / "tables", OUT / "results"
DOCX = HERE / "SMS_Spam_Report.docx"
PDF = HERE / "SMS_Spam_Report.pdf"

TITLE = ("SMS Spam Message Detection Using Machine Learning: A Comparative Study of Naive Bayes, "
         "Logistic Regression and Support Vector Machine with Bag-of-Words and TF-IDF Features")
SHORT_TITLE = "SMS Spam Message Detection Using Machine Learning"
COLLEGE = "B M S Institute of Technology and Management"
DEPT = "Department of Artificial Intelligence and Machine Learning"
PROGRAMME = "V Semester Bachelor of Engineering in Artificial Intelligence and Machine Learning"
YEAR = "2026–2027"
GUIDE = "[FILL: Guide name and designation]"
HOD = "Dr. Anupama H S"
TEAM = [("Javeria Taj", "1BY24AI057"), ("Shaik Mohammed Mustaqeem Rafi", "1BY24AI141"),
        ("Mohammed Mohsin", "1BY25AI418"), ("Varun Kalmane", "1BY24AI184")]
REPO = "https://github.com/Mustaqeem-Rafi/sms-spam-ham-classifier"
PROBLEM = ("Classify SMS messages as Spam or Ham. Apply text preprocessing such as lowercasing, "
           "tokenization and stop-word removal, represent messages using Bag of Words and TF-IDF, "
           "and compare classifiers.")

CLF_FULL = {"MultinomialNB": "Multinomial Naive Bayes", "LogisticRegression": "Logistic Regression",
            "LinearSVM": "Linear SVM"}
CLF_SHORT = {"MultinomialNB": "NB", "LogisticRegression": "LR", "LinearSVM": "SVM"}
VECS = ["BoW", "TF-IDF"]
CLFS = ["MultinomialNB", "LogisticRegression", "LinearSVM"]


# ============================================================== data
def load():
    D = {}
    D["eda"] = json.loads((RES / "eda_observations.json").read_text())
    D["info"] = json.loads((RES / "dataset_info.json").read_text())
    D["log"] = json.loads((RES / "data_log.json").read_text())
    D["m"] = json.loads((RES / "metrics.json").read_text())
    D["interp"] = json.loads((RES / "interpretation.json").read_text())
    D["env"] = json.loads((RES / "env.json").read_text())
    D["cv"] = pd.read_csv(RES / "cv_summary.csv").set_index("model")
    rt = RES / "runtime.json"
    D["runtime"] = json.loads(rt.read_text()) if rt.exists() else {}
    cli = RES / "predict_cli_output.txt"
    D["cli"] = cli.read_text() if cli.exists() else ""
    T = lambda n: pd.read_csv(TAB / n, keep_default_na=False)
    for key, name in [("t31", "table_3_1_overview.csv"), ("t32", "table_3_2_attributes.csv"),
                      ("t33", "table_3_3_sample_records.csv"), ("t34", "table_3_4_summary_stats.csv"),
                      ("t35", "table_3_5_duplicates.csv"), ("t41", "table_4_1_split.csv"),
                      ("t42m", "table_4_2_messages.csv"), ("t42b", "table_4_2_bow.csv"),
                      ("t42t", "table_4_2_tfidf.csv"), ("t61", "table_6_1_results.csv"),
                      ("t62", "table_6_2_cv_results.csv"), ("t63", "table_6_3_misclassified.csv"),
                      ("t64", "table_6_4_sample_predictions.csv"), ("t65", "table_6_5_custom_predictions.csv"),
                      ("t66", "table_6_6_top_features.csv"), ("tb1", "table_b_1_ablation.csv"),
                      ("tb2", "table_b_2_cv_top10.csv"), ("tb3", "table_b_3_learning_curve.csv"),
                      ("tdt", "table_3_x_dtypes.csv")]:
        D[key] = T(name)
    D["clf_reports"] = {f"{v}_{c}": (RES / f"clf_report_{v}_{c}.txt").read_text() for v in VECS for c in CLFS}
    D["versions"] = (RES / "versions.txt").read_text().split()
    D["cvres"] = {f"{v}_{c}": pd.read_csv(RES / f"cv_results_{v}_{c}.csv") for v in VECS for c in CLFS}
    return D


def _fmt_param(v):
    if v is None or (isinstance(v, float) and v != v):
        return "None"
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v).replace(" ", "")


def grid_values(D, name, param):
    """Search range of one hyperparameter, read from the saved GridSearchCV results."""
    col = D["cvres"][name][f"param_{param}"]
    vals = list(dict.fromkeys(col.tolist()))
    def key(v):
        try:
            return (0, float(v))
        except (TypeError, ValueError):
            return (1, str(v))
    nums = [v for v in vals if isinstance(v, (int, float)) and not (isinstance(v, float) and v != v)
            and not isinstance(v, bool)]
    if len(nums) == len(vals):
        vals = sorted(vals, key=key)
    elif any(isinstance(v, float) and v != v for v in vals):
        vals = [None] + [v for v in vals if not (isinstance(v, float) and v != v)]
    return ", ".join(_fmt_param(v) for v in vals)


def f4(x):
    return f"{float(x):.4f}"


def p2(x):
    """x is already a percentage."""
    return f"{float(x):.2f}%"


def pc(frac):
    """fraction -> percentage text."""
    return f"{100 * float(frac):.2f}%"


def n(x):
    return f"{int(x):,}"


def pretty(name):
    v, c = name.split("_", 1)
    return f"{v} + {CLF_SHORT[c]}"


def full(name):
    v, c = name.split("_", 1)
    return f"{v} + {CLF_FULL[c]}"


def claim(cond, what):
    if not cond:
        raise AssertionError(f"Result-dependent claim no longer holds: {what}")


# ============================================================== citations
REFS = {
    "almeida2011": 'T. A. Almeida, J. M. Gómez Hidalgo, and A. Yamakami, "Contributions to the study of SMS spam '
                   'filtering: New collection and results," in *Proc. 11th ACM Symp. Document Engineering '
                   '(DocEng ’11)*, 2011, pp. 259–262, doi: 10.1145/2034691.2034742.',
    "delany2012": 'S. J. Delany, M. Buckley, and D. Greene, "SMS spam filtering: Methods and data," *Expert Systems '
                  'with Applications*, vol. 39, no. 10, pp. 9899–9908, Aug. 2012, doi: 10.1016/j.eswa.2012.02.053.',
    "gomez2006": 'J. M. Gómez Hidalgo, G. Cajigas Bringas, E. Puertas Sánz, and F. Carrero García, "Content based '
                 'SMS spam filtering," in *Proc. 2006 ACM Symp. Document Engineering (DocEng ’06)*, 2006, '
                 'pp. 107–114, doi: 10.1145/1166160.1166191.',
    "uci": 'T. A. Almeida and J. M. Gómez Hidalgo, "SMS Spam Collection," UCI Machine Learning Repository, 2011. '
           '[Online]. Available: https://archive.ics.uci.edu/dataset/228/sms+spam+collection, '
           'doi: 10.24432/C5CC84. Accessed: 25 Nov. 2026.',
    "salton1988": 'G. Salton and C. Buckley, "Term-weighting approaches in automatic text retrieval," *Information '
                  'Processing & Management*, vol. 24, no. 5, pp. 513–523, 1988, doi: 10.1016/0306-4573(88)90021-0.',
    "sahami1998": 'M. Sahami, S. Dumais, D. Heckerman, and E. Horvitz, "A Bayesian approach to filtering junk '
                  'e-mail," in *Learning for Text Categorization: Papers from the 1998 AAAI Workshop*, AAAI Tech. '
                  'Rep. WS-98-05, 1998, pp. 55–62.',
    "mccallum1998": 'A. McCallum and K. Nigam, "A comparison of event models for Naive Bayes text classification," '
                    'in *Learning for Text Categorization: Papers from the 1998 AAAI Workshop*, AAAI Tech. Rep. '
                    'WS-98-05, 1998, pp. 41–48.',
    "joachims1998": 'T. Joachims, "Text categorization with support vector machines: Learning with many relevant '
                    'features," in *Machine Learning: ECML-98*, Lecture Notes in Computer Science, vol. 1398. '
                    'Berlin, Heidelberg: Springer, 1998, pp. 137–142, doi: 10.1007/BFb0026683.',
    "cortes1995": 'C. Cortes and V. Vapnik, "Support-vector networks," *Machine Learning*, vol. 20, no. 3, '
                  'pp. 273–297, 1995, doi: 10.1007/BF00994018.',
    "pedregosa2011": 'F. Pedregosa *et al.*, "Scikit-learn: Machine learning in Python," *Journal of Machine '
                     'Learning Research*, vol. 12, pp. 2825–2830, 2011.',
    "porter1980": 'M. F. Porter, "An algorithm for suffix stripping," *Program*, vol. 14, no. 3, pp. 130–137, 1980.',
    "bird2009": 'S. Bird, E. Klein, and E. Loper, *Natural Language Processing with Python*. O’Reilly Media, 2009.',
    "nltkdocs": 'NLTK Project, "nltk.stem.porter module," NLTK documentation. [Online]. Available: '
                'https://www.nltk.org/api/nltk.stem.porter.html. Accessed: 25 Nov. 2026.',
    "mckinney2010": 'W. McKinney, "Data structures for statistical computing in Python," in *Proc. 9th Python in '
                    'Science Conf. (SciPy 2010)*, 2010, pp. 56–61, doi: 10.25080/Majora-92bf1922-00a.',
    "hunter2007": 'J. D. Hunter, "Matplotlib: A 2D graphics environment," *Computing in Science & Engineering*, '
                  'vol. 9, no. 3, pp. 90–95, 2007, doi: 10.1109/MCSE.2007.55.',
    "waskom2021": 'M. L. Waskom, "seaborn: Statistical data visualization," *Journal of Open Source Software*, '
                  'vol. 6, no. 60, p. 3021, 2021, doi: 10.21105/joss.03021.',
}
_order = []


def cite(*keys):
    """IEEE citation in order of first use: cite('a', 'b') -> '[1], [2]'."""
    out = []
    for k in keys:
        if k not in REFS:
            raise KeyError(k)
        if k not in _order:
            _order.append(k)
        out.append(f"[{_order.index(k) + 1}]")
    return ", ".join(out)


# ============================================================== preliminary pages
def cover(R):
    c = WD_ALIGN_PARAGRAPH.CENTER
    logo = HERE / "assets" / "logo.png"
    R.para(COLLEGE.upper(), style="Cover", size=16, bold=True, space_after=0)
    R.para("Bengaluru", style="Cover", size=13, bold=True, space_after=10)
    if logo.exists():
        p = R.para(style="Cover", align=c)
        p.add_run().add_picture(str(logo), width=Inches(1.3))
    R.para(DEPT, style="Cover", size=14, bold=True, space_after=36)
    R.para("MACHINE LEARNING CASE STUDY REPORT", style="Cover", size=15, bold=True, space_after=10)
    R.para("on", style="Cover", size=12, italic=True, space_after=10)
    R.para(f"“{TITLE}”", style="Cover", size=15, bold=True, space_after=30)
    R.para("Submitted in partial fulfilment of the requirements for the", style="Cover", size=12, space_after=0)
    R.para(PROGRAMME, style="Cover", size=12, bold=True, space_after=30)
    R.para("Submitted by", style="Cover", size=13, bold=True, italic=True, space_after=8)
    for name, usn in TEAM:
        R.para(f"{name} – {usn}", style="Cover", size=12, space_after=3)
    R.para("", style="Cover", space_after=22)
    R.para("Under the guidance of", style="Cover", size=13, bold=True, italic=True, space_after=8)
    R.para(GUIDE, style="Cover", size=12, space_after=0)
    R.para(DEPT, style="Cover", size=12, space_after=36)
    R.para(f"Academic Year {YEAR}", style="Cover", size=13, bold=True, space_after=0)


def certificate(R):
    R.doc.add_paragraph("CERTIFICATE", style="Prelim Heading")
    R.para(f"{COLLEGE}, Bengaluru", align=WD_ALIGN_PARAGRAPH.CENTER, bold=True, space_after=0)
    R.para(DEPT, align=WD_ALIGN_PARAGRAPH.CENTER, bold=True, space_after=18)
    names = "; ".join(f"**{nm}** ({usn})" for nm, usn in TEAM)
    R.para(f"This is to certify that the Machine Learning case study entitled **“{SHORT_TITLE}”** is a bona fide "
           f"work carried out by {names}, students of the {PROGRAMME}, {COLLEGE}, Bengaluru, in partial "
           f"fulfilment of the requirements of the Machine Learning course during the academic year {YEAR}. "
           "The case study report has been approved as it satisfies the academic requirements prescribed for "
           "the said course.")
    R.para("", space_after=48)
    t = R.doc.add_table(rows=3, cols=2)
    rows = [("______________________________", "______________________________"),
            ("Signature of the Guide", "Signature of the Head of the Department"),
            (GUIDE, HOD)]
    for i, (a, b) in enumerate(rows):
        for j, txt in enumerate((a, b)):
            p = t.rows[i].cells[j].paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            R.runs(p, txt, bold=(i == 1))
    R.set_widths(t, [3.2, 3.2])
    R.para("", space_after=36)
    R.para("Date: ____________________", align=WD_ALIGN_PARAGRAPH.LEFT, space_after=0)
    R.para("Place: Bengaluru", align=WD_ALIGN_PARAGRAPH.LEFT)


def declaration(R):
    R.doc.add_paragraph("DECLARATION", style="Prelim Heading")
    R.para(f"We, the undersigned students of the {PROGRAMME}, {COLLEGE}, Bengaluru, hereby declare that the "
           f"Machine Learning case study entitled **“{SHORT_TITLE}”** has been carried out by us under the guidance "
           f"of {GUIDE}, {DEPT}, and is submitted in partial fulfilment of the requirements of the Machine Learning "
           f"course during the academic year {YEAR}. The work presented in this report has not been submitted, in "
           "part or in full, for any other degree or course.")
    R.para("", space_after=12)
    rows = [[str(i), nm, usn, ""] for i, (nm, usn) in enumerate(TEAM, 1)]
    t = R.table(["Sl. No.", "Name", "USN", "Signature"], rows, "", "", widths=[0.7, 2.6, 1.5, 1.6],
                size=11, align=["c", "l", "c", "c"])
    # this table is not a numbered table: drop its caption paragraph
    cap = t._tbl.getprevious()
    cap.getparent().remove(cap)
    for row in t.rows[1:]:
        row.height = Inches(0.45)
    R.para("", space_after=24)
    R.para("Date: ____________________", align=WD_ALIGN_PARAGRAPH.LEFT, space_after=0)
    R.para("Place: Bengaluru", align=WD_ALIGN_PARAGRAPH.LEFT)


def acknowledgement(R):
    R.doc.add_paragraph("ACKNOWLEDGEMENT", style="Prelim Heading")
    R.paras(
        f"We express our sincere gratitude to our guide, {GUIDE}, {DEPT}, for the guidance, encouragement and "
        "constructive feedback given throughout this case study.",
        f"We thank {HOD}, Head of the {DEPT}, for providing the facilities and support needed to carry out this "
        f"work. We are grateful to the Principal and the management of {COLLEGE}, Bengaluru, for the academic "
        "environment and resources made available to us.",
        f"We also thank all the faculty members and staff of the {DEPT} for their help and suggestions. We "
        "acknowledge the creators of the SMS Spam Collection, T. A. Almeida and J. M. Gómez Hidalgo, and the UCI "
        "Machine Learning Repository for making the dataset publicly available, and the developers of the "
        "open-source Python libraries used in this study.",
        "Finally, we thank our families and friends for their constant support and encouragement.")
    R.para("", space_after=24)
    for nm, usn in TEAM:
        R.para(f"{nm} ({usn})", align=WD_ALIGN_PARAGRAPH.RIGHT, space_after=0)


def abstract(R, D):
    e, m = D["eda"], D["m"]
    best = m["best_model"]
    b = m["models"][best]
    lin = [k for k in m["models"] if not k.endswith("MultinomialNB")]
    spread = max(m["models"][k]["f1"] for k in lin) - min(m["models"][k]["f1"] for k in lin)
    mc = m["mcnemar_top2"]
    claim(e["mean_chars"]["spam"] > e["mean_chars"]["ham"], "spam longer than ham")
    claim(not best.endswith("MultinomialNB"), "best model is a linear discriminative model")
    R.doc.add_paragraph("ABSTRACT", style="Prelim Heading")
    R.paras(
        "Short Message Service (SMS) is still widely used for personal messages, one-time passwords, banking "
        "alerts and delivery notices, which also makes it an attractive channel for unsolicited and fraudulent "
        "messages. This study classifies SMS messages as spam or ham (legitimate). The UCI SMS Spam Collection "
        f"was used. After removing {n(e['n_duplicates_removed'])} exact duplicate messages, {n(e['n_clean'])} "
        f"messages remained, of which {p2(e['spam_pct'])} are spam, so the data are clearly imbalanced. "
        "Exploratory analysis showed that spam messages are longer and contain far more digits, URLs and currency "
        "symbols than ham. Each message was lowercased; URLs, e-mail addresses, currency symbols and numbers were "
        "replaced by normalisation tokens; the text was tokenised with a regular expression; and English stop "
        "words were removed. Two text representations, Bag of Words (BoW) and Term Frequency–Inverse Document "
        "Frequency (TF-IDF), were combined with three classifiers, Multinomial Naive Bayes, Logistic Regression "
        "and a linear Support Vector Machine, giving six pipelines. Each pipeline was tuned by grid search with "
        "5-fold stratified cross-validation on the training set, using the F1 score of the spam class, and was "
        f"then evaluated once on the same stratified held-out test set of {n(m['n_test'])} messages.",
        f"The best model under the predefined selection rule was **{full(best)}**, with test F1 = {f4(b['f1'])}, "
        f"precision = {f4(b['precision'])}, recall = {f4(b['recall'])}, ROC-AUC = {f4(b['roc_auc'])} and a false "
        f"positive rate of {pc(b['fpr'])}. The four Logistic Regression and linear SVM pipelines were within "
        f"{f4(spread)} F1 of one another, and an exact McNemar test found no significant difference between the "
        f"two best models (p = {f4(mc['p_value_exact'])}). The study demonstrates that linear models on word "
        "features give an accurate and interpretable spam filter on this dataset, and that the choice between BoW "
        "and TF-IDF matters little once both are tuned.")
    R.para("**Keywords:** SMS spam detection; text classification; TF-IDF; Bag of Words; Naive Bayes; "
           "Support Vector Machine.")


def front_lists(R):
    R.doc.add_paragraph("TABLE OF CONTENTS", style="Prelim Heading NoTOC")
    R.toc('TOC \\o "1-3" \\h \\z \\u')
    R.doc.add_paragraph("LIST OF FIGURES", style="Prelim Heading")
    R.toc('TOC \\h \\z \\t "Figure Caption,2"')
    R.doc.add_paragraph("LIST OF TABLES", style="Prelim Heading")
    R.toc('TOC \\h \\z \\t "Table Caption,2"')


ABBREVIATIONS = [
    ("AP", "Average Precision"), ("AUC", "Area Under the Curve"), ("BoW", "Bag of Words"),
    ("CSV", "Comma-Separated Values"), ("CV", "Cross-Validation"), ("EDA", "Exploratory Data Analysis"),
    ("FN", "False Negative"), ("FP", "False Positive"), ("FPR", "False Positive Rate"),
    ("IDF", "Inverse Document Frequency"), ("IEEE", "Institute of Electrical and Electronics Engineers"),
    ("LR", "Logistic Regression"), ("MCC", "Matthews Correlation Coefficient"), ("ML", "Machine Learning"),
    ("MNB", "Multinomial Naive Bayes"), ("NB", "Naive Bayes"), ("NLP", "Natural Language Processing"),
    ("PR", "Precision–Recall"), ("ROC", "Receiver Operating Characteristic"), ("SMS", "Short Message Service"),
    ("SVM", "Support Vector Machine"), ("TF", "Term Frequency"),
    ("TF-IDF", "Term Frequency–Inverse Document Frequency"), ("TN", "True Negative"), ("TP", "True Positive"),
    ("UCI", "University of California, Irvine (Machine Learning Repository)"),
]


def abbreviations(R):
    R.doc.add_paragraph("LIST OF ABBREVIATIONS", style="Prelim Heading")
    t = R.table(["Abbreviation", "Full form"], [[a, b] for a, b in ABBREVIATIONS], "", "",
                widths=[1.6, 4.4], size=11, align=["l", "l"])
    cap = t._tbl.getprevious()
    cap.getparent().remove(cap)


# ============================================================== Chapter 1
def chapter1(R, D):
    e = D["eda"]
    R.h1("Chapter 1 – Introduction")
    R.h2("1.1 Background")
    R.paras(
        "Short Message Service (SMS) remains one of the most widely used forms of mobile communication. Besides "
        "personal conversation, it carries one-time passwords (OTPs) for logins and payments, banking and "
        "credit-card alerts, delivery notifications, appointment reminders and service updates. Because almost "
        "every mobile phone can receive SMS without an internet connection or an app, a message reaches its "
        "recipient directly and is usually read within a short time.",
        "The same properties make SMS attractive for unsolicited bulk messages, commonly called *spam*. SMS spam "
        "ranges from aggressive advertising to premium-rate traps (“call this number to claim your prize”) and "
        "phishing messages that try to make the recipient open a link, share an OTP or call a fraudulent number. "
        "Phishing through SMS is often called *smishing*. Spam wastes the recipient’s time, can cause direct "
        "financial loss, and erodes trust in legitimate messages from banks and service providers. A reliable "
        "filter that separates spam from legitimate messages (called *ham*) therefore has practical value for "
        "users and network operators.",
        "Filtering SMS is harder than filtering e-mail in several ways. A single SMS is short, often within a "
        "160-character limit, so each message offers only a few words of evidence. The language is informal, "
        "with abbreviations, slang, phonetic spellings and deliberate misspellings such as “txt”, “ur”, “2” for "
        "“to” and “4” for “for”. SMS also lacks the rich header information (sender domains, routing headers, "
        f"attachments) that e-mail filters exploit. Delany *et al.* {cite('delany2012')} discuss how SMS spam "
        f"filtering differs from e-mail spam filtering, and Gómez Hidalgo *et al.* {cite('gomez2006')} studied "
        "whether the Bayesian techniques that work for e-mail transfer to SMS. A content-based filter must "
        "therefore learn from the words of the message itself.",
        "Machine learning offers a natural approach. Given a collection of messages labelled as spam or ham, a "
        "classifier can learn which words and patterns indicate spam. Before learning, the text must be turned "
        "into numbers. Two classical representations are the *Bag of Words* (BoW), which counts how often each "
        "word occurs, and *TF-IDF*, which weights each count by how rare the word is across the collection. Both "
        "are combined in this study with three classical classifiers—Multinomial Naive Bayes, Logistic "
        "Regression and a linear Support Vector Machine—on the public SMS Spam Collection created by Almeida "
        f"and Gómez Hidalgo {cite('almeida2011', 'uci')}.")
    R.h2("1.2 Problem Statement")
    R.para("The problem statement assigned for this case study is reproduced below.")
    p = R.para(f"*“{PROBLEM}”*")
    p.paragraph_format.left_indent = Inches(0.4)
    p.paragraph_format.right_indent = Inches(0.4)
    p = R.para("*Dataset:* UCI SMS Spam Collection — https://archive.ics.uci.edu/dataset/228/sms+spam+collection. "
               "*Suggested ML techniques:* Naive Bayes, Logistic Regression, SVM, TF-IDF.")
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.left_indent = Inches(0.4)
    p.paragraph_format.right_indent = Inches(0.4)
    R.para("The system takes the raw text of one SMS message as input and classifies it as Spam (1) or Ham (0).")
    R.para("Because spam forms a minority of the messages and because hiding a genuine message is more harmful "
           "than letting one spam message through, the problem is treated as an imbalanced binary classification "
           "task in which spam is the positive class and the false positive rate is monitored explicitly.")
    R.h2("1.3 Objectives")
    R.para("The objectives of this case study are:")
    R.bullets([
        "To analyse the SMS Spam Collection with exploratory data analysis (EDA): class balance, message length, "
        "use of digits, URLs and currency symbols, frequent words and duplicate messages.",
        "To preprocess the message text (lowercasing, normalisation of numbers, URLs and currency, tokenization "
        "and stop-word removal) and to represent it with Bag of Words and TF-IDF features.",
        "To train Multinomial Naive Bayes, Logistic Regression and linear Support Vector Machine classifiers on "
        "both representations and tune their hyperparameters with stratified cross-validation.",
        "To compare the six resulting models on the same held-out test set using precision, recall, F1 score, "
        "ROC-AUC and related imbalance-aware metrics, and to identify the best model with a predefined rule.",
        "To analyse the errors of the best model and identify the limitations of the approach and possible "
        "improvements."], numbered=True)
    R.h2("1.4 Scope of the Study")
    R.para("**Included.** The English SMS messages of the UCI SMS Spam Collection; binary classification into spam "
           "and ham; text preprocessing and two bag-of-words style representations (BoW and TF-IDF); three classical "
           "classifiers (six models in total) tuned with grid search; offline evaluation on a held-out test set; "
           "interpretation of the learned word weights; and a command-line demonstration that classifies new "
           "messages with the saved best model.")
    R.para("**Excluded.** Deep learning and transformer models; messages in languages other than English and "
           "code-mixed messages; deployment on mobile phones or in an operator network; multimedia messages "
           "(MMS) and images; and sender metadata such as phone numbers, time of sending or sending frequency. "
           "These exclusions keep the comparison focused on the text representation and the classifier.")
    R.h2("1.5 Organization of the Report")
    R.bullets([
        "**Chapter 2** reviews earlier work on spam filtering, text representation and classifiers.",
        "**Chapter 3** describes the dataset and presents the exploratory data analysis.",
        "**Chapter 4** explains data cleaning, text preprocessing, feature engineering and the train/test split.",
        "**Chapter 5** describes the workflow, the three algorithms, the implementation and the hyperparameter "
        "tuning.",
        "**Chapter 6** defines the evaluation metrics and presents, compares and discusses the results.",
        "**Chapter 7** concludes the report and lists limitations and future enhancements.",
        "**Appendices A and B** give reproducibility instructions and additional results."])


# ============================================================== Chapter 2
def chapter2(R, D):
    R.h1("Chapter 2 – Literature Survey")
    R.para("Content-based spam filtering builds on term weighting, probabilistic and margin-based text classifiers, "
           "and studies that apply them to SMS. Table 2.1 summarises the sources that introduced the methods and the "
           "dataset used here; each is discussed below.")
    rows = [
        ["1", "G. Salton, C. Buckley (1988)", "Term-weighting approaches in automatic text retrieval — Inf. "
         "Process. Manage.", "Comparison of term-weighting schemes for retrieval", "Weighted single-term indexing "
         "outperforms more elaborate representations; basis of TF-IDF. Studied for retrieval, not classification."],
        ["2", "M. Sahami et al. (1998)", "A Bayesian approach to filtering junk e-mail — AAAI-98 Workshop",
         "Bayesian (Naive Bayes) classifier with differential misclassification costs", "Adding domain-specific "
         "features to the words gives much more accurate filters. Studied on e-mail, not SMS."],
        ["3", "A. McCallum, K. Nigam (1998)", "A comparison of event models for Naive Bayes text classification "
         "— AAAI-98 Workshop", "Multivariate Bernoulli vs multinomial Naive Bayes on five corpora",
         "Multinomial model usually better at larger vocabulary sizes; both models assume word independence."],
        ["4", "T. Joachims (1998)", "Text categorization with support vector machines — ECML-98",
         "Support vector machines for text categorisation", "SVMs are well suited to text and robust without "
         "manual parameter tuning. General text categorisation, not spam."],
        ["5", "J. M. Gómez Hidalgo et al. (2006)", "Content based SMS spam filtering — DocEng ’06",
         "Bayesian filters and other ML algorithms; several message representations; English and Spanish SMS",
         "Examines whether Bayesian e-mail filtering transfers to SMS spam. Predates the public SMS Spam "
         "Collection."],
        ["6", "T. A. Almeida et al. (2011)", "Contributions to the study of SMS spam filtering — DocEng ’11",
         "New public SMS Spam Collection; comparison of established classifiers", "SVM outperformed the other "
         "methods and is proposed as a good baseline. Results on a single, newly collected dataset."],
        ["7", "S. J. Delany et al. (2012)", "SMS spam filtering: Methods and data — Expert Syst. Appl.",
         "Review of content-based SMS spam filtering; analysis of recent SMS spam; new dataset",
         "Content-based SMS spam filtering differs from e-mail spam filtering; reviews research and releases a "
         "new dataset."],
    ]
    # cite in table order so numbering follows first appearance
    keys = ["salton1988", "sahami1998", "mccallum1998", "joachims1998", "gomez2006", "almeida2011", "delany2012"]
    for r, k in zip(rows, keys):
        r[1] = f"{r[1]} {cite(k)}"
    R.table(["Sl. No.", "Author(s) and Year", "Title / Source", "Method Used", "Key Findings / Limitations"],
            rows, "2.1", "Summary of the literature reviewed", widths=[0.45, 1.15, 1.55, 1.5, 1.8], size=9,
            align=["c", "l", "l", "l", "l"])
    R.paras(
        f"**Salton and Buckley (1988)** {cite('salton1988')} summarised experiments with automatic term-weighting "
        "schemes for document retrieval. They argued that weighted single-term indexing performs better than more "
        "elaborate text representations and discussed the term-frequency, inverse-document-frequency and "
        "length-normalisation components of a term weight, on which TF-IDF is based.",
        f"**Sahami et al. (1998)** {cite('sahami1998')} treated junk e-mail filtering as a decision-theoretic "
        "problem and used a probabilistic (Bayesian) classifier with different costs for the two kinds of error. "
        "They showed that adding domain-specific features to the raw words of a message gave much more accurate "
        "filters, which motivates the domain-specific normalisation tokens used here.",
        f"**McCallum and Nigam (1998)** {cite('mccallum1998')} compared two Naive Bayes event models for text: the "
        "multivariate Bernoulli model, which records only whether a word occurs, and the multinomial model, which "
        "uses word counts. Across five corpora, the multinomial model usually performed better at larger vocabulary "
        "sizes. This is the reason the present study uses Multinomial Naive Bayes.",
        f"**Joachims (1998)** {cite('joachims1998')} analysed the particular properties of learning with text "
        "data, such as its very large number of features, and explained why support vector machines are "
        "appropriate for text categorisation. The study reported that SVMs are robust and work without manual "
        "parameter tuning.",
        f"**Gómez Hidalgo et al. (2006)** {cite('gomez2006')} asked whether the Bayesian filtering techniques "
        "that are effective for e-mail spam can be transferred to SMS spam. They built English and Spanish SMS "
        "spam test collections and evaluated several message representations and machine-learning algorithms on "
        "them.",
        f"**Almeida et al. (2011)** {cite('almeida2011')} introduced the SMS Spam Collection, a real, public and "
        "non-encoded collection that the authors described as the largest of its kind at the time. They compared "
        "several established machine-learning methods on it and concluded that SVM outperformed the other methods "
        "and is a good baseline. The present study revisits this comparison under a deduplicated split.",
        f"**Delany et al. (2012)** {cite('delany2012')} motivated content-based SMS spam filtering, discussed how "
        "it differs from e-mail spam filtering, reviewed recent research, analysed recent SMS spam and released a "
        "further dataset.")
    R.h2("2.1 Research Gap and Motivation")
    R.paras(
        "Many studies report a single text representation or a single metric such as accuracy, which looks high on "
        "imbalanced data even when many spam messages are missed. Results are also hard to compare across papers "
        "because preprocessing, splits and tuning protocols differ, and duplicate messages—common in this "
        "dataset—may fall into both the training and the test data.",
        "This case study therefore makes a controlled comparison of BoW and TF-IDF across three classifiers under "
        "the same preprocessing, the same deduplicated and stratified train/test split and the same tuning "
        "protocol (grid search with 5-fold cross-validation on F1). It reports imbalance-aware metrics—precision, "
        "recall, F1, PR-AUC and the false positive rate—together with an error analysis and a statistical test "
        "between the two best models. The aim is a careful, reproducible comparison on one public dataset; no "
        "novelty is claimed for the methods themselves.")


# ============================================================== Chapter 3
def chapter3(R, D):
    e, info = D["eda"], D["info"]
    raw, cl = e["raw_class_counts"], e["clean_class_counts"]
    R.h1("Chapter 3 – Dataset Description and Exploratory Data Analysis")
    R.h2("3.1 Dataset Source")
    comp = info["composition"]
    claim(info["composition_matches_file"], "per-source counts on the UCI page add up to the file's class counts")
    R.paras(
        f"The study uses the **{info['name']}** {cite('uci')}, created by {info['creators']} and hosted by the "
        f"{info['repository']} (https://archive.ics.uci.edu/dataset/228/sms+spam+collection, DOI "
        f"{info['doi']}). The dataset was donated to the repository on {info['donated']} and was accessed for this "
        f"study on {info['date_accessed']}. It is distributed under the {info['licence']} licence, which permits "
        "sharing and adaptation for any purpose provided that appropriate credit is given. The accompanying paper "
        f"by Almeida *et al.* {cite('almeida2011')} describes its collection.",
        "The data are a single plain-text file, `SMSSpamCollection`, with one message per line in the form "
        "`label<TAB>message` and no header row. The label is either `ham` or `spam`. According to the UCI page, the "
        f"collection combines four free sources: {comp[0]['spam']} spam messages manually extracted from the "
        f"Grumbletext Web site, a UK forum on which mobile phone users report SMS spam; {n(comp[1]['ham'])} ham "
        "messages chosen at random from the NUS SMS Corpus, collected at the National University of Singapore "
        f"mostly from students; {comp[2]['ham']} ham messages from Caroline Tagg’s PhD thesis; and the SMS Spam "
        f"Corpus v.0.1 Big with {n(comp[3]['ham'])} ham and {comp[3]['spam']} spam messages. These sources add up to "
        f"{n(info['readme_totals']['ham'])} ham and {info['readme_totals']['spam']} spam messages, which matches "
        f"the {n(raw['ham'])} ham and {raw['spam']} spam messages counted in the downloaded file.")
    R.h2("3.2 Dataset Overview")
    R.para("Table 3.1 gives an overview of the dataset. The record counts are reported both for the raw file and "
           "after exact duplicate messages were removed (Section 3.4.8 and Section 4.1).")
    t31 = D["t31"]
    R.table(["Attribute", "Value"], t31.values.tolist(), "3.1", "Dataset overview", widths=[2.2, 4.2], size=10,
            align=["l", "l"])
    R.h2("3.3 Attribute Description")
    R.para("The dataset has only two original attributes: the class label and the message text. Several numeric "
           "attributes were derived from the text to support the exploratory analysis. Table 3.2 lists all of them. "
           "The derived attributes were used only for EDA; the classifiers receive only the message text, so that "
           "the comparison concerns the text representation alone.")
    R.table(list(D["t32"].columns), D["t32"].values.tolist(), "3.2", "Attribute description",
            widths=[1.05, 1.3, 2.75, 1.35], size=9.5, align=["l", "l", "l", "l"])

    R.h2("3.4 Exploratory Data Analysis")
    R.h3("3.4.1 Shape, Sample Records and Data Types")
    dt = D["tdt"]
    R.para(f"After loading with tab separation and quote handling disabled (messages contain quote characters), the "
           f"raw data frame has {n(e['n_raw'])} rows and two columns. After cleaning it has {n(e['n_clean'])} rows. "
           f"There are {e['missing_values']} missing values. The label and message are stored as text (pandas "
           f"`{dt.set_index('Column').loc['label', 'dtype']}` columns); the encoded target and the derived counts "
           f"are integers (`{dt.set_index('Column').loc['target', 'dtype']}`). Table 3.3 shows five randomly "
           "selected messages of each class (long messages are shortened). Even this small sample shows the typical "
           "contrast: ham messages are conversational and often abbreviated, whereas spam messages advertise "
           "offers, mention prices and ask the recipient to call or text a number.")
    t33 = D["t33"]
    R.table(["Label", "Message (shortened)", "Chars", "Words"], t33.values.tolist(), "3.3",
            "Sample records (5 ham and 5 spam)", widths=[0.55, 4.6, 0.6, 0.65], size=9,
            align=["c", "l", "c", "c"])
    R.h3("3.4.2 Summary Statistics")
    t34 = D["t34"]
    R.para(f"Table 3.4 gives summary statistics of message length and digit count by class. The mean spam message "
           f"has {e['mean_chars']['spam']:.2f} characters against {e['mean_chars']['ham']:.2f} for ham, and "
           f"{e['mean_words']['spam']:.2f} words against {e['mean_words']['ham']:.2f}. The difference in digits is "
           f"much larger: on average a spam message contains {e['mean_digits']['spam']:.2f} digits, while a ham "
           f"message contains {e['mean_digits']['ham']:.2f}. Ham messages are also far more variable in length: the "
           f"longest message in the dataset ({n(e['max_chars'])} characters) is ham, while spam lengths are "
           f"concentrated in a narrow band, as the smaller standard deviation for spam shows.")
    claim(e["mean_digits"]["spam"] > 10 * e["mean_digits"]["ham"], "spam has >10x the digits of ham")
    rows = [[r.Class, r.Feature, f"{r.Mean:.2f}", f"{r.Median:.2f}", f"{r.Std:.2f}", r.Min, r.Max]
            for r in t34.itertuples()]
    R.table(["Class", "Feature", "Mean", "Median", "Std. dev.", "Min", "Max"], rows, "3.4",
            "Summary statistics of length and digit count by class", widths=[0.8, 1.2, 0.9, 0.9, 0.9, 0.7, 0.7],
            size=10)
    R.h3("3.4.3 Class Distribution")
    R.para(f"Fig. 3.1 shows the class distribution before and after removing duplicate messages. In the raw file, "
           f"{p2(100 - e['raw_spam_pct'])} of the messages are ham and {p2(e['raw_spam_pct'])} are spam. After "
           f"deduplication the share of spam falls slightly to {p2(e['spam_pct'])} ({n(cl['spam'])} spam against "
           f"{n(cl['ham'])} ham), because duplicates are relatively more common among spam messages. The ratio is "
           f"about {e['imbalance_ratio_ham_per_spam']:.2f} ham messages per spam message. This imbalance has two "
           "consequences for the rest of the study. First, accuracy is a poor guide: a model that labels every "
           f"message as ham would already be {p2(e['ham_pct'])} accurate while detecting no spam at all. Second, the "
           "train/test split must be stratified so that both subsets keep the same class ratio, and class weighting "
           "should be considered during training.")
    R.figure(FIG / "fig_3_1_class_distribution.png", "3.1",
             "Class distribution before and after removing duplicate messages", width=5.0)
    R.h3("3.4.4 Message Length")
    claim(e["median_chars"]["spam"] > 2 * e["median_chars"]["ham"], "median spam length > 2x ham")
    R.para(f"Fig. 3.2 shows the distribution of message length in characters for each class, normalised so that "
           f"both classes have the same area. Ham messages are mostly short, with a median of "
           f"{e['median_chars']['ham']:.0f} characters and a long right tail. Spam messages have a median of "
           f"{e['median_chars']['spam']:.0f} characters and are concentrated just below the 160-character limit of "
           "a single SMS: senders of bulk messages use almost all of the available space. Only "
           f"{p2(e['pct_over_200_chars']['spam'])} of spam messages are longer than 200 characters, against "
           f"{p2(e['pct_over_200_chars']['ham'])} of ham messages, so the very long messages in the dataset are "
           "almost all personal conversation. Fig. 3.3 shows the same pattern in words: the median spam message has "
           f"{e['median_words']['spam']:.0f} words and the median ham message {e['median_words']['ham']:.0f}, and the "
           "spam box is narrow while the ham box has many long outliers. Length is therefore informative, but the two "
           "distributions overlap considerably in the middle of the range, so length alone cannot separate the classes.")
    R.figure(FIG / "fig_3_2_length_chars.png", "3.2", "Distribution of message length (characters) by class",
             width=5.3)
    R.figure(FIG / "fig_3_3_length_words.png", "3.3", "Word count per message by class", width=3.6)
    R.h3("3.4.5 Digits, URLs and Currency Symbols")
    R.para(f"Fig. 3.4 (left) shows that digits are a strong signal. Most ham messages contain no digits at all "
           f"(median {e['median_digits']['ham']:.0f}), whereas the median spam message contains "
           f"{e['median_digits']['spam']:.0f} digits. The right panel shows the share of messages that contain at "
           f"least one digit, a URL or a currency symbol: {p2(e['pct_with_digits']['spam'])} of spam messages contain "
           f"a digit against {p2(e['pct_with_digits']['ham'])} of ham; {p2(e['pct_with_url']['spam'])} of spam "
           f"messages contain a URL against {p2(e['pct_with_url']['ham'])} of ham; and "
           f"{p2(e['pct_with_currency']['spam'])} of spam messages contain a currency symbol (mostly £) against "
           f"{p2(e['pct_with_currency']['ham'])} of ham. In addition, {p2(e['pct_5plus_digit_run']['spam'])} of spam "
           f"messages contain a run of five or more digits—typically a premium-rate phone number or a short "
           f"code—compared with {p2(e['pct_5plus_digit_run']['ham'])} of ham messages. Because the exact numbers "
           "differ from message to message, a word-based model would treat each phone number as a separate rare "
           "word. This observation motivates replacing them with shared normalisation tokens (Section 4.3).")
    R.figure(FIG / "fig_3_4_digits_by_class.png", "3.4",
             "Digit count per message (left) and share of messages containing digits, URLs and currency symbols "
             "(right) by class", width=5.5)
    R.h3("3.4.6 Frequent Words and Word Clouds")
    top_h = [t for t, _ in e["top20_ham"]]
    top_s = [t for t, _ in e["top20_spam"]]
    norm = {"numtoken", "phonetoken", "moneytoken", "urltoken", "emailtoken"}
    s_words = [t for t in top_s if t not in norm][:10]
    h_words = [t for t in top_h if t not in norm][:10]
    R.para("Figs. 3.5 and 3.6 show the 20 most frequent tokens in ham and spam after the preprocessing described in "
           "Chapter 4 (lowercasing, normalisation tokens and stop-word removal). The two lists are strikingly "
           f"different. Ham is dominated by conversational words such as “{'”, “'.join(h_words[:8])}”, including "
           "SMS shorthand (“ur”, “lor”, “da”). Spam is dominated by the normalisation tokens for numbers, phone "
           "numbers and money, followed by words that promote offers or ask for action: "
           f"“{'”, “'.join(s_words[:10])}”. The token `numtoken` is frequent in both classes because ordinary numbers "
           "(times, dates, quantities) also occur in ham, whereas `phonetoken` and `moneytoken` are almost exclusive "
           f"to spam. The spam vocabulary is also much smaller ({n(e['vocab_size_spam'])} distinct tokens against "
           f"{n(e['vocab_size_ham'])} for ham), which reflects the repetitive, template-like wording of bulk messages.")
    R.figure_pair((FIG / "fig_3_5_top_words_ham.png", "3.5", "Top 20 tokens in ham messages"),
                  (FIG / "fig_3_6_top_words_spam.png", "3.6", "Top 20 tokens in spam messages"), width=3.05)
    R.para("The word clouds in Fig. 3.7 give the same picture at a glance, with word size proportional to frequency. "
           "The ham cloud is spread over many everyday words of similar size, while the spam cloud is dominated by a "
           "few tokens related to numbers, prizes, replies and services.")
    R.figure(FIG / "fig_3_7_wordclouds.png", "3.7", "Word clouds of ham (left) and spam (right) messages", width=5.5)
    R.h3("3.4.7 Correlation of Engineered Features")
    cwt = e["corr_with_target"]
    top_feat = max(cwt, key=cwt.get)
    claim(top_feat == "n_digits", "digit count is the derived feature most correlated with the target")
    R.para("Fig. 3.8 shows the Pearson correlation between the derived numeric features and the target (spam = 1). "
           f"The digit count has the strongest correlation with the target (r = {cwt['n_digits']:.2f}), followed by the "
           f"presence of a currency symbol (r = {cwt['has_currency']:.2f}), the character count "
           f"(r = {cwt['n_chars']:.2f}), the presence of a URL (r = {cwt['has_url']:.2f}) and the number of uppercase "
           f"letters (r = {cwt['n_upper']:.2f}). Character and word counts are almost perfectly correlated with each "
           f"other (r = {e['corr_chars_words']:.2f}), so they carry nearly the same information. These derived "
           "features were not given to the classifiers; the digit, URL and currency information reaches the models "
           "through the normalisation tokens instead, which keeps every model purely text-based.")
    R.figure(FIG / "fig_3_8_feature_correlation.png", "3.8",
             "Correlation heatmap of the engineered numeric features and the target", width=4.0)
    R.h3("3.4.8 Duplicate Messages")
    t35 = D["t35"]
    R.para(f"The dataset contains many exact duplicates: {n(e['n_duplicates_removed'])} messages "
           f"({p2(e['duplicates_pct_of_raw'])} of the raw file) are exact copies of an earlier message. Table 3.5 "
           f"shows that {n(e['duplicates_by_class']['spam'])} of them are spam and {n(e['duplicates_by_class']['ham'])} "
           "are ham, so spam is duplicated proportionally more often—a natural consequence of bulk sending. No message "
           f"text occurs with both labels ({e['n_texts_with_conflicting_labels']} conflicts). If duplicates were kept, "
           "the same message could appear in both the training and the test set, and the model would be rewarded for "
           "memorising it; the duplicates were therefore removed before splitting (Section 4.1).")
    rows = [[r.Class, n(r.Before), n(r._3), n(r.After), p2(r._5)] for r in t35.itertuples()]
    R.table(["Class", "Before", "Duplicates removed", "After", "% removed"], rows, "3.5",
            "Messages before and after removing exact duplicates", widths=[1.0, 1.0, 1.6, 1.0, 1.1], size=10)
    R.h2("3.5 Observations")
    R.para("The exploratory analysis leads to the following observations, each of which shapes a design decision in "
           "the later chapters:")
    R.bullets([
        f"**Class imbalance** (spam = {p2(e['spam_pct'])}): evaluation uses F1, precision, recall and PR-AUC rather "
        "than accuracy alone; the split is stratified; and `class_weight = \"balanced\"` is included in the "
        "hyperparameter search.",
        "**Spam is longer and digit-heavy**, with phone numbers, prices and URLs: numbers, URLs, e-mail addresses "
        "and currency symbols are replaced by normalisation tokens so that the model learns the pattern rather than "
        "individual numbers.",
        f"**Duplicates are common** ({n(e['n_duplicates_removed'])} messages): exact duplicates are removed before the "
        "train/test split to prevent leakage and inflated scores.",
        "**Frequent function words** such as “to”, “you” and “the” carry little class information on their own: "
        "stop-word removal reduces noise and vocabulary size (its effect is measured in Appendix B).",
        "**Strong lexical cues** (“free”, “txt”, “claim”, “prize”, “cash”, “urgent”, phone numbers and prices): "
        "linear models on word features are expected to perform well, and their weights can be inspected to see "
        "which words drive the decision."])


# ============================================================== Chapter 4
def chapter4(R, D):
    e, log = D["eda"], D["log"]
    R.h1("Chapter 4 – Data Preprocessing and Feature Engineering")
    R.h2("4.1 Data Cleaning")
    R.paras(
        f"**Missing values.** The raw file was checked for missing labels or messages; {e['missing_values']} were "
        "found, so no rows were removed or imputed for this reason.",
        f"**HTML entities.** {n(e['n_html_entity_messages'])} messages contain HTML character entities such as "
        "`&lt;`, `&gt;` and `&amp;`, left over from the web sources of the data. They were converted back to the "
        "characters they represent with Python’s `html.unescape`, and leading and trailing whitespace was stripped.",
        f"**Duplicates.** After these steps, exact duplicate messages were removed, keeping the first occurrence. "
        f"This removed {n(e['n_duplicates_removed'])} messages ({n(e['duplicates_by_class']['ham'])} ham and "
        f"{n(e['duplicates_by_class']['spam'])} spam), leaving {n(e['n_clean'])} unique messages. Deduplication was "
        "done before the train/test split. If a message occurred in both subsets, the test score would partly "
        "measure memorisation instead of generalisation and would be inflated. This decision makes the scores in "
        "this report somewhat harder to compare with studies that did not deduplicate, which is noted as a "
        "limitation in Chapter 7.",
        "**Label encoding.** The target was encoded as ham = 0 and spam = 1. Spam is the positive class for every "
        "metric in this report.",
        "**No outlier removal.** Very long messages were kept. They are genuine messages—the longest ones are "
        f"personal ham messages of up to {n(e['max_chars'])} characters—and removing them would make the training "
        "data less representative of what a filter would see. Long messages also do not distort the chosen models: "
        "TF-IDF vectors are length-normalised, and Naive Bayes, Logistic Regression and linear SVMs do not rely on "
        f"distances between raw feature values. Two messages consisting only of emoticons (for example “:)”) become "
        f"empty after cleaning; they were also kept ({e['empty_after_cleaning']} messages), and the model simply "
        "classifies them from its prior or intercept.")
    R.h2("4.2 Encoding and Feature Scaling")
    R.para("Machine-learning classifiers need numeric input, so each cleaned message is converted into a vector with "
           "one dimension per vocabulary term. Two representations are compared.")
    R.para("**Bag of Words (BoW).** Each message *d* is represented by the raw count of each term *t*:")
    R.equation("tf(t, d) = number of times term t occurs in message d", "4.1")
    R.para(f"**TF-IDF.** The count is multiplied by an inverse document frequency (IDF) that down-weights terms "
           f"occurring in many messages {cite('salton1988')}. With *n* training messages and df(t) the number of "
           f"training messages containing *t*, scikit-learn {cite('pedregosa2011')} uses the smoothed form")
    R.equation("idf(t) = ln[(1 + n) / (1 + df(t))] + 1", "4.2")
    R.equation("tfidf(t, d) = tf(t, d) × idf(t)", "4.3")
    R.para("and each message vector **v** is then divided by its Euclidean (L2) norm so that all messages have unit "
           "length:")
    R.equation("v̂ = v / ‖v‖₂ = v / √(v₁² + v₂² + … + v_{m}²)", "4.4")
    R.para("When `sublinear_tf` is enabled (a tuned option), the count tf is replaced by 1 + ln(tf), which reduces "
           "the influence of a word repeated many times in one message.")
    R.para("**Scaling.** No further feature scaling was applied. TF-IDF vectors are already L2-normalised, so all "
           "messages are on the same scale; BoW counts are small non-negative integers, and Multinomial Naive Bayes "
           "requires non-negative count-like features, which rules out standardisation (which would create negative "
           "values). The target needs only the label encoding described above.")
    msgs, bow, tf = D["t42m"], D["t42b"], D["t42t"]
    tokens = [c for c in bow.columns if c != "msg"]
    idf = tf[tf.msg == "idf"].iloc[0]
    high = [t for t in tokens if float(idf[t]) == max(float(idf[x]) for x in tokens)]
    low = [t for t in tokens if float(idf[t]) == min(float(idf[x]) for x in tokens)]
    m2_clean = msgs.iloc[1]["cleaned"]
    claim("free" in low, "'free' (in every example message) has the lowest idf")
    R.para("**Worked example.** Table 4.1 applies both representations, with exactly the settings used by the models, "
           f"to three short messages: m1 = “{msgs.iloc[0]['message']}”, m2 = “{msgs.iloc[1]['message']}” and "
           f"m3 = “{msgs.iloc[2]['message']}”. After cleaning, the phone number in m1 becomes `phonetoken`, and stop "
           f"words are removed. Message m2 (“{m2_clean}” after cleaning) keeps only “free”, because all of its other "
           "words—including “call”—are in the scikit-learn English stop-word list. The BoW rows simply count the "
           f"remaining terms. In the TF-IDF rows, “{low[0]}” occurs in all three messages and receives the lowest "
           f"weight (idf = {float(idf[low[0]]):.4f}), while “{'” and “'.join(high)}” occur in only one message and "
           f"receive the highest idf ({float(idf[high[0]]):.4f}); after L2 normalisation, m2, which contains a single "
           "term, has a weight of 1 for that term.")
    rows = []
    for r in bow.itertuples(index=False):
        rows.append([f"{r.msg} (BoW count)"] + [str(getattr(r, t)) for t in tokens])
    for r in tf.itertuples(index=False):
        label = "idf(t)" if r.msg == "idf" else f"{r.msg} (TF-IDF)"
        rows.append([label] + [f"{float(getattr(r, t)):.4f}" for t in tokens])
    R.table(["Row"] + tokens, rows, "4.1", "Worked example of BoW counts and TF-IDF weights for three short "
            "messages", widths=[1.6] + [0.95] * len(tokens), size=10)
    R.h2("4.3 Feature Engineering")
    R.para("Feature engineering in this study means turning raw text into informative tokens. The following steps "
           "are implemented in `src/preprocess.py` and are applied inside the vectorizer, so they run identically "
           "during training, cross-validation, testing and prediction:")
    stop_hits = e["top60_spam_tokens_in_stoplist"]
    R.bullets([
        "**Lowercasing**, so that “FREE”, “Free” and “free” become the same term.",
        "**Normalisation tokens.** URLs (`http…` or `www.…`) become `urltoken`, e-mail addresses become "
        "`emailtoken`, the symbols £, $ and € become `moneytoken`, digit runs of five or more digits (phone numbers "
        "and short codes) become `phonetoken`, and all other numbers become `numtoken`. This follows directly from "
        f"the EDA: {p2(e['pct_5plus_digit_run']['spam'])} of spam messages contain a five-or-more-digit run and "
        f"{p2(e['pct_with_currency']['spam'])} a currency symbol, but each specific number is rare.",
        "**Character filtering.** Every character other than the letters a–z and spaces is removed, which also "
        "removes punctuation and emoticons.",
        "**Tokenization** with the regular expression `\\b[a-z][a-z]+\\b`, which keeps tokens of two or more letters.",
        f"**Stop-word removal** with scikit-learn’s built-in English list of {e['n_stop_words']} words. The list "
        f"also contains some words that are frequent in spam, for example "
        f"“{'”, “'.join([w for w in stop_hits if w in ('call', 'now', 'get', 'please', 'only')])}” "
        f"(the word “call” occurs {n(e['call_count_spam'])} times in spam and {n(e['call_count_ham'])} times in ham). "
        "The net effect of stop-word removal is measured in the ablation study in Appendix B.",
        "**n-grams and minimum document frequency.** Unigrams only, (1,1), or unigrams plus bigrams, (1,2), and a "
        "minimum document frequency of 1 or 2 were treated as hyperparameters and chosen by cross-validation "
        "(Section 5.4). Bigrams can capture short phrases such as “won moneytoken” or “box numtoken”.",
        f"**Stemming** was not used in the main pipeline, to keep tokens readable in the interpretation of the "
        f"models; Porter stemming {cite('porter1980')} was tested separately in the ablation study."])
    cv = D["cv"]
    vs = {k: int(cv.loc[k, "vocab_size"]) for k in cv.index}
    R.para("The vocabulary size depends on the selected n-gram range and minimum document frequency. For the tuned "
           f"models it ranges from {n(min(vs.values()))} to {n(max(vs.values()))} terms (for example "
           f"{n(vs['BoW_LogisticRegression'])} terms for {pretty('BoW_LogisticRegression')} and "
           f"{n(vs['TF-IDF_LinearSVM'])} terms for {pretty('TF-IDF_LinearSVM')}; Table 6.2 lists all six). By "
           f"comparison, the full deduplicated dataset contains {n(e['vocab_size_eda'])} distinct unigram tokens after "
           "preprocessing. The numeric features derived for the EDA (length, digit count, uppercase count, URL and "
           "currency flags) were deliberately **not** added to the model input. This keeps the comparison purely about "
           "the text representation; the most useful of these signals reach the model anyway through the normalisation "
           "tokens.")
    R.h2("4.4 Train/Test Split")
    t41 = D["t41"]
    tr = t41[t41.Subset == "Training"].iloc[0]
    te = t41[t41.Subset == "Testing"].iloc[0]
    R.para(f"The {n(e['n_clean'])} deduplicated messages were split into a training set and a test set with a "
           f"stratified 80/20 split (`train_test_split`, `test_size=0.20`, `stratify=y`, `random_state=42`). The "
           f"training set has {n(tr.Records)} messages and the test set {n(te.Records)}; both keep the original spam "
           f"share ({p2(tr.Spam_percent)} and {p2(te.Spam_percent)}). No separate validation set was held out. Instead, "
           "validation is 5-fold stratified cross-validation inside the training set: the training data are divided "
           "into five folds, and each fold is held out once while the model is trained on the other four. Table 4.2 "
           "summarises the subsets.")
    fmt = lambda v: n(v) if str(v).isdigit() else str(v).replace("-", "–")
    rows = [[r.Subset, fmt(r.Records), f"{float(r.Percent):.2f}%", fmt(r.Ham), fmt(r.Spam),
             f"{float(r.Spam_percent):.2f}%", r.Note] for r in t41.itertuples()]
    R.table(["Subset", "Messages", "% of total", "Ham", "Spam", "Spam %", "Note"], rows, "4.2",
            "Training, validation and test subsets", widths=[0.85, 0.95, 0.7, 0.65, 0.6, 0.6, 2.1], size=9,
            align=["l", "c", "c", "c", "c", "c", "l"])
    R.para("**Leakage prevention.** The vectorizer (BoW or TF-IDF) is the first step of a scikit-learn `Pipeline`, "
           "followed by the classifier. During cross-validation, the whole pipeline is re-fitted on the four training "
           "folds of each split, so the vocabulary and IDF weights are learned without seeing the held-out fold. The "
           "final model is fitted on the training set only. Together with deduplication before the split, this ensures "
           "that no information from the test messages influences training or hyperparameter tuning. The test set is "
           "used once, to evaluate the six tuned models in Chapter 6; because the best of the six is then chosen by its "
           "test score, its score is slightly optimistic, and the cross-validation results are reported alongside as a "
           "check.")


# ============================================================== Chapter 5
def _params(D, name):
    return json.loads(D["cv"].loc[name, "best_params"])


def chapter5(R, D):
    env, cv = D["env"], D["cv"]
    R.h1("Chapter 5 – Model Development")
    R.h2("5.1 Workflow")
    R.para("Fig. 5.1 shows the workflow. The orange boxes inside the dashed frame form one scikit-learn `Pipeline` "
           "that is fitted only on training data.")
    R.figure(FIG / "fig_5_1_workflow.png", "5.1", "Workflow of the SMS spam detection study", width=5.0)
    R.bullets([
        "**Dataset and cleaning (1–2):** download, label encoding, HTML unescaping and deduplication (Section 4.1).",
        "**EDA (3):** describes the data and motivates the preprocessing (Chapter 3).",
        "**Split (4):** stratified 80/20; the test set is put aside (Section 4.4).",
        "**Pipeline (5–8):** preprocessing, BoW or TF-IDF features and a classifier, tuned together by `GridSearchCV` "
        "with 5-fold stratified cross-validation and re-fitted on the full training set.",
        "**Evaluation and comparison (9–10):** each tuned pipeline is scored once on the test set (Chapter 6).",
        "**Prediction (11):** the saved best pipeline classifies new messages from the command line."])
    R.h2("5.2 Algorithms Used")
    R.para("Three classifiers were chosen: a generative probabilistic model (Multinomial Naive Bayes) and two "
           "discriminative linear models (Logistic Regression and a linear Support Vector Machine). All three are "
           "standard, fast and well suited to high-dimensional sparse text features. In the formulas below, a message "
           "is represented by its feature vector **x** = (x₁, …, x_{m}) over a vocabulary *V* of *m* terms, and the "
           "class is *c* ∈ {ham, spam}.")
    nb_b, nb_t = _params(D, "BoW_MultinomialNB"), _params(D, "TF-IDF_MultinomialNB")
    R.h3("5.2.1 Algorithm 1: Multinomial Naive Bayes")
    R.para("Naive Bayes applies Bayes’ rule to obtain the probability of a class given the message:")
    R.equation("P(c | x) = P(c) · P(x | c) / P(x)", "5.1")
    R.para("It assumes that, given the class, the terms of a message occur independently of each other. Under the "
           f"multinomial event model {cite('mccallum1998')}, each occurrence of a term is drawn from a class-specific "
           "word distribution, so")
    R.equation("P(c | x) ∝ P(c) · ∏_{t∈V} P(t | c)^{xₜ}", "5.2")
    R.para("where P(c) is the share of class *c* in the training data. The word probabilities are estimated with "
           "additive (Laplace/Lidstone) smoothing so that a term never seen in one class does not force the product "
           "to zero:")
    R.equation("P(t | c) = (N_{c,t} + α) / (N_{c} + α·|V|)", "5.3")
    R.para("Here N_{c,t} is the total count (or TF-IDF weight) of term *t* in training messages of class *c*, N_{c} is "
           "the total over all terms, and α > 0 is the smoothing parameter (α = 1 is Laplace smoothing). Prediction "
           "uses logarithms to avoid numerical underflow:")
    R.equation("ĉ = argmax_{c} [ log P(c) + Σ_{t} x_{t} · log P(t | c) ]", "5.4")
    R.para(f"**Why it was chosen.** Multinomial Naive Bayes trains in a single pass over the data, has one "
           f"hyperparameter and is the classic baseline for spam filtering {cite('sahami1998')}. **Limitations.** The "
           "independence assumption is clearly false for language (for example “call” and “now” tend to occur "
           "together), which tends to push predicted probabilities towards 0 or 1, so they should not be read as "
           "calibrated probabilities. The model also cannot weigh evidence discriminatively the way a linear "
           f"classifier can. The grid searched α ∈ {{{grid_values(D, 'BoW_MultinomialNB', 'clf__alpha')}}}; the "
           "selected values were "
           f"α = {nb_b['clf__alpha']} with BoW and α = {nb_t['clf__alpha']} with TF-IDF.")
    R.h3("5.2.2 Algorithm 2: Logistic Regression")
    R.para("Logistic Regression models the probability of spam directly as a sigmoid function of a linear score:")
    R.equation("P(y = 1 | x) = σ(wᵀx + b) = 1 / (1 + e^{−(wᵀx + b)})", "5.5")
    R.para("The weights **w** and bias *b* are found by minimising the regularised log-loss. With labels "
           "y_{i} ∈ {−1, +1}, the L2-regularised form solved by the liblinear solver is")
    R.equation("min_{w,b}  ½‖w‖² + C · Σ_{i} s_{i} · log(1 + e^{−yᵢ(wᵀxᵢ + b)})", "5.6")
    R.para("where *C* is the inverse of the regularisation strength (small *C* = strong regularisation, simpler model) "
           "and s_{i} is a sample weight. With `class_weight=\"balanced\"`, s_{i} = n / (2·n_{c(i)}), so errors on the "
           "minority (spam) class cost more. A message is labelled spam when P(y = 1 | x) > 0.5.")
    R.para("**Why it was chosen.** Logistic Regression gives probabilistic outputs, its training problem is convex "
           "(a unique optimum) and each coefficient shows how strongly a term pushes a message towards spam or ham, "
           "which makes the model easy to interpret (Section 6.3). **Limitations.** The decision boundary is linear in "
           "the features, so interactions between words are captured only through n-grams, and performance depends on "
           f"choosing *C* well. The grid searched C ∈ {{{grid_values(D, 'BoW_LogisticRegression', 'clf__C')}}} and "
           f"class_weight ∈ {{{grid_values(D, 'BoW_LogisticRegression', 'clf__class_weight')}}}.")
    R.h3("5.2.3 Algorithm 3: Linear Support Vector Machine")
    R.para(f"A Support Vector Machine {cite('cortes1995')} looks for the separating hyperplane with the largest margin "
           "between the two classes. For a linear SVM the decision function is")
    R.equation("f(x) = wᵀx + b,   predict spam if f(x) > 0", "5.7")
    R.para("The margin between the two supporting hyperplanes f(x) = ±1 is 2/‖w‖, so maximising it is the same as "
           "minimising ‖w‖. Messages that violate the margin are penalised with a hinge-type loss, giving the "
           "soft-margin problem")
    R.equation("min_{w,b}  ½‖w‖² + C · Σ_{i} s_{i} · max(0, 1 − y_{i} f(x_{i}))²", "5.8")
    R.para("scikit-learn’s `LinearSVC` uses this squared hinge loss with an L2 penalty by default; the ordinary hinge "
           "loss uses the first power instead of the square. *C* again trades margin width against training errors, "
           "and the same balanced class weights are available.")
    R.para(f"**Why it was chosen.** Linear SVMs work well with the very high-dimensional, sparse feature vectors of "
           f"text {cite('joachims1998')}, and an SVM was the strongest method in the original study of this dataset "
           f"{cite('almeida2011')}. **Limitations.** An SVM does not produce probabilities; the signed distance f(x) "
           "is used as a score instead (this is valid for ROC and precision–recall curves, which depend only on the "
           "ranking of scores). Like Logistic Regression, it is a linear model. The grid searched "
           f"C ∈ {{{grid_values(D, 'BoW_LinearSVM', 'clf__C')}}} and "
           f"class_weight ∈ {{{grid_values(D, 'BoW_LinearSVM', 'clf__class_weight')}}}.")
    R.h2("5.3 Implementation Details")
    pk = env["packages"]
    R.para(f"The study was implemented in Python {env['python_version']} with scikit-learn {cite('pedregosa2011')} "
           f"for preprocessing, models and evaluation, pandas {cite('mckinney2010')} for data handling, matplotlib "
           f"{cite('hunter2007')} and seaborn {cite('waskom2021')} for figures, and NLTK {cite('bird2009')} for the "
           f"Porter stemmer used in the ablation study {cite('nltkdocs')}. No NLTK data downloads are needed. The "
           "software and hardware environment, recorded automatically by `run_all.py` in "
           "`outputs/results/env.json` and `versions.txt`, was:")
    R.bullets([
        f"**Language and libraries:** Python {env['python_version']}; scikit-learn {pk['scikit-learn']}; "
        f"NumPy {pk['numpy']}; pandas {pk['pandas']}; SciPy {pk['scipy']}; matplotlib {pk['matplotlib']}; "
        f"seaborn {pk['seaborn']}; wordcloud {pk['wordcloud']}; NLTK {pk['nltk']}; joblib {pk['joblib']}; "
        f"python-docx {pk['python-docx']}.",
        f"**Development environment:** Python {env['python_version']} virtual environment; scripts run from the "
        f"command line and the notebook executed with Jupyter (nbconvert {pk['nbconvert']}).",
        f"**Hardware:** {env['cpu'] or 'CPU not detected'}"
        + (f" ({env['logical_cores']} logical cores)" if env.get("logical_cores") else "")
        + (f", {env['ram_gb']:.0f} GB RAM" if env.get("ram_gb") else "")
        + f", {env['os']} ({env['machine']})."])
    R.para("The code is organised as a small Python package (`src/`) with one module per stage, and the notebook "
           "`notebooks/SMS_Spam_Detection.ipynb` calls these modules and shows their outputs: `config.py` (paths, "
           "seed 42, test size, folds and grids), `data.py` (download, cleaning, deduplication and split), "
           "`preprocess.py` (`clean_text()`, tokenizer, stop words), `eda.py` (Figs. 3.1–3.8, Tables 3.1–3.5), "
           "`train.py` (six pipelines and `GridSearchCV`), `evaluate.py` (test metrics, curves, error analysis, "
           "McNemar test), `interpret.py` (informative terms), `ablation.py` (Appendix B), `predict.py` "
           "(command-line prediction) and `flowchart.py` (Fig. 5.1).")
    R.para("The core of the training code is shown below. The vectorizer is a step of the pipeline, so it is "
           "re-fitted inside every cross-validation split:")
    R.code('''pipe = Pipeline([
    ("vec", TfidfVectorizer(preprocessor=clean_text, stop_words=STOP_WORDS,
                            token_pattern=r"\\b[a-z][a-z]+\\b")),
    ("clf", LinearSVC(dual="auto", random_state=42)),
])
grid = {"vec__ngram_range": [(1, 1), (1, 2)], "vec__min_df": [1, 2],
        "vec__sublinear_tf": [False, True],
        "clf__C": [0.01, 0.1, 1, 10],
        "clf__class_weight": [None, "balanced"]}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
search = GridSearchCV(pipe, grid, scoring="f1", cv=cv, n_jobs=-1, refit=True)
search.fit(X_train, y_train)        # the test set is not used here''')
    R.h2("5.4 Hyperparameter Tuning")
    total_fits = int(cv["n_fits"].sum())
    total_time = float(cv["search_time_s"].sum())
    R.para("Hyperparameters were tuned by exhaustive grid search over the vectorizer and classifier settings together, "
           "using 5-fold stratified cross-validation on the training set and the F1 score of the spam class as the "
           "selection criterion (accuracy would favour models that neglect the minority class). The best "
           "configuration of each pipeline was then re-fitted on the whole training set. Across the six pipelines, "
           f"{int(cv['n_candidates'].sum())} configurations were evaluated ({n(total_fits)} model fits), which took "
           f"{total_time:.1f} seconds in total on the hardware above. Fixed settings were `solver=\"liblinear\"` and "
           "`max_iter=2000` for Logistic Regression, and `dual=\"auto\"` and `max_iter=10000` for the linear SVM. "
           "Table 5.1 lists the search ranges and the selected values.")
    pname = {"vec__ngram_range": "ngram_range", "vec__min_df": "min_df", "vec__sublinear_tf": "sublinear_tf",
             "clf__alpha": "alpha (α)", "clf__C": "C", "clf__class_weight": "class_weight"}
    rows = []
    order = ["vec__ngram_range", "vec__min_df", "vec__sublinear_tf", "clf__alpha", "clf__C", "clf__class_weight"]
    for v in VECS:
        for c in CLFS:
            name = f"{v}_{c}"
            bp = _params(D, name)
            first = True
            for k in order:
                if k not in bp:
                    continue
                val = bp[k]
                val = f"({val[0]},{val[1]})" if isinstance(val, list) else ("None" if val is None else str(val))
                rng = grid_values(D, name, k)
                rows.append([pretty(name) if first else "", pname[k], rng, val,
                             "Grid search, 5-fold CV, F1" if first else "″"])
                first = False
    R.table(["Model", "Hyperparameter", "Search range", "Selected value", "Method"], rows, "5.1",
            "Hyperparameter settings (search ranges and values selected by cross-validation)",
            widths=[1.05, 1.15, 1.75, 1.0, 1.5], size=9, align=["l", "l", "l", "c", "l"])
    lin = [f"{v}_{c}" for v in VECS for c in ("LogisticRegression", "LinearSVM")]
    claim(all(_params(D, k)["clf__class_weight"] == "balanced" for k in lin), "all LR/SVM chose balanced weights")
    bow_all = [f"BoW_{c}" for c in CLFS]
    bigram_bow = all(_params(D, k)["vec__ngram_range"] == [1, 2] for k in bow_all)
    cw = {}
    for k in lin:
        r_ = D["cvres"][k]
        bal = r_[r_["param_clf__class_weight"] == "balanced"]["mean_test_score"].max()
        non = r_[r_["param_clf__class_weight"] != "balanced"]["mean_test_score"].max()
        cw[k] = (bal, non)
    gains = [cw[k][0] - cw[k][1] for k in lin]
    claim(all(g > 0 for g in gains), "balanced beats unweighted in CV for every linear model")
    R.para("Three patterns stand out in the selected values. First, all four Logistic Regression and linear SVM "
           "pipelines selected `class_weight=\"balanced\"`. The benefit is small, however: the best balanced "
           f"configuration beat the best unweighted one by only {min(gains):.4f}–{max(gains):.4f} in cross-validation "
           f"F1 (for example {f4(cw['TF-IDF_LinearSVM'][0])} against {f4(cw['TF-IDF_LinearSVM'][1])} for "
           f"{pretty('TF-IDF_LinearSVM')}), well below the fold-to-fold standard deviation. Second, "
           + ("all three BoW pipelines selected unigrams plus bigrams with min_df = 2, i.e. they used short phrases "
              "but discarded terms that occur in only one training message. " if bigram_bow and all(
                  _params(D, k)["vec__min_df"] == 2 for k in bow_all) else
              "the n-gram range and min_df varied between pipelines. ")
           + "Third, the TF-IDF pipelines for Logistic Regression and the SVM selected unigrams with sublinear term "
             "frequency. The selected C values differ widely between models; because C interacts with the feature "
             "scale and the loss function, they are not directly comparable.")
    claim(all(_params(D, k)["vec__ngram_range"] == [1, 1] and _params(D, k)["vec__sublinear_tf"]
              for k in ("TF-IDF_LogisticRegression", "TF-IDF_LinearSVM")), "TF-IDF LR/SVM chose unigrams+sublinear")


# ============================================================== Chapter 6
def chapter6(R, D):
    m, cv, e = D["m"], D["cv"], D["eda"]
    M = m["models"]
    names = [f"{v}_{c}" for v in VECS for c in CLFS]
    best = m["best_model"]
    b = M[best]
    second = m["ranking_by_rule"][1]
    R.h1("Chapter 6 – Results and Evaluation")
    R.h2("6.1 Evaluation Metrics")
    R.para("All metrics treat spam as the positive class. A true positive (TP) is a spam message predicted as spam, a "
           "true negative (TN) a ham message predicted as ham, a false positive (FP) a ham message wrongly predicted as "
           "spam, and a false negative (FN) a spam message wrongly predicted as ham. From these counts:")
    R.equation("Accuracy = (TP + TN) / (TP + TN + FP + FN)", "6.1")
    R.equation("Precision = TP / (TP + FP)", "6.2")
    R.equation("Recall = TP / (TP + FN)", "6.3")
    R.equation("F1 = 2 · Precision · Recall / (Precision + Recall)", "6.4")
    R.equation("FPR = FP / (FP + TN)", "6.5")
    R.equation("MCC = (TP·TN − FP·FN) / √[(TP+FP)(TP+FN)(TN+FP)(TN+FN)]", "6.6")
    R.para("Precision is the share of messages flagged as spam that really are spam; recall is the share of spam "
           "messages that are caught; F1 is their harmonic mean and is high only when both are high. Macro-F1 is the "
           "unweighted mean of the F1 scores of the two classes. The false positive rate (FPR) is the share of "
           "legitimate messages wrongly flagged. The Matthews correlation coefficient (MCC) uses all four counts and "
           "ranges from −1 to +1, with 0 for random guessing; it stays informative under class imbalance.")
    R.para("Two threshold-free metrics summarise how well a model *ranks* messages by its score. The ROC-AUC is the "
           "area under the curve of recall (true positive rate) against FPR as the decision threshold varies; it equals "
           "the probability that a randomly chosen spam message receives a higher score than a randomly chosen ham "
           "message. The PR-AUC, computed as average precision (AP), summarises the precision–recall curve:")
    R.equation("AP = Σ_{k} (R_{k} − R_{k−1}) · P_{k}", "6.7")
    R.para("where P_{k} and R_{k} are the precision and recall at the k-th threshold. The scores used for these curves are "
           "the predicted spam probability for Naive Bayes and Logistic Regression and the decision function f(x) for "
           "the linear SVM; because ROC and PR curves depend only on the ordering of the scores, both are valid.")
    R.para(f"**Why these metrics.** With {p2(e['ham_pct'])} ham messages, a trivial model that predicts “ham” for "
           f"every message reaches {p2(e['ham_pct'])} accuracy and an F1 of 0, so accuracy is reported but not used to "
           "choose models. F1 (spam) was used for tuning, and PR-AUC is more informative than ROC-AUC under imbalance "
           "because it focuses on the minority class. The FPR is reported separately because a false positive hides a "
           "genuine message—possibly a bank alert or an OTP—from the user, which costs more than letting one spam "
           "message through.")
    R.h2("6.2 Results")
    R.para(f"All six tuned pipelines were evaluated once on the same held-out test set of {n(m['n_test'])} messages "
           f"({n(m['n_test_ham'])} ham and {n(m['n_test_spam'])} spam). The **best-model rule**, fixed before "
           "evaluation, is: highest test F1 (spam); ties are broken by the lower false positive rate and then by the "
           "higher ROC-AUC. The reason is that a false positive costs more than a false negative. Table 6.1 reports the "
           "test results (Acc. = accuracy, Prec. = precision; best value in each column in bold) and Table 6.2 the "
           "cross-validation results and timings.")
    cols = [("Acc.", "accuracy", max), ("Prec.", "precision", max), ("Recall", "recall", max),
            ("F1", "f1", max), ("Macro-F1", "macro_f1", max), ("ROC-AUC", "roc_auc", max), ("PR-AUC", "pr_auc", max),
            ("MCC", "mcc", max), ("FPR", "fpr", min), ("FP", "fp", min), ("FN", "fn", min)]
    rows, bold = [], set()
    for i, nm in enumerate(names):
        row = [pretty(nm)]
        for j, (_, k, _) in enumerate(cols, 1):
            v = M[nm][k]
            row.append(str(v) if k in ("fp", "fn") else f4(v))
        rows.append(row)
    for j, (_, k, fn_) in enumerate(cols, 1):
        best_v = fn_(round(M[nm][k], 4) for nm in names)
        for i, nm in enumerate(names):
            if round(M[nm][k], 4) == best_v:
                bold.add((i, j))
    R.table(["Model"] + [c[0] for c in cols], rows, "6.1", "Test-set results of the six models (best value per "
            "column in bold)", widths=[0.92] + [0.5] * 11, size=8, bold_cells=bold,
            header_size=7.5)
    rows, bold = [], set()
    t62cols = [("cv", max), ("std", None), ("train", None), ("test", max), ("refit", min), ("inf", min), ("voc", None)]
    vals = {}
    for nm in names:
        vals[nm] = {"cv": cv.loc[nm, "cv_f1_mean"], "std": cv.loc[nm, "cv_f1_std"],
                    "train": cv.loc[nm, "cv_train_f1_mean"], "test": M[nm]["f1"],
                    "refit": cv.loc[nm, "refit_time_s"], "inf": M[nm]["inference_ms_per_1000"],
                    "voc": int(cv.loc[nm, "vocab_size"])}
        rows.append([pretty(nm), f"{f4(vals[nm]['cv'])} ± {f4(vals[nm]['std'])}", f4(vals[nm]["train"]),
                     f4(vals[nm]["test"]), f"{vals[nm]['refit']:.3f}", f"{vals[nm]['inf']:.2f}", n(vals[nm]["voc"])])
    colidx = {"cv": 1, "train": 2, "test": 3, "refit": 4, "inf": 5}
    for key, fn_ in t62cols:
        if fn_ is None:
            continue
        rnd = 3 if key == "refit" else (2 if key == "inf" else 4)
        bv = fn_(round(vals[nm][key], rnd) for nm in names)
        for i, nm in enumerate(names):
            if round(vals[nm][key], rnd) == bv:
                bold.add((i, colidx[key]))
    R.table(["Model", "CV F1 (mean ± std)", "CV train F1", "Test F1", "Refit time (s)",
             "Inference (ms per 1,000 msgs)", "Vocabulary size"], rows, "6.2",
            "Cross-validation results, test F1, training and inference time (best value per column in bold)",
            widths=[0.95, 1.3, 0.8, 0.75, 0.8, 1.05, 0.8], size=9, bold_cells=bold)
    f1s = {nm: M[nm]["f1"] for nm in names}
    lo, hi = min(f1s, key=f1s.get), max(f1s, key=f1s.get)
    acc_lo = min(M[nm]["accuracy"] for nm in names)
    lin = [nm for nm in names if not nm.endswith("MultinomialNB")]
    nb = [nm for nm in names if nm.endswith("MultinomialNB")]
    claim(hi == best, "best model has the highest F1")
    claim(all(b[k] >= max(M[x][k] for x in names) for k in ("accuracy", "recall", "mcc")),
          "best model has highest accuracy, recall and MCC")
    claim(all(M[x][k] > max(M[y][k] for y in nb) for x in lin for k in ("f1", "precision", "roc_auc", "pr_auc"))
          and all(M[x]["fpr"] < min(M[y]["fpr"] for y in nb) for x in lin), "linear beat NB on F1/P/FPR/AUCs")
    claim(all(M[x]["fp"] == min(M[y]["fp"] for y in names) for x in lin), "all LR/SVM share the lowest FP count")
    claim(all(M[x]["f1"] < min(M[y]["f1"] for y in lin) for x in nb), "both NB models below all linear models in F1")
    R.paras(
        f"**{full(best)}** achieved the highest test F1 ({f4(b['f1'])}) and is the best model under the rule. It also "
        f"had the highest accuracy ({f4(b['accuracy'])}), recall ({f4(b['recall'])}) and MCC ({f4(b['mcc'])}). It "
        f"misclassified {b['fp'] + b['fn']} of the {n(m['n_test'])} test messages: {b['fp']} ham messages flagged as "
        f"spam (FPR = {pc(b['fpr'])}) and {b['fn']} spam messages missed. All six models reach at least {f4(acc_lo)} accuracy, "
        "which shows how little accuracy separates them; the F1 scores spread more widely, from "
        f"{f4(f1s[lo])} ({pretty(lo)}) to {f4(f1s[hi])} ({pretty(hi)}).",
        f"The four Logistic Regression and SVM pipelines form a clear top group: each made only "
        f"{min(M[y]['fp'] for y in names)} false positives, and their F1 scores lie between "
        f"{f4(min(f1s[x] for x in lin))} and {f4(max(f1s[x] for x in lin))}. Both Naive Bayes pipelines had lower F1 "
        f"({f4(f1s['BoW_MultinomialNB'])} with BoW and {f4(f1s['TF-IDF_MultinomialNB'])} with TF-IDF) and more false "
        f"positives ({M['BoW_MultinomialNB']['fp']} and {M['TF-IDF_MultinomialNB']['fp']}). The threshold-free metrics "
        "tell the same story: the highest ROC-AUC "
        f"({f4(max(M[x]['roc_auc'] for x in names))}, {pretty(max(names, key=lambda x: M[x]['roc_auc']))}) and the "
        f"highest PR-AUC ({f4(max(M[x]['pr_auc'] for x in names))}, "
        f"{pretty(max(names, key=lambda x: M[x]['pr_auc']))}) both belong to linear discriminative models, although "
        "the differences between the top four are in the third decimal place.",
        "Table 6.2 shows that all models are fast. Re-fitting a tuned pipeline on the full training set took well under "
        f"one second ({min(vals[x]['refit'] for x in names):.3f}–{max(vals[x]['refit'] for x in names):.3f} s), and "
        f"classifying 1,000 messages took {min(vals[x]['inf'] for x in names):.2f}–"
        f"{max(vals[x]['inf'] for x in names):.2f} ms. For the four linear models, training-fold F1 is "
        f"{min(vals[x]['train'] for x in lin):.4f}–{max(vals[x]['train'] for x in lin):.4f}, so the gap of "
        f"{min(vals[x]['train'] - vals[x]['cv'] for x in lin):.4f}–{max(vals[x]['train'] - vals[x]['cv'] for x in lin):.4f} "
        "between training and cross-validation F1 indicates some variance; the learning curve in Appendix B shows "
        "that more training data would still help.")

    R.h2("6.3 Visual Results")
    R.para("**Confusion matrices.** Fig. 6.1 shows the confusion matrix of every model on the test set (rows: actual "
           "class; columns: predicted class; colour: share of the actual class). All models classify almost all ham "
           "messages correctly; the differences lie in the spam row. For every model, missed spam (FN) is at least as "
           f"common as false alarms (FP): for example, {pretty(best)} has {b['fn']} FN and {b['fp']} FP. Both Naive Bayes "
           "models make more false positives than any linear model, and TF-IDF + NB also misses the most spam.")
    claim(all(M[x]["fp"] > max(M[y]["fp"] for y in lin) for x in nb), "NB more FP than any linear model")
    claim(max(names, key=lambda x: M[x]["fn"]) == "TF-IDF_MultinomialNB", "TF-IDF NB has the most FN")
    claim(all(M[x]["fn"] >= M[x]["fp"] for x in names), "FN >= FP for every model")
    R.figure(FIG / "fig_6_1_confusion_matrices.png", "6.1",
             "Confusion matrices of the six models on the test set (rows: BoW, TF-IDF; columns: NB, LR, SVM)", width=5.5)
    R.para("**ROC curves.** Fig. 6.2 plots the ROC curves of all six models. On the full scale (left) all curves hug "
           "the top-left corner, consistent with ROC-AUC values between "
           f"{f4(min(M[x]['roc_auc'] for x in names))} and {f4(max(M[x]['roc_auc'] for x in names))}. The zoomed "
           "panel shows where the models differ: over most of the low false-positive range that matters for a spam "
           "filter, the Logistic Regression and SVM curves reach higher recall than the Naive Bayes curves.")
    R.figure(FIG / "fig_6_2_roc_curves.png", "6.2", "ROC curves of the six models (full range and zoom on FPR ≤ 0.10)",
             width=5.5)
    R.para("**Precision–recall curves.** Fig. 6.3 shows precision against recall. Under class imbalance this view is "
           "more useful than ROC, because the large number of true negatives does not dilute it: the no-skill baseline "
           f"is the spam share ({m['n_test_spam'] / m['n_test']:.3f}), not 0.5. The curves separate at high recall: "
           "the best precision achievable while catching at least 95% of the spam is "
           f"{f4(min(M[x]['precision_at_recall_95'] for x in lin))}–{f4(max(M[x]['precision_at_recall_95'] for x in lin))} "
           "for the four linear models but only "
           f"{f4(max(M[x]['precision_at_recall_95'] for x in nb))} for both Naive Bayes models. The average precision (AP) "
           f"values range from {f4(min(M[x]['pr_auc'] for x in names))} to {f4(max(M[x]['pr_auc'] for x in names))}.")
    claim(min(M[x]["precision_at_recall_95"] for x in lin) > max(M[x]["precision_at_recall_95"] for x in nb),
          "linear models have higher precision at recall >= 0.95 than NB")
    claim(len({round(M[x]["precision_at_recall_95"], 4) for x in nb}) == 1, "both NB models same precision@R95")
    R.figure(FIG / "fig_6_3_pr_curves.png", "6.3", "Precision–recall curves of the six models (full range and zoom on "
             "recall ≥ 0.80)", width=5.5)
    claim(all(M[x]["precision"] >= M[x]["recall"] for x in names), "precision >= recall for all models")
    R.para("**Precision, recall and F1.** Fig. 6.4 compares the three threshold-dependent metrics. Precision is at "
           "least as high as recall for every model, i.e. the models are conservative: when they flag a message as spam "
           "they are almost always right, but they let a few spam messages through. This is the desirable direction of "
           "error for a filter that must not hide genuine messages. The plot uses dots on a zoomed axis because bars "
           "starting at zero would hide differences of one or two hundredths.")
    R.figure(FIG / "fig_6_4_metric_comparison.png", "6.4", "Precision, recall and F1 of the six models on the test set",
             width=5.2)
    diffs = {c: M[f"TF-IDF_{c}"]["f1"] - M[f"BoW_{c}"]["f1"] for c in CLFS}
    R.para("**BoW versus TF-IDF.** Fig. 6.5 places the test F1 of each classifier with BoW and with TF-IDF side by "
           "side. The effect of the representation is small and not consistent in direction: TF-IDF changed F1 by "
           f"{diffs['MultinomialNB']:+.4f} for Naive Bayes, {diffs['LogisticRegression']:+.4f} for Logistic Regression "
           f"and {diffs['LinearSVM']:+.4f} for the SVM. In messages, TF-IDF changed the false positives / missed spam "
           "of Logistic Regression from "
           f"{M['BoW_LogisticRegression']['fp']}/{M['BoW_LogisticRegression']['fn']} to "
           f"{M['TF-IDF_LogisticRegression']['fp']}/{M['TF-IDF_LogisticRegression']['fn']}, of the SVM from "
           f"{M['BoW_LinearSVM']['fp']}/{M['BoW_LinearSVM']['fn']} to {M['TF-IDF_LinearSVM']['fp']}/{M['TF-IDF_LinearSVM']['fn']}, "
           f"and of Naive Bayes from {M['BoW_MultinomialNB']['fp']}/{M['BoW_MultinomialNB']['fn']} to "
           f"{M['TF-IDF_MultinomialNB']['fp']}/{M['TF-IDF_MultinomialNB']['fn']}—for Naive Bayes, fewer false alarms "
           "were traded for more missed spam.")
    R.figure(FIG / "fig_6_5_bow_vs_tfidf.png", "6.5", "Test F1 of each classifier with BoW and TF-IDF features",
             width=4.8)
    R.para("**Cross-validation versus test.** Fig. 6.6 compares the cross-validation F1 (mean ± one standard deviation "
           "over the five folds) with the test F1 of each model. The fold-to-fold standard deviation is about "
           f"{cv['cv_f1_std'].min():.2f}–{cv['cv_f1_std'].max():.2f}, so differences of a few thousandths between models "
           "are well within the noise. Every test score lies within about one standard deviation of its cross-validation "
           "mean or above it, so there is no sign that tuning over-fitted the validation folds. The test scores of the "
           "linear models are somewhat higher than their cross-validation means, which is expected because the final "
           "models are trained on all of the training data rather than four fifths of it.")
    claim(all(M[x]["f1"] >= cv.loc[x, "cv_f1_mean"] - 1.05 * cv.loc[x, "cv_f1_std"] for x in names),
          "test F1 within ~1 std of CV mean or above")
    claim(all(M[x]["f1"] > cv.loc[x, "cv_f1_mean"] for x in lin), "linear models' test F1 above CV mean")
    R.figure(FIG / "fig_6_6_cv_vs_test.png", "6.6", "Cross-validation F1 (mean ± std) and test F1 of each model",
             width=5.0)
    it = D["interp"]
    t66 = D["t66"]
    lr_spam = list(t66["LR spam token"])
    nb_spam = list(t66["NB spam token"])
    nb_ham = list(t66["NB ham token"])
    claim(lr_spam[0] == "phonetoken" and nb_spam[0] == "phonetoken", "phonetoken is the top spam feature for LR and NB")
    R.para(f"**Most informative terms.** Fig. 6.7 shows the 20 terms that push a message most strongly towards spam "
           f"and towards ham, (a) for the Logistic Regression model {pretty(it['lr_model'])} (coefficients) and (b) for "
           f"the Naive Bayes model {pretty(it['nb_model'])} (log-ratio of the class-conditional word probabilities). In "
           "both models the strongest spam indicator is `phonetoken`, the normalisation token for long digit runs, "
           "which confirms the value of the normalisation step. `urltoken` is also among the top spam terms of both "
           f"models. The Naive Bayes list is easy to read: “{'”, “'.join([t for t in nb_spam if ' ' not in t and 'token' not in t][:7])}” "
           "are typical spam vocabulary, and bigrams such as “numtoken cash” and “won moneytoken” capture prize "
           "templates. The Logistic Regression list also contains some rare words tied to particular spam campaigns "
           f"(for example “{'”, “'.join([t for t in lr_spam[1:6] if 'token' not in t][:3])}”), because the selected weak "
           f"regularisation (C = {_params(D, it['lr_model'])['clf__C']}) lets the model give large weights to terms that "
           "appear only in spam. On the ham side, both models rely on conversational words and SMS shorthand "
           f"(Naive Bayes: “{'”, “'.join(nb_ham[:6])}”; Logistic Regression: "
           f"“{'”, “'.join([t for t in t66['LR ham token'] if ' ' not in t][:6])}”).")
    R.figure(FIG / "fig_6_7_top_features.png", "6.7", "Top 20 spam- and ham-indicative terms: (a) Logistic Regression "
             "coefficients, (b) Naive Bayes log-probability ratios", width=5.1)
    R.para(f"**Best model.** Fig. 6.8 enlarges the confusion matrix of the best model, {full(best)}, with counts and "
           f"row percentages. Of the {n(m['n_test_ham'])} ham messages, {b['tn']} ({pc(b['tn'] / m['n_test_ham'])}) are "
           f"classified correctly and {b['fp']} ({pc(b['fpr'])}) are wrongly flagged as spam. Of the "
           f"{n(m['n_test_spam'])} spam messages, {b['tp']} ({pc(b['recall'])}) are caught and {b['fn']} "
           f"({pc(b['fn'] / m['n_test_spam'])}) are missed.")
    R.figure(FIG / "fig_6_8_best_confusion_matrix.png", "6.8", f"Confusion matrix of the best model ({pretty(best)}) "
             "with counts and row percentages", width=3.2)

    R.h2("6.4 Comparison and Discussion")
    R.h3("6.4.1 BoW versus TF-IDF")
    better = [c for c in CLFS if diffs[c] > 0]
    worse = [c for c in CLFS if diffs[c] < 0]
    R.para("Across the three classifiers, neither representation is consistently better on the test set: TF-IDF "
           f"helped {', '.join(CLF_FULL[c] for c in better) or 'none'} and BoW was better for "
           f"{' and '.join(CLF_FULL[c] for c in worse) or 'none'}. In cross-validation, by contrast, the TF-IDF version "
           "of every classifier scored slightly higher than the BoW version (Table 6.2). Both observations point the "
           "same way: once the n-gram range, minimum document frequency and regularisation are tuned for each "
           "representation, the two give very similar results on this dataset. A plausible reason is that SMS messages "
           "are short—after preprocessing, a spam message has on average "
           f"{e['mean_tokens_per_msg']['spam']:.2f} tokens and a ham message {e['mean_tokens_per_msg']['ham']:.2f}—so "
           "most terms occur once per message and term-frequency weighting changes little; the IDF factor and "
           "length normalisation are the main differences, and the tuned classifiers can compensate for them.")
    claim(all(cv.loc[f"TF-IDF_{c}", "cv_f1_mean"] > cv.loc[f"BoW_{c}", "cv_f1_mean"] for c in CLFS),
          "TF-IDF > BoW in CV for every classifier")
    R.h3("6.4.2 Naive Bayes versus Logistic Regression versus SVM")
    R.para("The classifier matters more than the representation. Both discriminative linear models clearly beat "
           "Multinomial Naive Bayes on the test set in F1, precision, FPR, ROC-AUC and PR-AUC. Naive Bayes estimates "
           "each word’s evidence separately and adds it up, so correlated spam words (for example “claim”, “prize” and "
           "“won”, which tend to occur together) are counted several times, and the resulting scores are pushed to "
           "extremes. Logistic Regression and the SVM fit all weights jointly to separate the classes, which handles "
           "such correlations and allows a few strong indicators such as `phonetoken` to dominate. Between Logistic "
           "Regression and the SVM there is no meaningful difference: with TF-IDF they made exactly the same numbers of "
           f"errors ({M['TF-IDF_LogisticRegression']['fp']} FP and {M['TF-IDF_LogisticRegression']['fn']} FN each), and "
           f"with BoW, Logistic Regression missed {M['BoW_LinearSVM']['fn'] - M['BoW_LogisticRegression']['fn']} fewer "
           "spam messages. Naive Bayes nevertheless remains a reasonable baseline: it is the simplest model and still "
           f"reached F1 above {min(f1s[x] for x in nb):.2f}.")
    claim(M["TF-IDF_LogisticRegression"]["fp"] == M["TF-IDF_LinearSVM"]["fp"] and
          M["TF-IDF_LogisticRegression"]["fn"] == M["TF-IDF_LinearSVM"]["fn"], "TF-IDF LR and SVM same error counts")
    R.h3("6.4.3 Precision–Recall Trade-off and Class Weighting")
    R.para("All models use the default threshold (probability 0.5, or decision value 0 for the SVM), at which the "
           f"best model has precision {f4(b['precision'])} and recall {f4(b['recall'])}. Fig. 6.3 shows that a lower "
           "threshold would catch more spam at the cost of more false positives; a deployed filter should set the "
           "threshold for a target FPR on validation data. The balanced class weights chosen by all four linear models "
           "shift the boundary towards higher recall for the minority spam class.")
    R.h3("6.4.4 Agreement between Cross-Validation and Test Results")
    rank_cv = m["ranking_by_cv_f1"]
    R.para(f"The ranking of the models differs between cross-validation and the test set. In cross-validation "
           f"{pretty(rank_cv[0])} was best and {pretty(rank_cv[-1])} last; on the test set {pretty(best)} was best. This "
           "is not a contradiction. The cross-validation means of all six models lie within "
           f"{cv['cv_f1_mean'].max() - cv['cv_f1_mean'].min():.4f} of each other—less than one fold-to-fold standard "
           f"deviation—and the test set contains only {m['n_test_spam']} spam messages, so a single spam message changes "
           f"recall by {100 / m['n_test_spam']:.2f} percentage points. The rankings are therefore not stable at this "
           "level of detail, and the reliable conclusion is the grouping: linear models above Naive Bayes, with the "
           "representation playing a minor role. Choosing the best model by test F1 also makes its score slightly "
           "optimistic; the cross-validation figures in Table 6.2 are less biased, although they too are the best of "
           "each grid (and the TF-IDF grids were twice as large because of the extra sublinear_tf option).")
    claim(cv["cv_f1_mean"].max() - cv["cv_f1_mean"].min() < cv["cv_f1_std"].min(), "CV means within one std")
    R.h3("6.4.5 Error Analysis")
    t63 = D["t63"]
    fn_rows = t63[t63.error_type.str.startswith("FN")].head(5)
    fp_rows = t63[t63.error_type.str.startswith("FP")].head(5)
    R.para(f"Table 6.3 lists the misclassified test messages of the best model: up to five false negatives "
           f"({len(fn_rows)} of {m['n_fn_best']} shown) and up to five false positives ({len(fp_rows)} of "
           f"{m['n_fp_best']} shown), with the predicted spam probability.")

    def short(s, k=150):
        s = " ".join(str(s).split())
        return s if len(s) <= k else s[: k - 1].rstrip() + "…"
    rows = [[("FN" if r.error_type.startswith("FN") else "FP"), short(r.text), f"{float(r.score):.4f}"]
            for r in pd.concat([fn_rows, fp_rows]).itertuples()]
    R.table(["Error", "Message", "P(spam)"], rows, "6.3", f"Misclassified test messages of the best model "
            f"({pretty(best)})", widths=[0.55, 5.1, 0.75], size=9, align=["c", "l", "c"])
    texts = " ".join(t63.text)
    claim("thesmszone.com" in texts and "AGE and GENDER" in texts and "SIB never asks" in texts,
          "error-analysis examples present")
    R.paras(
        "The false negatives share a pattern: they contain few of the cues the model has learned. Several are written "
        "in a conversational, first-person style (“Hi this is Amy…”, “dating: i have had two of these…”) and contain "
        "no phone number, price or prize word, so they look like ham. One message advertises a website written without "
        "“http” or “www” (“thesmszone.com”), which the URL pattern does not recognise, so no `urltoken` is produced. "
        "Another (“Welcome! Please reply with your AGE and GENDER…”) is short and its only number is a small one, which "
        "becomes `numtoken`—a token that is also common in ham. These messages are hard even for a human reader "
        "without context.",
        "The false positives are legitimate messages that use spam-like vocabulary. One is a bank security notice "
        "(“…SIB never asks for sensitive information like Passwords, ATM/SMS PIN…”) and the other a contact-sharing "
        "notification that mentions “FREE SMS”, a website and an instruction to “sms NO”. Both are the kind of "
        "automated service message that is hardest to separate from spam using words alone; sender information, "
        "which this study excludes, would help here.")
    mc = m["mcnemar_top2"]
    R.h3("6.4.6 Statistical Comparison of the Two Best Models")
    R.para(f"An exact McNemar test compared the test predictions of the two best models, {pretty(mc['model_a'])} and "
           f"{pretty(mc['model_b'])}. The test uses only the messages on which the models disagree: "
           f"{mc['a_correct_b_wrong']} messages were classified correctly by {pretty(mc['model_a'])} only and "
           f"{mc['a_wrong_b_correct']} by {pretty(mc['model_b'])} only. The two-sided p-value is "
           f"{f4(mc['p_value_exact'])}, so there is no evidence that one model is more accurate than the other on this "
           "test set. The difference between the top models should therefore not be over-interpreted.")
    claim(mc["p_value_exact"] > 0.05, "McNemar not significant")
    R.h3("6.4.7 Why the Best Model Wins")
    R.para(f"{full(best)} wins under the rule because it combines the lowest number of false positives "
           f"({b['fp']}, shared with the other linear models) with the fewest missed spam messages ({b['fn']}, against "
           f"{M[second]['fn']} for {pretty(second)}); its bigrams with min_df = 2 capture short spam-template phrases "
           "while ignoring one-off terms. "
           "However, its margin over the next three models is one or two test messages, the McNemar test finds no "
           f"significant difference, and it ranked {rank_cv.index(best) + 1} of 6 in cross-validation. Logistic "
           "Regression and the linear SVM, with either representation, are therefore equally good choices here.")

    R.h2("6.5 Sample Predictions")
    t64 = D["t64"]
    R.para("Table 6.4 shows five representative test messages with the prediction of the best model: two typical ham "
           "messages, two typical spam messages and the misclassified message that was closest to the decision "
           "threshold.")
    rows = [[r.case, short(r.text, 120), r.actual, r.predicted, f"{float(r.score):.4f}"] for r in t64.itertuples()]
    R.table(["Case", "Message (test set)", "Actual", "Predicted", "P(spam)"], rows, "6.4",
            "Sample predictions of the best model on test-set messages", widths=[1.05, 3.6, 0.6, 0.7, 0.65], size=9,
            align=["l", "l", "c", "c", "c"])
    t65 = D["t65"]
    R.para("Table 6.5 shows predictions for five new messages written by the team. **These messages are not from the "
           "dataset**; they test how the model behaves on realistic present-day messages.")
    rows = [[r.description, r.text, r.predicted, f"{float(r.score):.4f}"] for r in t65.itertuples()]
    R.table(["Type", "New message (not from the dataset)", "Predicted", "P(spam)"], rows, "6.5",
            "Predictions of the best model on new messages written by the team", widths=[1.35, 3.75, 0.7, 0.65],
            size=9, align=["l", "l", "c", "c"])
    pred = dict(zip(t65.description, zip(t65.predicted, t65.score)))
    otp = pred["One-time password notice"]
    dlv = pred["Delivery notice with URL"]
    claim(pred["Prize / premium-rate lure"][0] == "spam" and pred["Casual personal message"][0] == "ham"
          and pred["Casual message containing 'free'"][0] == "ham", "obvious custom cases correct")
    claim(otp[0] == "spam" and dlv[0] == "spam", "OTP and delivery notices predicted spam")
    R.para(f"The prize message is flagged with P(spam) = {float(pred['Prize / premium-rate lure'][1]):.4f}, and the two "
           "casual messages—including one that contains the word “free”—are correctly treated as ham, which shows that "
           "the model does not react to a single keyword. The two automated messages reveal a limitation: the OTP notice "
           f"(P(spam) = {float(otp[1]):.4f}) and the delivery notice with a tracking link (P(spam) = {float(dlv[1]):.4f}) "
           "are both labelled spam. In the dataset, almost every message with a code number or a URL is spam "
           f"(only {p2(e['pct_with_url']['ham'])} of ham messages contain a URL), and legitimate transactional messages "
           "of this kind are rare in a collection assembled around 2011. A filter for today’s SMS traffic would need "
           "training data that include such messages (Chapter 7).")
    if D["cli"]:
        R.para("The best model can also be used from the command line. The following output was produced by "
               "`src/predict.py` during the run that generated this report:")
        R.code(D["cli"].strip())


# ============================================================== Chapter 7
def chapter7(R, D):
    m, e, info = D["m"], D["eda"], D["info"]
    M = m["models"]
    best = m["best_model"]
    b = M[best]
    tb1 = D["tb1"].set_index("Variant")
    t66 = D["t66"]
    R.h1("Chapter 7 – Conclusion and Future Enhancements")
    R.h2("7.1 Conclusion")
    R.paras(
        "This case study built and compared six SMS spam classifiers on the UCI SMS Spam Collection: Multinomial Naive "
        "Bayes, Logistic Regression and a linear SVM, each with Bag of Words and TF-IDF features. After removing "
        f"{n(e['n_duplicates_removed'])} duplicate messages, the {n(e['n_clean'])} remaining messages were split into "
        "stratified training and test sets; every pipeline was tuned with grid search and 5-fold cross-validation on "
        "the training set and evaluated once on the same test set.",
        f"Under the predefined rule, the best model is **{full(best)}**, with test F1 = {f4(b['f1'])}, precision = "
        f"{f4(b['precision'])}, recall = {f4(b['recall'])}, ROC-AUC = {f4(b['roc_auc'])}, PR-AUC = {f4(b['pr_auc'])} "
        f"and a false positive rate of {pc(b['fpr'])} ({b['fp']} of {n(m['n_test_ham'])} ham messages). The main "
        "findings are that the two linear discriminative classifiers outperform Naive Bayes, that BoW and TF-IDF give "
        "very similar results once tuned, and that normalising numbers, URLs and currency symbols is the most useful "
        f"preprocessing step (removing it lowered cross-validation F1 from {f4(tb1.loc['(a)', 'CV F1 mean'])} to "
        f"{f4(tb1.loc['(c)', 'CV F1 mean'])}; Appendix B). The four best models differ by at most "
        f"{max(M[k]['f1'] for k in M if not k.endswith('NB')) - min(M[k]['f1'] for k in M if not k.endswith('NB')):.4f} "
        "in test F1, and the two best are not significantly different on this test set (McNemar p = "
        f"{f4(m['mcnemar_top2']['p_value_exact'])}).")
    claim(tb1["CV F1 mean"].idxmin() in ("(c)", "(e)"), "removing normalisation is among the largest drops")
    R.para("The objectives set in Section 1.3 were met as follows:")
    R.bullets([
        "**EDA** — met (Chapter 3); every observation informed a design decision.",
        "**Preprocessing and representation** — met: normalisation, tokenization, stop-word removal, BoW and TF-IDF "
        "(Chapter 4).",
        "**Training and tuning** — met: three classifiers on both representations, tuned with 5-fold stratified "
        "cross-validation (Chapter 5).",
        "**Comparison** — met: six models compared on one held-out test set with imbalance-aware metrics and a "
        "predefined selection rule (Chapter 6).",
        "**Limitations and improvements** — met: error analysis (Section 6.4.5) and Sections 7.2–7.3."],
        numbered=True)
    R.h2("7.2 Limitations")
    comp = info["composition"]
    nb_ham = list(t66["NB ham token"])
    claim("lor" in nb_ham[:5], "'lor' among top NB ham tokens")
    R.bullets([
        "**One dataset, one period, one language.** All results come from a single public collection assembled around 2011. Its "
        f"spam comes mainly from a UK forum ({comp[0]['spam']} messages from Grumbletext) and a public corpus, and most "
        f"of its ham ({n(comp[1]['ham'])} messages) from Singaporean university students. The model may partly learn "
        "these sources rather than spam itself—for example, Singapore English particles such as “lor” are among the "
        "strongest ham indicators in Fig. 6.7. All messages are in English; other languages and code-mixed text are "
        "not covered.",
        f"**Small spam class.** The deduplicated data contain only {n(e['clean_class_counts']['spam'])} spam messages "
        f"and the test set {m['n_test_spam']}, so one message changes test recall by {100 / m['n_test_spam']:.2f} "
        "percentage points and small differences between models are not reliable.",
        "**Concept drift.** Spam tactics change over time (new lures, link shorteners, OTP fraud). The new-message test "
        "in Table 6.5 shows that legitimate OTP and delivery messages, which are common today, can be misclassified.",
        "**No sender or metadata features.** Sender number, time and sending frequency are not used, although they "
        "would help with automated service messages such as those in the error analysis.",
        "**Bag-of-words assumptions.** BoW and TF-IDF ignore word order beyond bigrams and do not capture meaning or "
        "context; deliberately obfuscated words (for example “fr33”) become unknown tokens.",
        "**Comparability.** Removing duplicates makes the scores more honest but less directly comparable with studies "
        "that evaluated on the raw collection.",
        "**Near-duplicates remain.** Only exact duplicates were removed. After cleaning, "
        f"{m['test_cleaned_text_in_train']['spam']} of the {m['n_test_spam']} test spam messages and "
        f"{m['test_cleaned_text_in_train']['ham']} test ham messages have the same normalised text as a training "
        "message (typically one spam template sent with different numbers), which makes the test somewhat easier."])
    R.h2("7.3 Future Enhancements")
    R.bullets([
        "**Local and multilingual data:** collect or use SMS data from India, including multilingual and code-mixed "
        "messages (for example Hinglish or Kannada–English), and recent transactional messages such as OTPs and "
        "delivery notices.",
        "**Character n-grams** to handle misspelled and obfuscated words, and a broader URL pattern that also "
        "recognises bare domain names.",
        "**Transformer models** such as DistilBERT, compared with the linear models in both accuracy and computational "
        "cost.",
        "**Threshold tuning** to a target false positive rate, **probability calibration**, and **explainability** of "
        "individual predictions with LIME or SHAP.",
        "**Periodic retraining** on newly labelled messages to follow concept drift, and a simple web or mobile "
        "demonstration using the saved pipeline."])


# ============================================================== references & appendices
def references(R):
    R.h1("References")
    missing = [k for k in REFS if k not in _order]
    claim(not missing, f"every reference is cited (missing: {missing})")
    for i, k in enumerate(_order, 1):
        p = R.doc.add_paragraph(style="Reference")
        pf = p.paragraph_format
        pf.left_indent, pf.first_line_indent = Inches(0.45), Inches(-0.45)
        pf.tab_stops.add_tab_stop(Inches(0.45))
        p.add_run(f"[{i}]\t")
        R.runs(p, REFS[k])


def appendix_a(R, D):
    R.h1("Appendix A – Source Code and Reproducibility")
    R.para(f"**Repository:** {REPO}")
    R.para("The repository contains all code, the executed notebook, the generated outputs and this report’s "
           "generator. The raw dataset is not stored in the repository; it is downloaded automatically from the UCI "
           "repository. The folder structure is:")
    R.code("""sms-spam-ham-classifier/
├── README.md, requirements.txt, run_all.py, DECISIONS.md
├── data/raw/              SMSSpamCollection (downloaded, not committed)
├── data/processed/        sms_clean.csv
├── src/                   config, data, preprocess, eda, train, evaluate,
│                          interpret, ablation, predict, flowchart
├── notebooks/             SMS_Spam_Detection.ipynb (executed)
├── outputs/figures/       all figures (PNG, 300 dpi)
├── outputs/tables/        one CSV per report table
├── outputs/models/        six fitted pipelines (joblib)
├── outputs/results/       metrics.json, cv_summary.csv, env.json, versions.txt, ...
└── report/                build_report.py, SMS_Spam_Report.docx, SMS_Spam_Report.pdf""")
    R.para("**Installation and data.** With Python 3.10 or later:")
    R.code("""git clone https://github.com/Mustaqeem-Rafi/sms-spam-ham-classifier.git
cd sms-spam-ham-classifier
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt""")
    R.para("The first run downloads `sms+spam+collection.zip` from the UCI repository and extracts "
           "`SMSSpamCollection` into `data/raw/`. Without internet access, download the zip manually from "
           "https://archive.ics.uci.edu/dataset/228/sms+spam+collection and place `SMSSpamCollection` in `data/raw/`.")
    R.para("**Reproducing all results.** One command runs every stage in order—data preparation, EDA, training, "
           "evaluation, interpretation, ablation, notebook execution and report generation (PDF export requires "
           "LibreOffice):")
    R.code("python run_all.py              # or: python run_all.py --no-notebook --no-pdf")
    rt = D["runtime"].get("run_all_seconds_before_report")
    R.para("All random operations use the fixed seed `random_state = 42`, so the data split, cross-validation folds "
           "and model results are identical on every run; only the measured timings vary between machines."
           + (f" On the hardware listed in Section 5.3, the full run up to report generation took about "
              + (f"{rt:.0f} seconds." if rt < 120 else f"{rt / 60:.1f} minutes.") if rt else ""))
    R.para("**Classifying new messages** with the saved best model:")
    R.code('python -m src.predict "Congratulations! You won a prize, call 09061234567 now"')
    R.para("No API keys, passwords or private data are used anywhere in the project.")


def appendix_b(R, D):
    m = D["m"]
    best = m["best_model"]
    tb1, tb3 = D["tb1"], D["tb3"]
    R.h1("Appendix B – Additional Results")
    R.h2("B.1 Preprocessing Ablation")
    a = tb1.set_index("Variant")
    R.para(f"To measure the contribution of each preprocessing step, the best pipeline ({full(best)}) was re-trained "
           "with its tuned hyperparameters while one step was changed at a time: (a) the full preprocessing of the main "
           "pipeline; (b) without stop-word removal; (c) without the normalisation of numbers, URLs, e-mail addresses "
           "and currency symbols; (d) with Porter stemming added; and (e) raw text with only lowercasing (default "
           "scikit-learn tokenization, no stop-word removal, no normalisation). Each variant was scored with 5-fold "
           "cross-validation on the training set and on the test set (Table B.1, Fig. B.1).")
    rows = [[r.Variant, r.Description, f"{f4(r._3)} ± {f4(r._4)}", f4(r._5), n(r.Vocabulary)] for r in tb1.itertuples()]
    R.table(["Variant", "Description", "CV F1 (mean ± std)", "Test F1", "Vocabulary"], rows, "B.1",
            f"Preprocessing ablation for {pretty(best)}", widths=[0.65, 2.65, 1.35, 0.8, 0.9], size=9.5,
            align=["c", "l", "c", "c", "c"])
    claim(a.loc["(a)", "CV F1 mean"] > a.loc["(b)", "CV F1 mean"] and a.loc["(a)", "Test F1"] > a.loc["(b)", "Test F1"],
          "stop-word removal helps")
    claim(abs(a.loc["(d)", "CV F1 mean"] - a.loc["(a)", "CV F1 mean"]) < 0.005, "stemming changes CV F1 < 0.005")
    claim(a.loc["(e)", "Test F1"] == a["Test F1"].min(), "raw text worst on test")
    R.paras(
        f"Removing the normalisation tokens caused the largest loss in cross-validation F1 among the single-step "
        f"changes (from {f4(a.loc['(a)', 'CV F1 mean'])} to {f4(a.loc['(c)', 'CV F1 mean'])}), which confirms the EDA "
        "finding that numbers, prices and URLs are strong spam signals only when they are grouped into shared tokens. "
        "Removing stop-word filtering also lowered F1 (to "
        f"{f4(a.loc['(b)', 'CV F1 mean'])} in cross-validation and {f4(a.loc['(b)', 'Test F1'])} on the test set) and "
        f"nearly doubled the vocabulary, even though the stop list contains a few spam-related words such as “call”. "
        f"Porter stemming made almost no difference ({f4(a.loc['(d)', 'CV F1 mean'])} against "
        f"{f4(a.loc['(a)', 'CV F1 mean'])} in cross-validation), so keeping readable, unstemmed tokens costs nothing. "
        f"Raw text with only lowercasing gave the lowest test F1 ({f4(a.loc['(e)', 'Test F1'])}), so the full "
        "preprocessing pipeline is clearly beneficial.")
    R.figure(FIG / "fig_b_1_ablation.png", "B.1", "Preprocessing ablation: cross-validation and test F1 of each variant",
             width=5.5)
    R.h2("B.2 Learning Curve")
    first, last = tb3.iloc[0], tb3.iloc[-1]
    R.para(f"Fig. B.2 shows the learning curve of {full(best)}: training and cross-validation F1 as the number of "
           "training messages grows. Cross-validation F1 rises from "
           f"{f4(first['CV F1 mean'])} with {n(first['Training messages'])} messages to {f4(last['CV F1 mean'])} with "
           f"{n(last['Training messages'])} messages, while training F1 stays close to 1. The curve is still rising "
           "at the right edge, so more labelled data—particularly more spam—would probably improve the model further.")
    claim(last["CV F1 mean"] > first["CV F1 mean"], "learning curve rises")
    R.figure(FIG / "fig_b_2_learning_curve.png", "B.2", f"Learning curve of {pretty(best)}", width=5.0)
    R.h2("B.3 Top-10 Cross-Validation Configurations per Model")
    R.para("Table B.2 lists the ten best hyperparameter configurations of each pipeline in the grid search, ranked by "
           "mean cross-validation F1. Within each pipeline the top configurations are separated by only a few "
           "thousandths, so the selected values are one of several nearly equivalent choices.")
    tb2 = D["tb2"]
    rows = [[r.Model, r.Rank, r.Parameters.replace("; ", ", "), f"{f4(r._4)} ± {f4(r._5)}"] for r in tb2.itertuples()]
    R.table(["Model", "Rank", "Parameters", "CV F1 (mean ± std)"], rows, "B.2",
            "Ten best grid-search configurations of each model (5-fold CV F1)", widths=[0.95, 0.5, 3.6, 1.35], size=8,
            align=["l", "c", "l", "c"])
    R.h2("B.4 Classification Reports")
    R.para("The per-class precision, recall and F1 of each model on the test set, as produced by scikit-learn’s "
           "`classification_report`, are reproduced below.")
    for v in VECS:
        for c in CLFS:
            R.code(D["clf_reports"][f"{v}_{c}"].strip())


# ============================================================== build
def build():
    _order.clear()
    D = load()
    R = ReportDoc()
    cover(R)
    R.new_section("lowerRoman")
    certificate(R)
    declaration(R)
    acknowledgement(R)
    abstract(R, D)
    front_lists(R)
    abbreviations(R)
    R.new_section("decimal")
    chapter1(R, D)
    chapter2(R, D)
    chapter3(R, D)
    chapter4(R, D)
    chapter5(R, D)
    chapter6(R, D)
    chapter7(R, D)
    references(R)
    appendix_a(R, D)
    appendix_b(R, D)
    R.save(DOCX)
    print(f"wrote {DOCX.relative_to(ROOT)}")
    return DOCX


def to_pdf():
    from lo_export import export, find_soffice
    if find_soffice() is None:
        print("LibreOffice not found: open the .docx in MS Word, press Ctrl+A then F9 (update entire tables), "
              "and save as PDF.")
        return None
    export(DOCX, PDF)
    print(f"wrote {PDF.relative_to(ROOT)}")
    return PDF


if __name__ == "__main__":
    build()
    if "--no-pdf" not in sys.argv:
        to_pdf()
