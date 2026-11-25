# Literature and dataset verification (25 November 2026)

Checked via Crossref, OpenAlex, Semantic Scholar, AAAI, Springer, ScienceDirect, JMLR, JOSS, nltk.org.
Descriptions below are from abstracts that were actually read. Numeric results are not quoted from any paper.

| # | Status | Corrected citation | DOI / URL | What the paper does (from abstract) |
|---|---|---|---|---|
| 1 | Verified | T. A. Almeida, J. M. Gómez Hidalgo, A. Yamakami, "Contributions to the study of SMS spam filtering: New collection and results," Proc. 11th ACM Symp. Document Engineering (DocEng '11), 2011, pp. 259–262. | 10.1145/2034691.2034742 | New real, public, non-encoded SMS spam collection (largest known to authors); compares established ML methods; concludes SVM outperforms the others and is a good baseline. |
| 2 | Corrected (Spanish double surnames) | J. M. Gómez Hidalgo, G. Cajigas Bringas, E. Puertas Sánz, F. Carrero García, "Content based SMS spam filtering," Proc. 2006 ACM Symp. Document Engineering (DocEng '06), 2006, pp. 107–114. | 10.1145/1166160.1166191 | Asks whether Bayesian e-mail spam filtering transfers to SMS; builds English and Spanish SMS spam test collections; tests several representations and ML algorithms. |
| 3 | Verified | S. J. Delany, M. Buckley, D. Greene, "SMS spam filtering: Methods and data," Expert Systems with Applications, vol. 39, no. 10, pp. 9899–9908, Aug. 2012. | 10.1016/j.eswa.2012.02.053 | Motivates content-based SMS spam filtering, how it differs from e-mail; reviews research; analyses recent SMS spam; releases a dataset. |
| 4 | Corrected (pages, report no.) | M. Sahami, S. Dumais, D. Heckerman, E. Horvitz, "A Bayesian approach to filtering junk e-mail," in Learning for Text Categorization: Papers from the 1998 AAAI Workshop, AAAI Tech. Rep. WS-98-05, 1998, pp. 55–62. | https://aaai.org/papers/055-ws98-05-009/ | Junk-mail filtering as decision-theoretic problem; probabilistic learning with differential misclassification costs; domain-specific features added to raw text give much more accurate filters. |
| 5 | Verified | A. McCallum, K. Nigam, "A comparison of event models for Naive Bayes text classification," AAAI-98 Workshop on Learning for Text Categorization, AAAI Tech. Rep. WS-98-05, 1998, pp. 41–48. | https://aaai.org/papers/041-ws98-05-007/ | Compares multivariate Bernoulli vs multinomial NB on five corpora; multinomial usually better at larger vocabulary sizes. |
| 6 | Verified | T. Joachims, "Text categorization with support vector machines: Learning with many relevant features," in Machine Learning: ECML-98, LNCS vol. 1398, Springer, 1998, pp. 137–142. | 10.1007/BFb0026683 | Analyses properties of text data; explains why SVMs suit text categorisation; SVMs robust, no manual parameter tuning. |
| 7 | Verified | G. Salton, C. Buckley, "Term-weighting approaches in automatic text retrieval," Information Processing & Management, vol. 24, no. 5, pp. 513–523, 1988. | 10.1016/0306-4573(88)90021-0 | Summarises term-weighting insights; weighted single-term indexing beats more elaborate representations; baseline weighting models. |
| 8 | Verified | F. Pedregosa et al., "Scikit-learn: Machine learning in Python," J. Machine Learning Research, vol. 12, pp. 2825–2830, 2011. | https://jmlr.org/papers/v12/pedregosa11a.html | Python ML module; ease of use, performance, documentation, consistent API. |
| 9 | Verified | C. Cortes, V. Vapnik, "Support-vector networks," Machine Learning, vol. 20, no. 3, pp. 273–297, 1995. | 10.1007/BF00994018 | Maps inputs to high-dimensional feature space, linear decision surface; soft margin for non-separable data. |
| 10 | Verified | J. D. Hunter, "Matplotlib: A 2D graphics environment," Computing in Science & Engineering, vol. 9, no. 3, pp. 90–95, 2007. | 10.1109/MCSE.2007.55 | 2D graphics package for Python. |
| 11a | Verified | W. McKinney, "Data structures for statistical computing in Python," Proc. 9th Python in Science Conf. (SciPy 2010), 2010, pp. 56–61. | 10.25080/Majora-92bf1922-00a | pandas data structures. |
| 11b | Verified | M. L. Waskom, "seaborn: statistical data visualization," J. Open Source Software, vol. 6, no. 60, p. 3021, 2021. | 10.21105/joss.03021 | High-level statistical plotting on matplotlib. |
| 12 | Verified | S. Bird, E. Klein, E. Loper, Natural Language Processing with Python, O'Reilly Media, 2009. Porter stemmer docs: https://www.nltk.org/api/nltk.stem.porter.html (HTTP 200). Porter 1980: M. F. Porter, "An algorithm for suffix stripping," Program, vol. 14, no. 3, pp. 130–137, 1980. | — | NLTK book; Porter stemmer module page. |

## UCI page (verbatim facts)
- DOI: 10.24432/C5CC84
- Citation: Almeida, T. & Hidalgo, J. (2011). SMS Spam Collection [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CC84.
- Licence: Creative Commons Attribution 4.0 International (CC BY 4.0).
- Creators: Tiago Almeida, José Hidalgo. Donated on 6/21/2012. Instances: 5574.
- Composition: 425 spam from Grumbletext (UK forum); 3,375 ham randomly chosen from NUS SMS Corpus (National University of Singapore, mostly students); 450 ham from Caroline Tagg's PhD thesis (University of Birmingham); SMS Spam Corpus v.0.1 Big: 1,002 ham + 322 spam.
- Totals (readme in the zip, confirmed by counting the file): 4,827 ham (86.6%), 747 spam (13.4%).
