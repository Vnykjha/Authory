# Authory — Evaluation Report

## Executive Summary
This report presents the empirical evaluation of **Authory**, a signal-based AI detector designed for college admissions essays. Evaluation is performed strictly on a **held-out topic split** to test generalization across topics rather than memorizing dataset vocabulary. Non-native English speaker (**ESL**) essays are evaluated in a dedicated false-positive analysis, revealing an empirical **43.6% ESL false-positive rate** vs **0.0% native human false-positive rate**.

---

## 1. Held-Out Topic Test Results

- **Training Set:** 3,176 sentences from 72 essays (native human + AI generated)
- **Held-Out Test Set:** 681 sentences from 26 essays
- **Held-Out Test Topics:** `urban_planning`, `technology_communication`

| Metric | AI Class | Native Human Class | Overall Dataset |
|---|---|---|---|
| **Precision** | 30.5% | **99.8%** | 77.4% Accuracy |
| **Recall** | **98.5%** | 75.0% | — |
| **F1-Score** | 0.465 | **0.857** | — |

### Sentence-Level Confusion Matrix
| | Predicted Human | Predicted AI |
|---|---|---|
| **True Human** | 460 (TN) | 153 (FP - ESL bias) |
| **True AI** | 1 (FN) | 67 (TP) |

- **AI Detection Sensitivity (Recall):** **98.5%** of AI-generated sentences correctly flagged (only 1 FN out of 68).
- **Native English Specificity:** **100.0%** specificity on native human essays (0 out of 246 sentences misclassified).

---

## 2. ESL False-Positive Rate (FPR) Comparison

| Subset | False Positive Rate | FP / Total Sentences |
|---|---|---|
| **Native English Human Essays** | **0.0%** | 0 / 246 |
| **ESL Non-Native Human Essays** | **43.6%** | 153 / 351 |

### Behavioral & Statistical Mechanism
ESL writing naturally exhibits lower vocabulary richness (TTR/MTLD) and lower per-token surprise (perplexity) due to simpler syntax and standard exam template word choices. Single-model perplexity detectors frequently mistake these characteristics for LLM generation. Incorporating cross-perplexity (Binoculars ratio) and structural burstiness helps mitigate this bias, but a residual false-positive gap (43.6% vs 0.0%) persists and is transparently disclosed in the Authory UI.

---

## 3. Top Contributing Features (Logistic Regression)

Features ranked by absolute coefficient magnitude ($\beta$ weight):

| Rank | Feature Name | Coefficient ($\beta$) | Direction | Signal Interpretation |
|---|---|---|---|---|
| 1 | `burst_ppl_mean` | -4.867 | Negative | Higher overall mean perplexity $\rightarrow$ Human |
| 2 | `burst_ppl_iqr` | -4.026 | Negative | Wider perplexity spread $\rightarrow$ Human |
| 3 | `lex_ttr` | +3.983 | Positive | Type-Token Ratio weighting |
| 4 | `burst_ppl_cv` | +2.347 | Positive | Perplexity variance weighting |
| 5 | `burst_len_iqr` | -1.847 | Negative | Wider sentence length IQR $\rightarrow$ Human |
| 6 | `lex_ai_phrase_rate` | +1.649 | Positive | High stock AI phrase frequency $\rightarrow$ AI |
| 7 | `burst_len_mean` | +1.302 | Positive | Mean sentence length contribution |
| 8 | `lex_mtld` | -1.212 | Negative | Higher MTLD lexical richness $\rightarrow$ Human |
| 9 | `burst_len_cv` | +1.161 | Positive | Uniform sentence length distribution $\rightarrow$ AI |
| 10 | `ppl_mean_logprob` | +0.972 | Positive | High average token probability $\rightarrow$ AI |

---

## 4. Three Confidently-Wrong Case Studies

### Case Study 1: ESL False Positive (`human_esl/essay_009.txt`)
- **Sentence Text:** *"In this essay, I'll discuss the reasons why this is happening and provide evidence as to why this is a negative trend."*
- **True Label:** Human (ESL)
- **Predicted AI Probability:** **0.9999** (False Positive)
- **Feature Breakdown:** `ppl_pct_rank_1` = 0.423, `lex_ai_phrase_rate` = 0.444, `burst_len_cv` = 0.843
- **Grounded Hypothesis:** This sentence uses standard IELTS/TOEFL writing exam template phrases (*"In this essay, I'll discuss..."*). The highly predictable template syntax produces low perplexity and matches stock AI transition patterns (`lex_ai_phrase_rate = 0.444`), causing the classifier to assign a 99.9% AI probability to genuine non-native human writing.

### Case Study 2: Borderline Formal Native Sentence (`human_native/essay_014.txt`)
- **Sentence Text:** *"The fundamental principles of thermodynamics govern all mechanical energy transformations."*
- **True Label:** Human (Native)
- **Predicted AI Probability:** **0.3800** (Correctly Human, but higher than average)
- **Feature Breakdown:** `ppl_pct_rank_1` = 0.350, `burst_len_cv` = 0.480, `lex_mtld` = 76.5
- **Grounded Hypothesis:** Formal textbook definitions in physics use high-frequency academic vocabulary that lowers single-model perplexity. However, the higher sentence length burstiness across context sentences and zero stock AI transition phrases kept the verdict below the 0.50 AI threshold.

### Case Study 3: AI False Negative (`ai_generated/essay_063.txt`)
- **Sentence Text:** *"First impressions, the initial evaluation of another person or situation, are frequently believed to be highly influential and resistant to change."*
- **True Label:** AI Generated
- **Predicted AI Probability:** **0.3616** (False Negative)
- **Feature Breakdown:** `ppl_pct_rank_1` = 0.211, `lex_ai_phrase_rate` = 0.000, `burst_len_cv` = 0.344
- **Grounded Hypothesis:** This AI-generated sentence uses complex academic vocabulary and non-standard syntax (*"frequently believed to be highly influential..."*), which resulted in an unusually low GPT-2 top-token prediction rate (21.1% vs typical 50%). Because it lacked stock transition phrases (`lex_ai_phrase_rate = 0.0`), the classifier misclassified it as human-written.

---

## 5. Product Safety Guidelines
1. **Qualitative Verdict Bands:** The interface reports qualitative summary bands rather than arbitrary percentage scores (e.g. *"Likely AI-Assisted in Places"*).
2. **Mandatory Bias Disclosure:** The UI limitations panel explicitly displays the 43.6% ESL false-positive rate to prevent unfair accusations against non-native writers.
