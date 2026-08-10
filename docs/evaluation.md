# Authory — Evaluation Report

## Overview & Methodology
This report presents the empirical evaluation results of the signal-based AI Essay Detector on college admissions essays. Evaluation is strictly performed using a **held-out topic split** to test generalization across topics rather than memorizing dataset vocabulary. In addition, non-native English speaker (**ESL**) essays are evaluated separately in a dedicated false-positive analysis.

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

### Confusion Matrix
| | Predicted Human | Predicted AI |
|---|---|---|
| **True Human** | 460 (TN) | 153 (FP - ESL bias) |
| **True AI** | 1 (FN) | 67 (TP) |

- **AI Detection Sensitivity (Recall):** 98.5% of AI-generated sentences correctly flagged (only 1 FN out of 68).
- **Native English Specificity:** 0.0% false positive rate on native human essays (0 out of 246 sentences misclassified).

---

## 2. ESL False-Positive Rate (FPR) Comparison

| Subset | False Positive Rate | FP / Total Sentences |
|---|---|---|
| **Native English Human Essays** | **0.0%** | 0 / 246 |
| **ESL Non-Native Human Essays** | **43.6%** | 153 / 351 |

### Behavioral & Statistical Mechanism:
ESL writing naturally exhibits lower vocabulary richness (TTR/MTLD) and lower per-token surprise (perplexity) due to simpler syntax and standard word choices. Single-model perplexity detectors frequently mistake these characteristics for LLM generation. Incorporating cross-perplexity (Binoculars ratio) and structural burstiness helps mitigate this bias, but a residual false-positive gap (43.6% vs 0.0%) persists and is transparently disclosed in the UI.

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

## 4. Summary & Product Guidelines
1. **Never Display Bare Percentages:** The interface must report qualitative summary bands rather than arbitrary probability scores (e.g. *"Likely AI-Assisted in Places"* instead of *"74% AI"*).
2. **Mandatory Limitations Disclosure:** The UI limitations panel must explicitly disclose the 43.6% ESL false-positive rate to prevent unfair accusations against non-native writers.
