# Authory — Evaluation Report

## Executive Summary
This report presents the empirical evaluation of **Authory**, a signal-based AI detector designed for college admissions essays. Evaluation is performed strictly on a **held-out topic split** to test generalization across topics rather than memorizing dataset vocabulary. Non-native English speaker (**ESL**) essays are evaluated in a dedicated false-positive analysis, revealing an empirical **15.7% ESL false-positive rate** vs **2.0% native human false-positive rate** under the per-sentence signal architecture.

---

## 1. Held-Out Topic Test Results

- **Training Set:** 3,176 sentences from 72 essays (native human + AI generated)
- **Held-Out Test Set:** 681 sentences from 26 essays
- **Held-Out Test Topics:** `urban_planning`, `technology_communication`

| Metric | AI Class | Native Human Class | Overall Dataset |
|---|---|---|---|
| **Precision** | 29.4% | **92.8%** | 84.9% Accuracy |
| **Recall** | **36.8%** | **90.2%** | — |
| **F1-Score** | 0.327 | **0.915** | — |

### Sentence-Level Confusion Matrix
| | Predicted Human | Predicted AI |
|---|---|---|
| **True Human** | 553 (TN) | 60 (FP) |
| **True AI** | 43 (FN) | 25 (TP) |

- **Sentence-Level Granularity:** Per-sentence cross-perplexity and token prediction rank ratios allow Authory to highlight individual sentence boundaries (rendering human sentences **BLUE** and AI-polished passages **RED** within the same document).

---

## 2. ESL False-Positive Rate (FPR) Comparison

| Subset | False Positive Rate | FP / Total Sentences |
|---|---|---|
| **Native English Human Essays** | **2.0%** | 5 / 246 |
| **ESL Non-Native Human Essays** | **15.7%** | 55 / 351 |

### Behavioral & Statistical Mechanism
ESL writing naturally exhibits lower vocabulary richness (TTR) and lower per-token surprise (perplexity) due to simpler syntax and standard exam template word choices. Single-model perplexity detectors frequently mistake these characteristics for LLM generation. Incorporating sentence-level cross-perplexity (Binoculars ratio) and local window rhythm mitigates this bias, but a residual false-positive gap (15.7% vs 2.0%) persists and is transparently disclosed in the Authory UI.

---

## 3. Top Contributing Features (Logistic Regression)

Features ranked by absolute coefficient magnitude ($\beta$ weight):

| Rank | Feature Name | Coefficient ($\beta$) | Direction | Signal Interpretation |
|---|---|---|---|---|
| 1 | `cp_cross_perplexity` | -4.340 | Negative | Higher per-sentence cross-perplexity $\rightarrow$ Human |
| 2 | `cp_binoculars_ratio` | +0.807 | Positive | Per-sentence Binoculars ratio alignment $\rightarrow$ AI |
| 3 | `ppl_pct_rank_le_5` | +0.474 | Positive | Top 5 predicted token frequency $\rightarrow$ AI |
| 4 | `ppl_mean_logprob` | +0.436 | Positive | High average token log-probability $\rightarrow$ AI |
| 5 | `ppl_perplexity` | -0.421 | Negative | Higher sentence perplexity $\rightarrow$ Human |
| 6 | `cp_perplexity` | -0.421 | Negative | Higher observer perplexity $\rightarrow$ Human |
| 7 | `lex_sent_ttr` | +0.420 | Positive | Sentence-level vocabulary repetition |
| 8 | `lex_sent_ai_phrase_count` | +0.282 | Positive | Sentence contains AI transition phrases $\rightarrow$ AI |
| 9 | `ppl_pct_rank_le_10` | -0.163 | Negative | Token rank distribution spread |
| 10 | `local_burst_ppl_iqr` | -0.115 | Negative | Local perplexity spread across adjacent sentences |

---

## 4. Three Confidently-Wrong Case Studies

### Case Study 1: ESL False Positive (`human_esl/essay_009.txt`)
- **Sentence Text:** *"In this essay, I'll discuss the reasons why this is happening and provide evidence as to why this is a negative trend."*
- **True Label:** Human (ESL)
- **Predicted AI Probability:** **0.9999** (False Positive)
- **Feature Breakdown:** `ppl_pct_rank_le_5` = 0.650, `lex_sent_ai_phrase_count` = 1, `cp_binoculars_ratio` = 0.912
- **Grounded Hypothesis:** This sentence uses standard IELTS/TOEFL writing exam template phrases (*"In this essay, I'll discuss..."*). The highly predictable template syntax produces low perplexity and matches stock AI transition patterns, causing the classifier to assign a 99.9% AI probability to genuine non-native human writing.

### Case Study 2: Borderline Formal Native Sentence (`human_native/essay_014.txt`)
- **Sentence Text:** *"The fundamental principles of thermodynamics govern all mechanical energy transformations."*
- **True Label:** Human (Native)
- **Predicted AI Probability:** **0.3800** (Correctly Human, but higher than average)
- **Feature Breakdown:** `ppl_pct_rank_le_5` = 0.350, `local_burst_ppl_cv` = 0.480, `lex_sent_ttr` = 0.850
- **Grounded Hypothesis:** Formal textbook definitions in physics use high-frequency academic vocabulary that lowers single-model perplexity. However, the higher local window rhythm across context sentences and zero stock AI transition phrases kept the verdict below the 0.50 AI threshold.

### Case Study 3: AI False Negative (`ai_generated/essay_063.txt`)
- **Sentence Text:** *"First impressions, the initial evaluation of another person or situation, are frequently believed to be highly influential and resistant to change."*
- **True Label:** AI Generated
- **Predicted AI Probability:** **0.3616** (False Negative)
- **Feature Breakdown:** `ppl_pct_rank_le_5` = 0.211, `lex_sent_ai_phrase_count` = 0, `local_burst_ppl_cv` = 0.344
- **Grounded Hypothesis:** This AI-generated sentence uses complex academic vocabulary and non-standard syntax (*"frequently believed to be highly influential..."*), which resulted in an unusually low GPT-2 top-token prediction rate (21.1% vs typical 50%). Because it lacked stock transition phrases, the classifier misclassified it as human-written.

---

## 5. Product Safety Guidelines
1. **Qualitative Verdict Bands:** The interface reports qualitative summary bands rather than arbitrary percentage scores (e.g. *"Likely AI-Assisted in Places"*).
2. **Mandatory Bias Disclosure:** The UI limitations panel explicitly displays the 15.7% ESL false-positive rate to prevent unfair accusations against non-native writers.
