# Project 2: AI Detector for College Admissions Essays
### Full Project Description & Implementation Guide
**Hackathon: Callus i12 HR Drive Hackathon 2026 · Build window: 5 days (Days 6–10 of a 14-day event) · 4 shared buffer days**

*This document is self-contained: it includes the original brief requirements, research into how existing AI detectors work, the resulting solution design, dataset plan, evaluation methodology, and tech stack. It is written to be handed to another engineer (human or AI) with no other context and still be actionable.*

---

## 1. Original Brief (context)

Build a real, interactive application — not a script or notebook — that takes a college admissions essay as input and shows **which parts** were probably written by a machine and **why**. A single confidence percentage ("73% AI") is explicitly called out as insufficient.

Hard constraint from the brief: the app must not be a wrapper that sends the essay to a chat model and relays its verdict — that's called out as "unreliable, cannot explain its reasoning, and takes an afternoon to build." A model may be used **as an instrument** (e.g., to produce token probabilities), but the judgment must come from the app's own logic, not a model's opinion.

Also required:
- A dataset you build and document yourself (sources, size, gaps)
- Honest accuracy reporting: real test results, 3 confidently-wrong examples with a theory of why, not a bare accuracy claim
- Awareness that these detectors tend to flag non-native English writers — if the detector does this, the brief wants it acknowledged

**Judging criteria (all four apply to this project too):** clear communication, organised documentation, code quality, handling unfamiliar problems. AI tool use is expected and disclosed at the end, not scored against you.

---

## 2. Prior Art: How Existing AI Detectors Actually Work

Before designing the solution, it's worth knowing what's already out there, what works, and what fails — this directly informs both the architecture below and the honesty-reporting section, since a project that engages with known failure modes (rather than reinventing them blind) is stronger evidence of "handling unfamiliar problems."

### 2.1 GPTZero — perplexity + burstiness (the original approach)
GPTZero pioneered using two statistical signals computed from a language model run over the essay:
- **Perplexity**: how "surprised" a language model is by the text. A model assigns a probability to each next word given prior context; text that consistently uses the model's highest-probability word has low perplexity. AI-generated text tends to have systematically lower perplexity, because generation itself is a process of picking high-probability tokens.
- **Burstiness**: how much perplexity (or sentence length) varies across the document. Human writing is "bursty" — short and long sentences mixed, occasional low-probability word choices — while AI text tends to stay in a narrow, consistent band throughout.

GPTZero's original model combined these two signals into a small classifier (e.g., logistic regression) that outputs a per-sentence and per-document verdict. GPTZero has since moved to a more complex multilayer/deep-learning system, but perplexity and burstiness remain the conceptual foundation for most other statistical detectors on the market (ZeroGPT, Copyleaks, Originality.ai, and others reportedly build on the same core idea).

**This maps directly onto our "Signal Extraction Layer" design below — this project's core signals are essentially a from-scratch, transparent reimplementation of this same idea, extended with a few more modern techniques.**

### 2.2 DetectGPT / Fast-DetectGPT — probability curvature
DetectGPT takes a different zero-shot approach: it perturbs a candidate passage (small word substitutions/rewrites) and re-scores it under a language model. The idea is that machine-generated text sits near a local maximum of the model's probability landscape, so small perturbations tend to *lower* its probability more than they would for human text, which sits in a flatter, less-optimized region. Fast-DetectGPT approximates the same "curvature" signal far more cheaply, using conditional probability estimates instead of many perturbation re-scorings, making it far more practical to run.

This is powerful but computationally heavier than perplexity/burstiness (many re-scoring passes per passage) — noted here as a considered-but-deliberately-cut signal for this project's timeframe (see §5).

### 2.3 Binoculars — cross-perplexity between two models (state of the art, zero-shot)
Binoculars is a more recent (ICML 2024) zero-shot method that doesn't need any training data at all. It scores a passage using **two** closely related language models instead of one:
- An "observer" model computes the straightforward perplexity of the text.
- A "performer" model generates next-token predictions for the same text, which the observer then scores (this is called cross-perplexity).
- The **ratio** of perplexity to cross-perplexity is the "Binoculars score." Because both AI-generated and human text pass through the same two-model comparison, this ratio is far more robust across different source LLMs and writing domains than perplexity alone.

Reported results show very high detection accuracy at an extremely low false-positive rate, without training on any labeled AI text, and — notably — **it is reported to be less biased against non-native English writers than single-model perplexity detectors**, because the ratio partly cancels out the "low perplexity because of simple vocabulary" confound described in §2.5 below.

**This project adopts a simplified, single-extra-model version of this idea as its second core signal (see §4) — it's a meaningfully stronger, still-implementable-in-a-day upgrade over plain perplexity.**

### 2.4 Trained classifiers (Turnitin, Pangram, Originality.ai, OpenAI's now-retired classifier)
Commercial detectors used at scale (Turnitin's AI writing indicator, Pangram, Originality.ai) are generally supervised classifiers trained on large labeled datasets of human vs. AI text, sometimes combined with statistical features like the above. OpenAI released and then retired its own classifier in 2023 due to low accuracy. These tools are proprietary black boxes — useful to know they exist, not useful as an architecture reference since none publish their exact method.

### 2.5 The known failure mode: bias against non-native English (ESL) writers
This is directly called out in the brief, and it's a well-documented, real phenomenon — not a hypothetical to guess at:

- A widely-cited Stanford study (Liang et al., 2023, published in *Patterns*) tested seven commercial GPT detectors on 91 real TOEFL essays (written by non-native English speakers) against 88 U.S. eighth-grade essays. The detectors were near-perfect on the native-speaker essays but **misclassified more than half the TOEFL essays as AI-generated** (average false-positive rate ~61%), and nearly 20% of TOEFL essays were unanimously flagged by all seven detectors.
- The mechanism is exactly what §2.1 predicts: non-native writers tend to use more standard, less varied vocabulary and simpler sentence constructions — which produces **lower perplexity**, the same statistical signature perplexity-based detectors use to flag AI text. The bias isn't a quirk of one bad detector; it falls directly out of the core signal itself.
- The same study found that artificially enriching the TOEFL essays' vocabulary (via an LLM prompt to "sound more native") dropped the false-positive rate from ~61% to under 12% — strong evidence the mechanism above is the actual cause.
- Some newer systems report much lower ESL false-positive rates by deliberately including non-native writing samples in their training data, and the Binoculars paper reports its two-model ratio is comparatively more robust to this bias than plain single-model perplexity.

**Implication for this project:** don't just "check for" ESL bias as an afterthought at the end — design the dataset and evaluation to actively measure it from day one (§6, §7), since it is the single most well-documented and citable failure mode of this whole category of tool.

---

## 3. Solution Overview

Given the above, the design is: build the same statistical foundation these detectors are built on, transparently, with a second-model cross-perplexity signal for robustness, feed the results into a classifier you train yourself, and — critically — measure and report the ESL bias explicitly rather than hoping it doesn't show up.

```
Essay text
   │
   ▼
[1] Signal Extraction Layer   ── computes numeric features per sentence
   │        (single-model perplexity/rank + two-model cross-perplexity
   │         + stylometric stats — no verdicts, just numbers)
   ▼
[2] Classifier Layer          ── trained model turns features → per-sentence
   │                              AI-likelihood + top contributing reasons
   ▼
[3] Interface Layer           ── essay rendered with highlighted spans,
                                  hover explanations, honest limitations panel
                                  (including a measured ESL false-positive rate)
```

The language model(s) only ever produce **numbers** — log-probabilities, token rank, cross-perplexity ratios. The classifier, trained on your own labeled dataset, makes the actual call. This is the exact line the brief draws between "instrument" and "wrapper."

---

## 4. Signal Extraction (the core differentiator)

Four signal families, chosen to be implementable within a single day of work each while directly building on the prior-art research above rather than reinventing it blind.

| Signal | What it measures | Why it works | Basis |
|---|---|---|---|
| **Single-model perplexity & token rank** | For each token, how predictable it was to a local LM (log-probability + rank in the model's predicted distribution) | AI text sits close to a language model's probability peak; human text reaches for lower-probability, higher-rank word choices more often | GPTZero's original approach (§2.1) |
| **Cross-perplexity ratio (simplified Binoculars)** | Ratio between perplexity from one local model and cross-perplexity computed by scoring one model's predictions with a second, different model | More robust across writing styles/domains than single-model perplexity alone; partially cancels out the "simple vocabulary = low perplexity" confound that drives ESL false positives | Binoculars (§2.3) |
| **Burstiness / sentence-length variance** | Std. dev. of sentence length (and of per-sentence perplexity) across the essay, divided by the mean | Human writing varies pacing; AI writing tends to stay in a narrow, consistent band throughout a passage | GPTZero (§2.1) |
| **Lexical & phrasal fingerprints** | Type-token ratio, lexical diversity (MTLD), frequency of stock AI transition phrases ("Moreover," "This experience taught me," "As I reflect on...") | LLMs over-use certain connective phrasing and under-use rare words | Observed pattern across detector literature |

**Deliberately cut from v1 (a documented scope decision, not an oversight):**
- **DetectGPT/Fast-DetectGPT-style perturbation curvature** (§2.2) — meaningfully more compute per passage (multiple re-scoring passes) for a build window this short; the cross-perplexity signal captures a similar idea more cheaply.
- **Large-scale POS n-gram comparison against a reference corpus** — nice-to-have, not essential to the core argument, and the added engineering time is better spent on the ESL evaluation work in §7.

Being explicit about what was left out and *why*, with reference to what it would have bought, is itself a "decision you can defend" — exactly what's graded.

---

## 5. Classifier Layer

- **Model:** Logistic Regression as the baseline; XGBoost/LightGBM as a stretch goal if a day allows.
  - Chosen over a neural classifier deliberately: interpretable (coefficients map straight to plain-language reasons), fast to train on a dataset this size, and avoids adding a second "black box" on top of an already partly-opaque LM signal.
- **Granularity:** sentence-level, with a small neighboring-sentence context window — because the realistic failure case (per the brief) is a human-drafted paragraph with an AI polish pass, not a whole essay generated wholesale. Whole-essay classification would miss exactly the case the brief cares most about.
- **Output per sentence:** AI-likelihood score (0–1) + top 2–3 contributing features translated into plain language (e.g. "unusually smooth word choice," "sentence rhythm very close to essay average," "stock transition phrase").
- **Aggregation:** roll sentence scores into paragraph/essay-level summaries, but always keep the sentence-level evidence visible in the UI — never collapse to a bare percentage, per the brief's explicit instruction.

---

## 6. Dataset (build Day 6, in parallel with signal extraction)

| Category | Source | Target size | Purpose |
|---|---|---|---|
| Human essays (native-English baseline) | Public example-essay collections (university-published samples, open example archives) | 60–100 | Core human class |
| Human essays — **ESL subset** | Non-native-English-authored human essays, sourced/tagged separately | 15–20 | Dedicated bias-measurement set — mirrors the TOEFL-vs-native design from the Stanford study (§2.5), on a much smaller scale |
| AI-generated essays | 2 different models/prompt styles, varied topics | 60–100 | Core AI class |
| Hybrid essays (human draft + AI polish pass) | Take a subset of human essays, run an "improve this essay" pass through a model | 20–30 | The realistic case the brief flags — a detector that only handles wholesale AI generation misses the actual use case |

**Split by topic and by source, not randomly** — enables an honest held-out-topic generalization test rather than an inflated same-distribution accuracy number.

Document in `docs/dataset.md`: exact sources/URLs, essay counts per category, topics covered, and known gaps (e.g. "all human essays are US-style CommonApp prompts; the ESL subset is drawn from a single regional background and may not generalize to all L2-English profiles").

---

## 7. Evaluation & Honesty Reporting

This is a first-class deliverable, not an appendix — and now explicitly modeled on the real, published methodology that exposed this exact bias in commercial detectors.

- Precision/recall/F1 per class + confusion matrix (never accuracy alone)
- **Held-out-topic test**: train on 4 topics, test on a 5th, report the accuracy drop honestly
- **ESL false-positive check** (directly modeled on §2.5): report the false-positive rate on the ESL subset vs. the general human subset side by side. If there's a large gap — and based on the published literature, there is a real chance there will be — report the number plainly rather than hiding it, and explain the likely mechanism (lower lexical variability → lower perplexity → misread as "AI-like").
- **3 confidently-wrong examples**, each with a written hypothesis grounded in the actual feature values for that sentence, not a generic guess.
- All of this in `docs/evaluation.md`, and summarized in an in-app "Model Limitations" panel that states the measured ESL disparity in plain language — this is a strong, cheap signal to a reviewer that the known failure mode wasn't just acknowledged in passing but actually measured.

---

## 8. Interface

- **Paste box → Analyze button → rendered essay** with sentence spans highlighted, color intensity mapped to AI-likelihood.
- **Click/hover a span** → side panel shows the 2–3 contributing signals in plain language.
- **Summary panel**: overall estimate shown as a qualitative band ("mixed," "likely AI-assisted in places"), never a bare percentage.
- **"How this works" + "Known limitations" panel**: links to the evaluation doc's failure cases and the measured ESL false-positive gap — framed honestly ("this tool, like most AI detectors, is measurably less reliable on non-native English writing — here's our measured rate").

---

## 9. Tech Stack

| Layer | Choice | Why |
|---|---|---|
| Local LM(s) for signals | Two small causal LMs via Hugging Face `transformers` (e.g. GPT-2 as observer + a second small model such as GPT-2-medium or Pythia-160m as performer), CPU inference | Small enough to run fast on essay-length text; a genuine two-model pair is what the cross-perplexity signal (§4) needs |
| Feature engineering | Python: `textstat`, `spaCy`, `numpy` | Mature libraries for readability/lexical stats |
| Classifier | `scikit-learn` (LogReg baseline) → `xgboost`/`lightgbm` if time allows | Interpretable, fast to train on a small dataset |
| Backend/API | Python `FastAPI` | Clean fit with the ML stack, quick to stand up endpoints |
| Frontend | React (or plain Next.js page) + Tailwind | Simple span-highlighting UI, no heavy framework needed |
| Storage | Flat files / SQLite for dataset + eval results | No need for anything heavier at this scale |

---

## 10. 5-Day Build Plan

| Day | Focus | Output |
|---|---|---|
| **Day 6** | Dataset sourcing + generation (human, AI, hybrid, **ESL subset**); signal extraction scaffold | `data/` populated + documented, single-model perplexity working |
| **Day 7** | Add cross-perplexity (second model) + burstiness + lexical stats; run full feature extraction over dataset | `features.csv` for every sentence in the dataset |
| **Day 8** | Train classifier, run first evaluation pass including the held-out-topic split | Trained model + baseline metrics |
| **Day 9** | Build interface: paste box, highlighting, explanation panel; wire to backend | End-to-end working app |
| **Day 10** | Honest evaluation write-up: confusion matrix, 3 wrong examples, held-out-topic test, **ESL false-positive comparison**; wire limitations panel into UI | `docs/evaluation.md` + polished limitations panel with real ESL numbers |

*(Days 11–14 shared buffer with Project 1 for debugging, polish, README, and video recording.)*

---

## 11. Judging Criteria Alignment

| Criterion | How this design addresses it |
|---|---|
| **Clear communication** | Plain-language per-sentence explanations; limitations panel written for a lay reader, stating the measured ESL gap explicitly |
| **Organised documentation** | `docs/dataset.md`, `docs/evaluation.md`, README with architecture diagram, explicit citation of the prior-art research this design is based on |
| **Code quality** | Clean separation of signal extraction / classifier / interface layers; interpretable model choice over an opaque neural net |
| **Handling unfamiliar problems** | Explicit engagement with real, published failure modes (ESL bias, wrapper vs. instrument distinction) rather than discovering them by accident; documented, reasoned scope cuts (DetectGPT-style perturbation, large POS n-gram corpus) |

---

## 12. Deliverables Checklist

- [ ] Public GitHub repo, pushed (not a fork)
- [ ] Working app: paste essay → highlighted output with explanations
- [ ] `docs/dataset.md` — sources, counts, coverage gaps, ESL subset methodology
- [ ] `docs/evaluation.md` — metrics, confusion matrix, 3 wrong examples + hypotheses, held-out-topic test, **ESL false-positive rate vs. general human rate**
- [ ] README with architecture overview, scope decisions, and citations to the prior-art detectors this design builds on
- [ ] (Optional) short video walkthrough
- [ ] Honest AI-tool-usage disclosure, written at submission time

---

## 13. References Consulted (for context, not for reproduction)

- GPTZero's published explanations of perplexity and burstiness as detection signals
- Mitchell et al., *DetectGPT: Zero-Shot Machine-Generated Text Detection using Probability Curvature* (ICML 2023)
- Hans et al., *Spotting LLMs With Binoculars: Zero-Shot Detection of Machine-Generated Text* (ICML 2024)
- Liang et al., *GPT detectors are biased against non-native English writers*, Patterns (2023) — source of the TOEFL false-positive-rate findings used to design §6–§7
- Pangram AI-Generated Text Classifier technical report — for context on how a modern trained classifier handles the ESL benchmark
