# Implementation Plan: AI Essay Detector

**Project:** AI Detector for College Admissions Essays  
**Build Window:** 5 focused days (Days 6–10) + 4 buffer days  
**Created:** 2026-08-04  
**Status:** Planning phase — awaiting review before Day 6 start

---

## Overview

This plan breaks the 5-day build window from `docs/project2-ai-essay-detector.md` into granular, checkable daily tasks with explicit "Definition of Done" (DoD) criteria. Each day ends with a review checkpoint — **no code for the next day starts until the current day's DoD is verified.**

The architecture has three layers:
1. **Signal Extraction** — computes numeric features per sentence (no verdicts)
2. **Classifier** — trained model turns features → per-sentence AI-likelihood + reasons
3. **Interface** — rendered essay with highlighted spans, hover explanations, limitations panel

---

## Day 6 — Dataset Sourcing + Signal Extraction Scaffold

**Goal:** Populate `data/` with documented human/AI/hybrid/ESL essays; get single-model perplexity working end-to-end.

### Tasks

| # | Task | Details |
|---|------|---------|
| 6.1 | Create project structure & dependencies | `pyproject.toml` with `transformers`, `torch`, `scikit-learn`, `fastapi`, `pydantic`, `textstat`, `spacy`, `numpy`, `pandas`; directory layout: `data/`, `src/signals/`, `src/classifier/`, `src/api/`, `src/frontend/` |
| 6.2 | Source human essays (native) | 60–100 essays from public university example collections; save as `data/human_native/*.txt`; log sources/URLs in `docs/dataset.md` |
| 6.3 | Source ESL human essays | 15–20 non-native English essays (separate folder `data/human_esl/*.txt`); document provenance separately |
| 6.4 | Generate AI essays | 60–100 essays via 2+ models/prompt styles; save `data/ai_generated/*.txt`; log model, prompt, temperature |
| 6.5 | Generate hybrid essays | 20–30: take human essays → run "improve this essay" pass via LLM; save `data/hybrid/*.txt` |
| 6.6 | Document dataset | Write `docs/dataset.md` with exact sources, counts, topics, gaps (per §6 of project doc) |
| 6.7 | Signal extraction scaffold | `src/signals/perplexity.py`: load GPT-2, compute per-token log-probs & rank, aggregate to per-sentence perplexity; unit test on sample text |
| 6.8 | Sentence segmentation | `src/signals/segment.py`: spaCy-based sentence splitter that preserves offsets for UI highlighting |

### Definition of Done (Day 6)

- [ ] `data/` contains four subdirectories with target essay counts
- [ ] `docs/dataset.md` exists with sources, counts, topics, gaps documented
- [ ] `src/signals/perplexity.py` loads GPT-2 and returns per-sentence perplexity + token ranks for a test essay
- [ ] `src/signals/segment.py` splits text into sentences with character offsets
- [ ] All dependencies install cleanly via `pip install -e .`
- [ ] No classifier or interface code written yet

---

## Day 7 — Full Feature Extraction (Cross-Perplexity + Burstiness + Lexical)

**Goal:** Add second model for cross-perplexity, burstiness, lexical stats; produce `features.csv` for every sentence in the dataset.

### Tasks

| # | Task | Details |
|---|------|---------|
| 7.1 | Add second model for cross-perplexity | Load GPT-2 + GPT-2-medium (or Pythia-160m) as observer/performer pair; implement cross-perplexity ratio per sentence |
| 7.2 | Implement burstiness | Std-dev of sentence length & per-sentence perplexity divided by mean; per-essay + per-sentence context window |
| 7.3 | Implement lexical/phrasal fingerprints | Type-token ratio, MTLD (via `textstat` or custom), stock AI transition phrase frequency list |
| 7.4 | Feature extraction pipeline | `src/signals/extract.py`: orchestrate all signals → single feature dict per sentence; handle errors gracefully |
| 7.5 | Run extraction over full dataset | Script `scripts/extract_features.py` processes all essays → `data/features.csv` with columns: `essay_id`, `sentence_idx`, `text`, `label`, `source_category`, all feature columns |
| 7.6 | Validate feature distributions | Quick EDA: check for NaNs, extreme outliers, class separation on key features |

### Definition of Done (Day 7)

- [ ] Two models loaded and cross-perplexity ratio computed per sentence
- [ ] Burstiness (sentence length + perplexity variance) computed per essay
- [ ] Lexical diversity (TTR, MTLD) + AI transition phrase counts per sentence
- [ ] `data/features.csv` exists with one row per sentence, all features + labels
- [ ] No NaNs in feature matrix; basic class separation visible on 2+ features
- [ ] Extraction script is re-runnable and idempotent

---

## Day 8 — Classifier Training + First Evaluation Pass

**Goal:** Train Logistic Regression (baseline) + optional XGBoost; run held-out-topic evaluation.

### Tasks

| # | Task | Details |
|---|------|---------|
| 8.1 | Prepare train/test splits | Split by **topic** (not randomly): 4 topics train, 1 held-out test; separate ESL subset never in training |
| 8.2 | Train Logistic Regression | `src/classifier/train.py`: LogReg with class balancing; output per-sentence probability + feature coefficients |
| 8.3 | (Stretch) Train XGBoost/LightGBM | If time allows; compare to LogReg on held-out-topic |
| 8.4 | Implement prediction + explanation | `src/classifier/predict.py`: given features → probability + top-3 contributing features (by coefficient × value) mapped to plain-language reasons |
| 8.5 | Run held-out-topic evaluation | Confusion matrix, precision/recall/F1 per class on held-out topic; log to `docs/evaluation.md` |
| 8.6 | Run ESL false-positive check | Evaluate on ESL subset vs. general human subset; record FPR gap |
| 8.7 | Save model artifacts | `models/logreg.joblib` + feature name list for API loading |

### Definition of Done (Day 8)

- [ ] Logistic Regression trained and saved
- [ ] Held-out-topic test results in `docs/evaluation.md` (confusion matrix, P/R/F1)
- [ ] ESL false-positive rate vs. general human rate recorded side-by-side
- [ ] Per-sentence prediction returns probability + top-3 plain-language reasons
- [ ] Model loads in < 1s for API use
- [ ] (Optional) XGBoost trained and compared

---

## Day 9 — Interface Build (Paste Box → Highlighted Essay + Explanations)

**Goal:** End-to-end working web app: paste essay → analyze → highlighted spans with hover explanations.

### Tasks

| # | Task | Details |
|---|------|---------|
| 9.1 | Backend API | `src/api/main.py`: FastAPI with `/analyze` endpoint; loads model + signal extractors; returns per-sentence: text, start/end offsets, AI probability, top reasons |
| 9.2 | Frontend scaffold | React + Tailwind (or Next.js single page); paste box + "Analyze" button |
| 9.3 | Essay rendering with highlights | Render essay as `<span>` per sentence; color intensity = AI probability (e.g., transparent → red); click/hover → side panel |
| 9.4 | Explanation panel | Side panel shows top 2–3 reasons in plain language for selected sentence |
| 9.5 | Summary panel | Qualitative band ("mixed," "likely AI-assisted in places") — **never a bare percentage** |
| 9.6 | "How this works" + "Limitations" panel | Static content linking to evaluation findings; includes measured ESL FPR gap in plain language |

### Definition of Done (Day 9)

- [ ] `uvicorn src.api.main:app` starts and serves `/analyze`
- [ ] Frontend loads, accepts paste, calls API, renders highlighted essay
- [ ] Hover/click shows explanation panel with correct reasons
- [ ] Summary panel shows qualitative band, not percentage
- [ ] Limitations panel displays measured ESL FPR gap
- [ ] End-to-end test: paste a known AI essay → shows high AI spans; paste human → shows low

---

## Day 10 — Honest Evaluation Write-Up + Polish

**Goal:** Complete `docs/evaluation.md` with all required sections; polish UI; final review.

### Tasks

| # | Task | Details |
|---|------|---------|
| 10.1 | Confusion matrix + per-class metrics | Final numbers on held-out-topic test |
| 10.2 | 3 confidently-wrong examples | For each: essay excerpt, predicted vs. true label, feature values, written hypothesis grounded in those features |
| 10.3 | Held-out-topic accuracy drop | Compare same-topic vs. held-out-topic performance; report honestly |
| 10.4 | ESL false-positive comparison | Side-by-side FPR table; mechanism explanation (lower lexical variability → lower perplexity) |
| 10.5 | Finalize `docs/evaluation.md` | All sections complete, readable by lay audience |
| 10.6 | Wire limitations panel to real numbers | Ensure UI panel reads from evaluation doc or embeds the exact numbers |
| 10.7 | README | Architecture diagram, scope decisions (what was cut + why), citations to prior art |
| 10.8 | AI tool usage disclosure | Write at submission time |
| 10.9 | Final polish & smoke test | Full end-to-end run on 5 diverse essays; no crashes |

### Definition of Done (Day 10)

- [ ] `docs/evaluation.md` complete with: confusion matrix, 3 wrong examples + hypotheses, held-out-topic drop, ESL FPR comparison
- [ ] README exists with architecture, scope cuts, citations
- [ ] UI limitations panel shows real measured ESL gap
- [ ] No bare percentages anywhere in UI
- [ ] Smoke test passes on 5 diverse essays
- [ ] Repo pushed to GitHub (public, not fork)
- [ ] All deliverables from §12 of project doc checked off

---

## Buffer Days (Days 11–14) — Shared with Project 1

| Day | Focus |
|-----|-------|
| 11 | Bug fixes, edge cases, performance tuning |
| 12 | Video walkthrough recording |
| 13 | Final README/ doc polish, AI disclosure |
| 14 | Submission preparation |

---

## Review Checkpoints

**After each day (6–10), STOP and request review.** Do not proceed to the next day's tasks until:
1. All DoD items for the current day are checked
2. User confirms the day's output is acceptable
3. Any feedback is incorporated

---

## Scope Decisions (Documented Up Front)

| Cut | Reason |
|-----|--------|
| DetectGPT / Fast-DetectGPT perturbation curvature | Multiple re-scoring passes per passage = too much compute for 5-day window; cross-perplexity captures similar signal cheaper |
| Large-scale POS n-gram comparison | Nice-to-have; engineering time better spent on ESL evaluation |
| Neural classifier | LogReg/XGBoost are interpretable (coefficients → plain-language reasons); avoids second black box |
| Random train/test split | Split by topic enables honest generalization test; random split inflates accuracy |
| Whole-essay classification only | Brief explicitly wants per-sentence detection for hybrid essays |

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| ESL false-positive rate very high (>50%) | High (per literature) | Core deliverable requirement | Measure from Day 6; design dataset with ESL subset; report honestly |
| Cross-perplexity needs GPU | Medium | Blocks Day 7 | Use small models (GPT-2 + Pythia-160m) on CPU; batch inference |
| Feature extraction too slow for UI | Low | UX degradation | Cache model outputs; optimize batch size; consider ONNX export if needed |
| Dataset sourcing takes >1 day | Medium | Compresses later days | Start Day 6 early; use parallel collection scripts |
| Held-out-topic accuracy near random | Medium | Undermines credibility | Ensure topic diversity in dataset; LogReg is robust baseline |

---

## Quick Reference: File Targets by Day

| Day | New Files |
|-----|-----------|
| 6 | `pyproject.toml`, `docs/dataset.md`, `src/signals/perplexity.py`, `src/signals/segment.py`, `data/` dirs |
| 7 | `src/signals/cross_perplexity.py`, `src/signals/burstiness.py`, `src/signals/lexical.py`, `src/signals/extract.py`, `scripts/extract_features.py`, `data/features.csv` |
| 8 | `src/classifier/train.py`, `src/classifier/predict.py`, `models/logreg.joblib`, `docs/evaluation.md` (initial) |
| 9 | `src/api/main.py`, `src/frontend/` (React app), `src/api/schemas.py` |
| 10 | `docs/evaluation.md` (final), `README.md`, AI disclosure |

---

**Next Step:** Await review of this plan. Once approved, begin Day 6 implementation.