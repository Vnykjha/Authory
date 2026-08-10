# Module 10: Final Evaluation Write-Up & Polish
**Day 10 — Tasks 10.1–10.9**  
**Estimated time:** 4–5 hours  
**Depends on:** Modules 6, 7, 8, 9 (all code complete)

---

## Objective
Complete the honest evaluation document with all required sections, wire real ESL numbers into the UI, finalize README, and run smoke tests.

---

## Inputs Required
- `docs/evaluation.md` (initial version from Module 6)
- Module 6 test predictions on held-out topics + ESL subset
- Trained model artifacts (`models/logreg.joblib`)
- Working end-to-end app (Modules 7–9)

---

## Outputs Produced
| Path | Description |
|------|-------------|
| `docs/evaluation.md` (final) | Complete evaluation with all required sections |
| `README.md` | Architecture, scope decisions, citations |
| `AI_USAGE_DISCLOSURE.md` | Honest AI tool usage statement |
| Updated `src/frontend/src/components/LimitationsPanel.tsx` | Real ESL FPR numbers |
| GitHub repo pushed (public) | All deliverables committed |

---

## Step-by-Step Tasks

### 10.1 Confusion Matrix + Per-Class Metrics (Finalize)

Run final evaluation on held-out-topic test set:
```bash
cd /path/to/project
python -m src.classifier.train  # Re-run to get fresh metrics
```

Record in `docs/evaluation.md`:
```markdown
## Final Held-Out Topic Test Results

| Metric | Value |
|---|---|
| Accuracy | 0.XXX |
| AI Precision / Recall / F1 | X.XXX / X.XXX / X.XXX |
| Human Precision / Recall / F1 | X.XXX / X.XXX / X.XXX |

### Confusion Matrix (Sentence-Level)
| | Predicted Human | Predicted AI |
|---|---|---|
| **True Human** | TN = XX | FP = XX |
| **True AI** | FN = XX | TP = XX |

*Test set: N sentences from M essays on [held-out topic name(s)]*
```

---

### 10.2 Three Confidently-Wrong Examples

**Process:**
1. Run classifier on test set, collect misclassified sentences with high confidence (|prob - 0.5| > 0.3)
2. Pick 3 diverse examples (false positive human, false negative AI, hybrid confusion)
3. For each, document:

```markdown
### Wrong Example 1: [False Positive / False Negative / Hybrid]

**Essay excerpt:**
> "Sentence text here..."

**True label:** Human (Native) / AI / Hybrid
**Predicted:** AI (0.87) / Human (0.12) / etc.

**Top contributing features:**
| Feature | Value | Coefficient | Contribution |
|---|---|---|---|
| ppl_pct_rank_1 | 0.42 | +1.23 | +0.52 |
| burst_len_cv | 0.15 | -0.89 | -0.13 |
| ... | ... | ... | ... |

**Hypothesis:** 
This human-written sentence uses unusually consistent word choices (high rank-1 percentage) and uniform sentence length, mimicking the statistical signature of AI generation. The writer may have been following a strict template or consciously using "formal" vocabulary that aligns with model predictions. The low burstiness (CV=0.15) further reinforces the AI-like pattern.
```

**Requirements for each hypothesis:**
- Grounded in **actual feature values** for that sentence
- References specific signals (perplexity, burstiness, lexical)
- Explains *why* the feature misled the classifier
- Not a generic guess

---

### 10.3 Held-Out-Topic Accuracy Drop

```markdown
## Held-Out Topic Generalization

| Topic | Test Sentences | Accuracy | F1 (AI) | F1 (Human) |
|---|---|---|---|---|
| [Training topic 1] | N | X.XXX | X.XXX | X.XXX |
| [Training topic 2] | N | X.XXX | X.XXX | X.XXX |
| ... | ... | ... | ... | ... |
| **Held-out topic: [name]** | N | **X.XXX** | **X.XXX** | **X.XXX** |

**Accuracy drop (avg training → held-out):** X.X percentage points

**Analysis:** [Brief explanation — e.g., "Drop of 8% suggests topic-specific vocabulary affects perplexity signals; cross-perplexity ratio partially mitigates but doesn't eliminate."]
```

---

### 10.4 ESL False-Positive Comparison (Side-by-Side)

```markdown
## ESL False-Positive Rate Analysis

| Subset | Essays | Sentences | False Positives | Total Human | FPR |
|---|---|---|---|---|---|
| Native Human (held-out topics) | N | N | FP | TN+FP | X.XXX |
| **ESL Human (never in training)** | N | N | **FP** | **TN+FP** | **X.XXX** |

**Gap:** ESL FPR is **X.X× higher** than native human FPR.

### Mechanism Explanation
Per the Liang et al. (2023) study and our feature analysis:
1. **Lower lexical diversity:** ESL essays show lower TTR (mean: X.XX vs X.XX) and MTLD (X.XX vs X.XX)
2. **Lower perplexity:** Simpler vocabulary → higher model prediction confidence → lower perplexity
3. **Reduced burstiness:** More uniform sentence structures → lower length/perplexity CV
4. **Cross-perplexity mitigation:** The Binoculars ratio reduces but doesn't eliminate the gap (ESL ratio: X.XX vs Native: X.XX)

This is a **known, documented limitation** of perplexity-based detection — not a bug in our implementation. Users should be aware that non-native English writing will be flagged more often.
```

---

### 10.5 Finalize `docs/evaluation.md`

Assemble all sections into complete document:
```markdown
# Evaluation Report: AI Essay Detector

## Executive Summary
[2-3 sentences: what was built, key results, main limitation]

## Dataset Summary
[Reference docs/dataset.md; key stats only]

## Methodology
- Signal extraction: [list 4 signals]
- Classifier: Logistic Regression, balanced classes
- Split: Held-out topic (topic: [name]), ESL never in training
- Evaluation: Sentence-level, per-class metrics

## Results
### Held-Out Topic Test
[Confusion matrix, metrics table]

### ESL False-Positive Rate
[Side-by-side table + mechanism]

### Topic Generalization
[Per-topic table + drop analysis]

## Three Confidently-Wrong Examples
[Three examples with hypotheses]

## Limitations Summary
[Bulleted list from LimitationsPanel]

## Reproducibility
- `python -m src.classifier.train` reproduces training
- `data/features.csv` + `data/essay_metadata.csv` for independent analysis
- Random seed: 42
```

---

### 10.6 Wire Real ESL Numbers Into UI

Update `src/frontend/src/components/LimitationsPanel.tsx`:
```tsx
// Replace placeholder props with real numbers from evaluation
<LimitationsPanel 
  eslFpr={0.XX}   // e.g., 0.42
  nativeFpr={0.XX} // e.g., 0.08
/>
```

**Also update the explanation panel link** to point to the actual evaluation doc.

---

### 10.7 README.md

```markdown
# AI Essay Detector for College Admissions Essays

> **Signal-based, not a wrapper.** This detector runs essays through two small language models (GPT-2 family) to extract statistical signals — perplexity, cross-perplexity (Binoculars), burstiness, and lexical fingerprints — then classifies sentences with a logistic regression trained on a documented dataset. No LLM is asked for a verdict.

## Architecture
```
Essay → [Signal Extraction] → [Classifier] → [Interface]
         (4 signal families)    (LogReg)       (Highlights + Reasons)
```

### Signal Extraction Layer (No Verdicts)
| Signal | Basis | What It Measures |
|---|---|---|
| Single-model perplexity + token rank | GPTZero (§2.1) | Predictability of word choices |
| Cross-perplexity ratio (Binoculars) | Hans et al., ICML 2024 | Robustness across styles; mitigates ESL bias |
| Burstiness (length + perplexity variance) | GPTZero | Variation in sentence rhythm |
| Lexical/phrasal fingerprints | Observed patterns | Vocabulary diversity + AI stock phrases |

### Classifier
- Logistic Regression (interpretable coefficients → plain-language reasons)
- Trained on 140+ essays: human (native + ESL), AI (2+ models), hybrid
- Split by **topic**, not randomly — honest generalization test

### Interface
- Per-sentence highlighting (color intensity = AI probability)
- Click for contributing signals in plain language
- Qualitative summary bands — **never a bare percentage**
- "Known Limitations" panel with measured ESL false-positive gap

## Scope Decisions (What We Cut & Why)
| Cut | Reason |
|---|---|
| DetectGPT / Fast-DetectGPT perturbation | Too much compute (multiple re-scoring passes); cross-perplexity captures similar signal |
| Large POS n-gram corpus | Nice-to-have; time better spent on ESL evaluation |
| Neural classifier | LogReg is interpretable; avoids second black box |
| Random train/test split | Topic split enables honest generalization test |
| Whole-essay classification only | Brief requires per-sentence detection for hybrid essays |

## Evaluation Highlights
- **Held-out topic F1:** X.XX (AI) / X.XX (Human)
- **ESL false-positive rate:** X.XX% vs native X.XX% (gap documented honestly)
- **3 wrong examples analyzed** with feature-grounded hypotheses
- Full details: [`docs/evaluation.md`](docs/evaluation.md)

## Quick Start
```bash
# Backend
pip install -e .
uvicorn src.api.main:app --reload

# Frontend (new terminal)
cd src/frontend
npm install
npm run dev
```
Then open http://localhost:5173

## Citations
- GPTZero: Perplexity + burstiness as detection signals
- Mitchell et al., *DetectGPT* (ICML 2023)
- Hans et al., *Spotting LLMs With Binoculars* (ICML 2024)
- Liang et al., *GPT detectors are biased against non-native English writers*, Patterns (2023)

## AI Tool Usage Disclosure
See [`AI_USAGE_DISCLOSURE.md`](AI_USAGE_DISCLOSURE.md)
```

---

### 10.8 AI Usage Disclosure (`AI_USAGE_DISCLOSURE.md`)

```markdown
# AI Tool Usage Disclosure

This project was developed with assistance from AI coding tools (Claude Code / Anthropic Claude).

## Where AI Was Used
- **Architecture & planning:** Initial design, module breakdown, implementation plan
- **Code generation:** Boilerplate for FastAPI, React components, signal extraction classes, classifier training loops
- **Documentation:** Drafting PROJECT.md, IMPLEMENTATION_PLAN.md, module instruction files, README
- **Debugging:** Interpreting error traces, suggesting fixes for PyTorch/transformers issues

## Where Human Judgment Prevailed
- **Scope decisions:** What to cut (DetectGPT, POS n-grams, neural classifier) and why
- **Dataset design:** Source selection, ESL subset methodology, hybrid essay generation prompts
- **Evaluation methodology:** Held-out-topic split, ESL bias measurement, wrong-example selection
- **Honest reporting:** Decision to prominently display ESL false-positive gap in UI
- **Code review:** All generated code reviewed, modified, and tested by human

## No AI Used For
- Final verdict on any essay (the classifier is trained on human-labeled data)
- Writing the three confidently-wrong example hypotheses (grounded in actual feature values)
- Determining the "definition of done" for each module
```

---

### 10.9 Smoke Test & Push

```bash
# 1. Full backend test
cd /path/to/project
python -c "
from src.classifier.predict import EssayClassifier
clf = EssayClassifier('models/logreg.joblib')
result = clf.predict_essay('I have always been fascinated by the stars. When I was five, my father bought me a telescope.')
print('Backend OK:', result[0]['ai_probability'])
"

# 2. API test
uvicorn src.api.main:app --port 8000 &
sleep 3
curl -X POST http://localhost:8000/analyze -H "Content-Type: application/json" -d '{"essay_text": "Test essay with multiple sentences. This should work fine."}'

# 3. Frontend test
cd src/frontend && npm run build  # Should compile without errors

# 4. Git commit & push
git add .
git commit -m "Complete AI Essay Detector: signals, classifier, API, UI, evaluation"
git push origin main
```

**Verify GitHub repo is public** (not fork).

---

## Definition of Done (Module 10 — PROJECT COMPLETE)
- [ ] `docs/evaluation.md` complete with all 4 required sections (confusion matrix, 3 wrong examples + hypotheses, held-out-topic drop, ESL FPR comparison)
- [ ] `README.md` with architecture, scope decisions, citations
- [ ] `AI_USAGE_DISCLOSURE.md` written
- [ ] `LimitationsPanel` shows real measured ESL FPR numbers
- [ ] End-to-end smoke test passes (backend + API + frontend build)
- [ ] Public GitHub repo pushed with all deliverables
- [ ] All checklist items from `docs/project2-ai-essay-detector.md` §12 satisfied

---

## Final Deliverables Checklist (from Project Doc §12)
- [ ] Public GitHub repo, pushed (not a fork)
- [ ] Working app: paste essay → highlighted output with explanations
- [ ] `docs/dataset.md` — sources, counts, coverage gaps, ESL subset methodology
- [ ] `docs/evaluation.md` — metrics, confusion matrix, 3 wrong examples + hypotheses, held-out-topic test, **ESL false-positive rate vs. general human rate**
- [ ] README with architecture overview, scope decisions, citations to prior-art detectors
- [ ] (Optional) short video walkthrough
- [ ] Honest AI-tool-usage disclosure

---

**🎉 PROJECT COMPLETE** — Ready for hackathon submission.