# Implementation Modules Index
**AI Essay Detector — 5-Day Build, 10 Modular Work Units**

This index maps each module to its day, dependencies, and suggested agent assignment. Each module is a **self-contained instruction file** in `docs/implementation_modules/` that an AI agent can execute independently.

---

## Module Overview

| Module | File | Day | Focus | Est. Time | Depends On |
|---|---|---|---|---|---|
| **01** | `MODULE_01_PROJECT_SETUP_AND_DATASET.md` | 6 | Project skeleton + dataset sourcing (4 categories) | 4–6h | — |
| **02** | `MODULE_02_PERPLEXITY_AND_SEGMENTATION.md` | 6 | GPT-2 perplexity + sentence segmentation | 2–3h | 01 (partial: 6.1) |
| **03** | `MODULE_03_CROSS_PERPLEXITY.md` | 7 | Second model (Binoculars-style cross-perplexity) | 2–3h | 02 |
| **04** | `MODULE_04_BURSTINESS_AND_LEXICAL.md` | 7 | Burstiness + lexical diversity + AI phrase detection | 2–3h | 02 |
| **05** | `MODULE_05_FEATURE_PIPELINE_AND_DATASET_RUN.md` | 7 | Orchestrate all signals → `features.csv` | 3–4h | 02, 03, 04 |
| **06** | `MODULE_06_CLASSIFIER_TRAINING_AND_EVALUATION.md` | 8 | LogReg training, held-out-topic eval, ESL FPR | 4–5h | 05 |
| **07** | `MODULE_07_BACKEND_API.md` | 9 | FastAPI `/analyze` endpoint | 2–3h | 06 |
| **08** | `MODULE_08_FRONTEND_RENDERING_AND_HIGHLIGHTING.md` | 9 | React + Tailwind: paste box → highlighted essay | 3–4h | 07 |
| **09** | `MODULE_09_FRONTEND_EXPLANATION_AND_LIMITATIONS.md` | 9 | Explanation panel + Limitations panel (ESL bias) | 2–3h | 08 |
| **10** | `MODULE_10_FINAL_EVALUATION_AND_POLISH.md` | 10 | Final eval doc, README, AI disclosure, push | 4–5h | 06, 07, 08, 09 |

---

## Dependency Graph

```
Day 6:  MODULE_01 ──────────┐
         │                  │
         ▼                  ▼
      MODULE_02 ◄──────────┘  (after 01.6.1 done)
      
Day 7:  MODULE_03 ──┐
         │          │      (can run in parallel)
         ▼          ▼
      MODULE_04    MODULE_05 ◄────── requires 03 + 04 complete
      
Day 8:  MODULE_06 ◄────────────── requires 05
      
Day 9:  MODULE_07 ◄────────────── requires 06
         │
         ▼
      MODULE_08 ──┐
         │        │      (can run in parallel)
         ▼        ▼
      MODULE_09 ◄──┘       requires 08
      
Day 10: MODULE_10 ◄────────── requires all prior
```

---

## Suggested Agent Assignment (Parallel Windows)

### Window A: Data & Signals (Day 6–7)
```
Agent A1 → MODULE_01 (Project Setup + Dataset)
Agent A2 → MODULE_02 (Perplexity + Segmentation)  [start after A1 completes 6.1]
Agent A3 → MODULE_03 (Cross-Perplexity)           [start after A2 done]
Agent A4 → MODULE_04 (Burstiness + Lexical)       [start after A2 done]
Agent A5 → MODULE_05 (Feature Pipeline)           [start after A3 + A4 done]
```

### Window B: Classifier & API (Day 8–9)
```
Agent B1 → MODULE_06 (Classifier Training + Eval) [after MODULE_05]
Agent B2 → MODULE_07 (Backend API)                [after MODULE_06]
```

### Window C: Frontend (Day 9)
```
Agent C1 → MODULE_08 (Frontend Rendering)         [after MODULE_07]
Agent C2 → MODULE_09 (Explanation + Limitations)  [after MODULE_08 starts]
```

### Window D: Final Polish (Day 10)
```
Agent D1 → MODULE_10 (Evaluation + README + Push) [after all above]
```

---

## How to Use

1. **Copy a module file's content** into a new Claude Code window/tab
2. **The agent reads the module** and executes all tasks
3. **Agent stops at Definition of Done** — you verify, then proceed
4. **Handoff outputs** (files created) to dependent modules

Each module file contains:
- ✅ Clear objective
- ✅ Required inputs (from prior modules)
- ✅ Exact outputs to produce (file paths)
- ✅ Step-by-step implementation tasks with code skeletons
- ✅ Quick test scripts
- ✅ **Definition of Done** checklist
- ✅ Handoff notes for next modules

---

## Daily Review Checkpoints

Per `CLAUDE.md` and `IMPLEMENTATION_PLAN.md`: **Stop at end of each day (6–10) for review.**

| Day | Modules to Complete | Review Before Next Day |
|---|---|---|
| 6 | 01, 02 | `data/` populated, `docs/dataset.md` done, perplexity working |
| 7 | 03, 04, 05 | `data/features.csv` exists, all signals extracted, EDA shows separation |
| 8 | 06 | Model trained, held-out-topic metrics, ESL FPR measured |
| 9 | 07, 08, 09 | End-to-end app works: paste → highlights → explanations → limitations |
| 10 | 10 | All docs final, real ESL numbers in UI, repo public |

---

## Quick Start for First Agent (Module 01)

```bash
# In a new Claude Code window:
# 1. Read the module file
cat docs/implementation_modules/MODULE_01_PROJECT_SETUP_AND_DATASET.md

# 2. Execute tasks in order
# 3. Report completion with Definition of Done checklist
```

---

## Module Files List

```
docs/implementation_modules/
├── MODULE_01_PROJECT_SETUP_AND_DATASET.md
├── MODULE_02_PERPLEXITY_AND_SEGMENTATION.md
├── MODULE_03_CROSS_PERPLEXITY.md
├── MODULE_04_BURSTINESS_AND_LEXICAL.md
├── MODULE_05_FEATURE_PIPELINE_AND_DATASET_RUN.md
├── MODULE_06_CLASSIFIER_TRAINING_AND_EVALUATION.md
├── MODULE_07_BACKEND_API.md
├── MODULE_08_FRONTEND_RENDERING_AND_HIGHLIGHTING.md
├── MODULE_09_FRONTEND_EXPLANATION_AND_LIMITATIONS.md
└── MODULE_10_FINAL_EVALUATION_AND_POLISH.md
```

All 10 modules created and ready for distribution to agent windows.