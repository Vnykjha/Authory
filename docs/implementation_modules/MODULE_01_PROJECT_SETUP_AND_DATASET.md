# Module 1: Project Setup & Dataset Sourcing
**Day 6 — Tasks 6.1–6.6**  
**Estimated time:** 4–6 hours  
**Can run in parallel with:** Module 2 (after 6.1 done)

---

## Objective
Create the project skeleton, install dependencies, and populate `data/` with four categories of essays (human-native, human-ESL, AI-generated, hybrid) — fully documented in `docs/dataset.md`.

---

## Inputs Required
- None (starts from empty repo with only `CLAUDE.md` and `docs/project2-ai-essay-detector.md`)

---

## Outputs Produced
| Path | Description |
|------|-------------|
| `pyproject.toml` | Project metadata, dependencies, entry points |
| `data/human_native/*.txt` | 60–100 human essays (native English) |
| `data/human_esl/*.txt` | 15–20 human essays (non-native English) |
| `data/ai_generated/*.txt` | 60–100 AI-generated essays (2+ models/prompts) |
| `data/hybrid/*.txt` | 20–30 hybrid essays (human draft + AI polish) |
| `docs/dataset.md` | Full provenance: sources, counts, topics, gaps |

---

## Step-by-Step Tasks

### 6.1 Project Structure & Dependencies
```bash
# Create directory layout
mkdir -p data/human_native data/human_esl data/ai_generated data/hybrid
mkdir -p src/signals src/classifier src/api src/frontend scripts models docs
```

**`pyproject.toml`** — include:
```toml
[project]
name = "ai-essay-detector"
version = "0.1.0"
description = "AI detector for college admissions essays — signal-based, not wrapper"
requires-python = ">=3.10"
dependencies = [
    "transformers>=4.40",
    "torch>=2.3",
    "scikit-learn>=1.4",
    "fastapi>=0.110",
    "uvicorn>=0.29",
    "pydantic>=2.7",
    "textstat>=0.7",
    "spacy>=3.7",
    "numpy>=1.26",
    "pandas>=2.2",
    "joblib>=1.4",
    "tqdm>=4.66",
]
# Optional: xgboost for stretch
# "xgboost>=2.0",

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.pytest.ini_options]
testpaths = ["tests"]
```

Run: `pip install -e .` and `python -m spacy download en_core_web_sm`

---

### 6.2 Source Human Essays (Native)
- Target: **60–100 essays**
- Sources: Public university example collections (e.g., Johns Hopkins "Essays That Worked", MIT admissions blog, Common App essay examples, etc.)
- Save each as `data/human_native/essay_XXX.txt`
- **Log in `docs/dataset.md`**: source URL, date accessed, essay count per source, prompt/topic if known

---

### 6.3 Source ESL Human Essays
- Target: **15–20 essays**
- Sources: TOEFL sample essays, IELTS writing samples, non-native student essay repositories
- Save as `data/human_esl/essay_XXX.txt`
- **Log separately in `docs/dataset.md`**: mark clearly as ESL subset; note geographic/language background if known

---

### 6.4 Generate AI Essays
- Target: **60–100 essays** across **≥2 models** and **≥2 prompt styles**
- Example models: GPT-3.5-turbo, GPT-4o, Claude-3-Haiku, Llama-3-8B-Instruct (via local or API)
- Prompt styles:
  1. "Write a college admissions essay about [topic]"
  2. "Write a personal statement for college application on [topic], 650 words"
  3. "Continue this essay opening: [first paragraph]"
- Save as `data/ai_generated/essay_XXX.txt`
- **Log in `docs/dataset.md`**: model, prompt template, temperature, topic per essay

---

### 6.5 Generate Hybrid Essays
- Target: **20–30 essays**
- Method: Take a subset of `human_native` essays → run through LLM with prompt: "Improve this college admissions essay for clarity, flow, and impact. Keep the same core story and voice."
- Save as `data/hybrid/essay_XXX.txt`
- **Log in `docs/dataset.md`**: source human essay ID, model used for polish, prompt

---

### 6.6 Document Dataset (`docs/dataset.md`)
Use this template:
```markdown
# Dataset Documentation

## Summary
| Category | Count | Source Details |
|---|---|---|
| Human (Native) | XX | [List sources + counts] |
| Human (ESL) | XX | [List sources + counts] |
| AI-Generated | XX | [Models + prompts + counts] |
| Hybrid | XX | [Source human IDs + polish model] |

## Human (Native) — Sources
- Source 1: URL, date, N essays, topics covered
- Source 2: ...

## Human (ESL) — Sources
- Source 1: URL, date, N essays, L1 languages represented

## AI-Generated — Generation Log
| Essay ID | Model | Prompt Style | Temperature | Topic |
|---|---|---|---|---|

## Hybrid — Generation Log
| Essay ID | Source Human ID | Polish Model | Prompt |

## Topics Covered
List all topics/themes across categories — enables held-out-topic split.

## Known Gaps
- e.g., "All human essays are US CommonApp style; no UK UCAS or other systems"
- e.g., "ESL subset drawn from East Asian L1 backgrounds only"
- e.g., "AI essays use only 2 models; may not generalize to others"
```

---

## Definition of Done (Module 1)
- [ ] `pyproject.toml` exists; `pip install -e .` succeeds; spaCy model downloaded
- [ ] Four `data/` subdirectories populated with target essay counts
- [ ] `docs/dataset.md` complete with all tables, sources, gaps documented
- [ ] Essay files are plain text, one essay per file, UTF-8 encoded
- [ ] No signal extraction or classifier code written yet

---

## Handoff to Next Modules
- **Module 2** needs: `pyproject.toml` installed, `data/` populated, `docs/dataset.md` done
- **Modules 3–5** need: `data/` + `docs/dataset.md` (for topic-aware splitting later)