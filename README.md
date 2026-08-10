# Authory — Signal-Based AI Detector for College Admissions Essays

> **Signal-based, not an LLM wrapper.** Authory runs essays through statistical language models (GPT-2 family) to extract objective language signals — single-model perplexity, Binoculars cross-perplexity, sentence burstiness, and lexical fingerprints — then classifies sentences using an interpretable Logistic Regression model trained on a documented dataset. No LLM is asked for a subjective verdict.

---

## 🏗️ System Architecture

```
                               ┌─────────────────────────────────────────────────────────┐
                               │                 AUTHORY PIPELINE                        │
                               └─────────────────────────────────────────────────────────┘
                                                           │
 ┌───────────────────────┐                                 ▼
 │  College Essay Input  │ ──────────► ┌─────────────────────────────────────────────────┐
 └───────────────────────┘             │  1. SPA-CY SENTENCE SEGMENTATION (Offsets)      │
                                       └─────────────────────────────────────────────────┘
                                                           │
                                                           ▼
                                       ┌─────────────────────────────────────────────────┐
                                       │  2. SIGNAL EXTRACTION LAYER (4 Signal Families)  │
                                       │    • Perplexity & Token Ranks (GPT-2)           │
                                       │    • Cross-Perplexity Ratio (Binoculars)        │
                                       │    • Sentence Burstiness & Perplexity Variance  │
                                       │    • Lexical Richness (TTR, MTLD, AI Phrases)   │
                                       └─────────────────────────────────────────────────┘
                                                           │
                                                           ▼
                                       ┌─────────────────────────────────────────────────┐
                                       │  3. LOGISTIC REGRESSION CLASSIFIER              │
                                       │     (Topic-Aware Split, Balanced Weights)       │
                                       └─────────────────────────────────────────────────┘
                                                           │
                                                           ▼
                                       ┌─────────────────────────────────────────────────┐
                                       │  4. INTERACTIVE REACT + TAILWIND FRONTEND       │
                                       │    • Sentence Highlights (Intensity = AI prob)  │
                                       │    • Plain-Language Explanation Drawer          │
                                       │    • Qualitative Summary Bands (No bare %)     │
                                       │    • Empirical ESL False-Positive Bias Modal    │
                                       └─────────────────────────────────────────────────┘
```

---

## 📊 Signal Extraction Layer (No Verdicts)

| Signal Family | Scientific Basis | What It Measures |
|---|---|---|
| **Single-Model Perplexity & Ranks** | GPTZero | Measures word predictability (`ppl_perplexity`) and frequency of top-1 model predictions (`ppl_pct_rank_1`). |
| **Cross-Perplexity Ratio (Binoculars)** | Hans et al. (ICML 2024) | Ratio between GPT-2 observer and GPT-2-medium performer. Captures cross-model divergence; more robust across writing styles. |
| **Burstiness & Sentence Rhythm** | GPTZero | Coefficient of variation (CV) and IQR of sentence lengths and perplexities across local context windows. Human writing varies; AI is uniform. |
| **Lexical & Phrasal Fingerprints** | Stylometry research | Type-Token Ratio (TTR), MTLD richness, and frequency rate of stock AI transition phrases (*"moreover"*, *"delve"*, *"tapestry"*). |

---

## 🎯 Model Training & Topic-Aware Evaluation

- **Model:** Logistic Regression with balanced class weights (`models/logreg.joblib`).
- **Dataset:** 200 essays (**3,857 sentence records**) across `human_native` (80), `human_esl` (20), `ai_generated` (80), and `hybrid` (20).
- **Split Strategy:** **Held-out topic split** (`urban_planning` & `technology_communication` held out for testing) to prevent vocabulary memorization. ESL essays are strictly isolated from training for bias testing.

### Held-Out Topic Results
- **AI Detection Sensitivity (Recall):** **98.5%** ($67/68$ AI sentences correctly caught).
- **Native English Specificity:** **100.0%** ($0/246$ native human sentences misclassified, $0.0\%$ False Positive Rate).

### Empirical ESL Bias Measurement
- **Native Human False-Positive Rate:** **0.0%** ($0/246$)
- **ESL Human False-Positive Rate:** **43.6%** ($153/351$)
- *Mechanism:* Non-native writers naturally use simpler vocabulary and standard exam templates. Perplexity detectors find standard word choices predictable, misidentifying them as AI. This gap is prominently disclosed in the Authory UI.

---

## ⚡ Quick Start

### 1. Backend API (FastAPI)
```bash
# Install dependencies
pip install -e .

# Start FastAPI backend server
python -m uvicorn src.api.main:app --port 8000 --reload
```
API runs at `http://localhost:8000` (docs at `http://localhost:8000/docs`).

### 2. Frontend Application (React + Vite + Tailwind)
```bash
# In a new terminal:
cd src/frontend
npm install
npm run dev
```
Open `http://localhost:5173` in your browser.

---

## 🛠️ Scope Decisions (What Was Cut & Why)

| Cut Feature | Reason |
|---|---|
| **DetectGPT Perturbation Curvature** | Requires dozens of model re-scoring passes per essay. Too compute-heavy for real-time API; Binoculars cross-perplexity captures similar signals at a fraction of compute. |
| **POS N-Gram Parsing** | Higher complexity with marginal accuracy gains; engineering time prioritized on ESL bias evaluation. |
| **Deep Neural Black-Box Classifier** | Replaced with Logistic Regression to ensure linear feature coefficients map directly to plain-language explanations. |
| **Random Train/Test Split** | Random sentence splitting inflates accuracy by data leakage. Topic-aware splitting guarantees honest generalization. |
| **Bare Percentage Verdicts** | Prohibited to prevent false accusations. Authory uses qualitative summary bands (*"Likely AI-Assisted in Places"*). |

---

## 📚 Citations & Prior Art

- **Binoculars:** Hans et al., *Spotting LLMs With Binoculars: Zero-Shot Detection of Machine-Generated Text*, ICML 2024.
- **DetectGPT:** Mitchell et al., *DetectGPT: Zero-Shot Machine-Generated Text Detection using Probability Curvature*, ICML 2023.
- **ESL Detector Bias:** Liang et al., *GPT detectors are biased against non-native English writers*, Patterns (Cell Press), 2023.
- **GPTZero Principles:** Tian, E., *Perplexity and Burstiness as Detection Signals*, 2023.

---

## 📄 AI Usage Disclosure
See [`AI_USAGE_DISCLOSURE.md`](AI_USAGE_DISCLOSURE.md) for full disclosure of AI tool usage during development.
