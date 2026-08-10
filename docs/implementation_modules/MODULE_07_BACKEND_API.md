# Module 7: Backend API (FastAPI)
**Day 9 — Task 9.1**  
**Estimated time:** 2–3 hours  
**Depends on:** Module 6 (`EssayClassifier`, `models/logreg.joblib`)

---

## Objective
Build a FastAPI backend with `/analyze` endpoint that accepts an essay, runs the full signal extraction + classification pipeline, and returns per-sentence results with offsets, probabilities, and plain-language reasons.

---

## Inputs Required
- `models/logreg.joblib` — trained classifier
- `src/classifier/predict.py` — `EssayClassifier` class
- `src/signals/extract.py` — `FeatureExtractor` (used internally by classifier)

---

## Outputs Produced
| Path | Description |
|------|-------------|
| `src/api/main.py` | FastAPI app with `/analyze` endpoint |
| `src/api/schemas.py` | Pydantic request/response models |
| `src/api/__init__.py` | Package exports |

---

## Step-by-Step Tasks

### 9.1 Pydantic Schemas (`src/api/schemas.py`)

```python
# src/api/schemas.py
from pydantic import BaseModel, Field
from typing import List, Optional, Literal

class AnalyzeRequest(BaseModel):
    essay_text: str = Field(..., min_length=50, max_length=10000, description="College admissions essay text")
    essay_id: Optional[str] = Field(None, description="Optional identifier for logging")
    topic: Optional[str] = Field(None, description="Essay topic/prompt if known")

class SentenceResult(BaseModel):
    sentence_idx: int
    text: str
    start_char: int
    end_char: int
    ai_probability: float = Field(..., ge=0.0, le=1.0)
    reasons: List[str] = Field(default_factory=list)

class EssaySummary(BaseModel):
    qualitative_band: Literal[
        "likely human-written",
        "mixed signals, possibly human with AI-like passages",
        "likely AI-assisted in places",
        "strongly indicative of AI generation"
    ]
    avg_ai_probability: float
    max_ai_probability: float
    high_ai_sentences: str  # e.g., "3/12"

class AnalyzeResponse(BaseModel):
    essay_id: str
    sentences: List[SentenceResult]
    summary: EssaySummary
    processing_time_ms: int
```

---

### 9.2 FastAPI App (`src/api/main.py`)

```python
# src/api/main.py
import time
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .schemas import AnalyzeRequest, AnalyzeResponse
from src.classifier.predict import EssayClassifier

# Global classifier instance (loaded at startup)
classifier: EssayClassifier = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global classifier
    print("Loading classifier...")
    classifier = EssayClassifier('models/logreg.joblib')
    print("Classifier loaded.")
    yield
    print("Shutting down.")

app = FastAPI(
    title="AI Essay Detector API",
    description="Signal-based AI detection for college admissions essays",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS for local frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": classifier is not None}

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    if classifier is None:
        raise HTTPException(503, "Classifier not loaded")
    
    start = time.perf_counter()
    essay_id = request.essay_id or str(uuid.uuid4())[:8]
    
    try:
        # Run classification
        sentence_results = classifier.predict_essay(
            request.essay_text,
            essay_id=essay_id,
            topic=request.topic or "unknown"
        )
        summary = classifier.summarize_essay(sentence_results)
        
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        
        return AnalyzeResponse(
            essay_id=essay_id,
            sentences=sentence_results,
            summary=summary,
            processing_time_ms=elapsed_ms,
        )
    except Exception as e:
        raise HTTPException(500, f"Analysis failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
```

---

### 9.3 Run & Test

```bash
# Start server
uvicorn src.api.main:app --reload --port 8000

# Test with curl
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "essay_text": "I have always been fascinated by the stars. When I was five, my father bought me a small telescope. We would spend hours on the backyard deck tracing constellations. Moreover, this experience taught me the value of patience and careful observation. As I reflect on those nights, I realize they shaped my academic path toward astrophysics.",
    "essay_id": "test_001",
    "topic": "stem_passion"
  }'
```

**Expected response:**
```json
{
  "essay_id": "test_001",
  "sentences": [
    {
      "sentence_idx": 0,
      "text": "I have always been fascinated by the stars.",
      "start_char": 0,
      "end_char": 43,
      "ai_probability": 0.12,
      "reasons": ["no strong signals"]
    },
    {
      "sentence_idx": 3,
      "text": "Moreover, this experience taught me the value of patience and careful observation.",
      "start_char": 124,
      "end_char": 208,
      "ai_probability": 0.78,
      "reasons": ["frequent use of stock AI transition phrases", "consistently high-probability token selections"]
    }
  ],
  "summary": {
    "qualitative_band": "likely AI-assisted in places",
    "avg_ai_probability": 0.34,
    "max_ai_probability": 0.78,
    "high_ai_sentences": "1/5"
  },
  "processing_time_ms": 2340
}
```

---

## Definition of Done (Module 7)
- [ ] `uvicorn src.api.main:app` starts without errors
- [ ] `/health` returns `{"status": "ok", "model_loaded": true}`
- [ ] `/analyze` accepts JSON, returns `AnalyzeResponse` with all fields
- [ ] Per-sentence results include: text, offsets, AI probability, reasons array
- [ ] Summary includes qualitative band (never bare percentage)
- [ ] Processing time < 10 seconds for 650-word essay on CPU
- [ ] CORS enabled for local frontend ports

---

## Handoff to Next Modules
- **Module 8** (Frontend Rendering) needs: `/analyze` endpoint working
- **Module 9** (Frontend Panels) needs: same endpoint