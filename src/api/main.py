"""
FastAPI application for Authory AI Detector backend.
Exposes /health and /analyze endpoints.
"""

import time
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from .schemas import AnalyzeRequest, AnalyzeResponse, HealthResponse, SentenceResult, EssaySummary
from src.classifier.predict import EssayClassifier

# Global classifier instance (loaded at startup)
classifier: EssayClassifier = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global classifier
    print("Loading Authory classifier pipeline...")
    try:
        classifier = EssayClassifier('models/logreg.joblib')
        print("Authory classifier successfully loaded into memory.")
    except Exception as e:
        print(f"Warning: Failed to load classifier model: {e}")
        classifier = None
    yield
    print("Shutting down Authory API.")


app = FastAPI(
    title="Authory API",
    description="Signal-based AI detection and transparency engine for college admissions essays",
    version="0.1.0",
    lifespan=lifespan,
)

# Enable CORS for local frontend dev servers (React, Vite, Next.js)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint confirming API status and model availability."""
    return HealthResponse(
        status="ok",
        model_loaded=classifier is not None
    )


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    """
    Analyze essay text and return sentence-level AI probabilities, character offsets,
    plain-language signal explanations, and qualitative verdict summary.
    """
    if classifier is None:
        raise HTTPException(
            status_code=status.HTTP_533_SERVICE_UNAVAILABLE,
            detail="Classifier model is not loaded. Train model via python -m src.classifier.train"
        )

    start_time = time.perf_counter()
    essay_id = request.essay_id or f"essay_{str(uuid.uuid4())[:8]}"

    try:
        # Run sentence-level prediction
        sentence_results = classifier.predict_essay(
            essay_text=request.essay_text,
            essay_id=essay_id,
            topic=request.topic or "unknown"
        )

        # Generate qualitative summary band
        summary_dict = classifier.summarize_essay(sentence_results)

        elapsed_ms = int((time.perf_counter() - start_time) * 1000)

        formatted_sentences = [
            SentenceResult(
                sentence_idx=r['sentence_idx'],
                text=r['text'],
                start_char=r['start_char'],
                end_char=r['end_char'],
                ai_probability=r['ai_probability'],
                reasons=r['reasons']
            )
            for r in sentence_results
        ]

        formatted_summary = EssaySummary(
            qualitative_band=summary_dict['qualitative_band'],
            summary_description=summary_dict['summary_description'],
            avg_ai_probability=summary_dict['avg_ai_probability'],
            max_ai_probability=summary_dict['max_ai_probability'],
            high_ai_sentences_count=summary_dict['high_ai_sentences_count'],
            total_sentences=summary_dict['total_sentences']
        )

        return AnalyzeResponse(
            essay_id=essay_id,
            sentences=formatted_sentences,
            summary=formatted_summary,
            processing_time_ms=elapsed_ms
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Essay analysis failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
