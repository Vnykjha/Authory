"""
Pydantic schemas for Authory API request and response validation.
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class AnalyzeRequest(BaseModel):
    essay_text: str = Field(..., min_length=20, max_length=15000, description="College admissions essay text to analyze")
    essay_id: Optional[str] = Field(None, description="Optional essay identifier for tracking")
    topic: Optional[str] = Field(None, description="Optional essay prompt or topic category")


class SentenceResult(BaseModel):
    sentence_idx: int = Field(..., description="0-indexed sentence position")
    text: str = Field(..., description="Sentence text content")
    start_char: int = Field(..., description="Starting character offset in original text")
    end_char: int = Field(..., description="Ending character offset in original text (exclusive)")
    ai_probability: float = Field(..., ge=0.0, le=1.0, description="Predicted AI probability between 0.0 and 1.0")
    reasons: List[str] = Field(default_factory=list, description="Top plain-language signal explanations")


class EssaySummary(BaseModel):
    qualitative_band: str = Field(..., description="Qualitative verdict band (no bare percentage)")
    summary_description: str = Field(..., description="Plain-language description of statistical findings")
    avg_ai_probability: float = Field(..., ge=0.0, le=1.0, description="Average sentence AI probability")
    max_ai_probability: float = Field(..., ge=0.0, le=1.0, description="Peak sentence AI probability")
    high_ai_sentences_count: int = Field(..., description="Count of sentences with AI probability >= 0.70")
    total_sentences: int = Field(..., description="Total sentences analyzed")


class AnalyzeResponse(BaseModel):
    essay_id: str = Field(..., description="Essay tracking identifier")
    sentences: List[SentenceResult] = Field(..., description="Per-sentence analysis results")
    summary: EssaySummary = Field(..., description="Overall qualitative essay summary")
    processing_time_ms: int = Field(..., description="Total processing time in milliseconds")


class HealthResponse(BaseModel):
    status: str = Field("ok", description="API health status")
    model_loaded: bool = Field(..., description="Whether classifier model pipeline is loaded")
