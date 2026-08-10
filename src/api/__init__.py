"""
Authory API package.
Exposes FastAPI application and Pydantic schemas.
"""

from .schemas import AnalyzeRequest, AnalyzeResponse, HealthResponse, SentenceResult, EssaySummary
from .main import app

__all__ = [
    "app",
    "AnalyzeRequest",
    "AnalyzeResponse",
    "HealthResponse",
    "SentenceResult",
    "EssaySummary",
]
