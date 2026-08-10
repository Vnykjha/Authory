"""
Classifier module package for AI Essay Detector.
Provides train/test splitting, Logistic Regression classifier training,
inference engine, and qualitative summary generation.
"""

from .split import held_out_topic_split
from .train import train_logreg, evaluate_esl_fpr, save_model, prepare_features
from .predict import EssayClassifier, REASON_MAP

__all__ = [
    "held_out_topic_split",
    "train_logreg",
    "evaluate_esl_fpr",
    "save_model",
    "prepare_features",
    "EssayClassifier",
    "REASON_MAP",
]
