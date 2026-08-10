"""
Signal extraction package for AI Essay Detector.
Provides perplexity computation, cross-perplexity (Binoculars), sentence segmentation,
burstiness features, and lexical/phrasal fingerprints.
"""

from .perplexity import PerplexityComputer
from .cross_perplexity import CrossPerplexityComputer
from .segment import segment_sentences, get_nlp
from .burstiness import (
    sentence_lengths,
    perplexity_burstiness,
    length_burstiness,
    compute_burstiness_features,
    sentence_context_burstiness,
)
from .lexical import (
    type_token_ratio,
    mtld_score,
    ai_phrase_count,
    ai_phrase_count_per_sentence,
    lexical_diversity_per_sentence,
    compute_lexical_features,
    compute_lexical_features_per_sentence,
    AI_TRANSITION_PHRASES,
)

__all__ = [
    "PerplexityComputer",
    "CrossPerplexityComputer",
    "segment_sentences",
    "get_nlp",
    "sentence_lengths",
    "perplexity_burstiness",
    "length_burstiness",
    "compute_burstiness_features",
    "sentence_context_burstiness",
    "type_token_ratio",
    "mtld_score",
    "ai_phrase_count",
    "ai_phrase_count_per_sentence",
    "lexical_diversity_per_sentence",
    "compute_lexical_features",
    "compute_lexical_features_per_sentence",
    "AI_TRANSITION_PHRASES",
]