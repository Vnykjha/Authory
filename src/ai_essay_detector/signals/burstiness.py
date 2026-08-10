"""
Burstiness features: sentence-length and perplexity variance.
Human writing varies in sentence length and "surprise" (perplexity); AI writing stays in a narrow band.
"""

from typing import List, Tuple, Dict
import numpy as np
from .segment import segment_sentences


def sentence_lengths(text: str, spans: List[Tuple[str, int, int]] = None) -> List[int]:
    """
    Character length of each sentence.

    Args:
        text: Input text.
        spans: Optional precomputed sentence spans from segment_sentences().

    Returns:
        List of character lengths per sentence.
    """
    if spans is None:
        spans = segment_sentences(text)
    return [end - start for _, start, end in spans]


def perplexity_burstiness(perplexities: List[float]) -> Dict[str, float]:
    """
    Compute burstiness metrics from per-sentence perplexities.

    Args:
        perplexities: List of perplexity values per sentence.

    Returns:
        Dict with:
        - 'ppl_mean': mean perplexity
        - 'ppl_std': standard deviation
        - 'ppl_cv': coefficient of variation (std/mean) — primary burstiness signal
        - 'ppl_iqr': interquartile range (75th - 25th percentile)
    """
    arr = np.array(perplexities, dtype=float)
    if len(arr) == 0:
        return {
            'ppl_mean': 0.0,
            'ppl_std': 0.0,
            'ppl_cv': 0.0,
            'ppl_iqr': 0.0,
        }

    mean = float(arr.mean())
    std = float(arr.std())
    return {
        'ppl_mean': mean,
        'ppl_std': std,
        'ppl_cv': float(std / mean) if mean > 0 else 0.0,
        'ppl_iqr': float(np.percentile(arr, 75) - np.percentile(arr, 25)),
    }


def length_burstiness(lengths: List[int]) -> Dict[str, float]:
    """
    Compute burstiness metrics from sentence lengths (characters or words).

    Args:
        lengths: List of sentence lengths.

    Returns:
        Dict with same structure as perplexity_burstiness.
    """
    arr = np.array(lengths, dtype=float)
    if len(arr) == 0:
        return {
            'len_mean': 0.0,
            'len_std': 0.0,
            'len_cv': 0.0,
            'len_iqr': 0.0,
        }

    mean = float(arr.mean())
    std = float(arr.std())
    return {
        'len_mean': mean,
        'len_std': std,
        'len_cv': float(std / mean) if mean > 0 else 0.0,
        'len_iqr': float(np.percentile(arr, 75) - np.percentile(arr, 25)),
    }


def compute_burstiness_features(
    text: str,
    sentence_perplexities: List[float] = None,
    spans: List[Tuple[str, int, int]] = None
) -> Dict[str, float]:
    """
    Combined burstiness features for an essay.

    Args:
        text: Full essay text.
        sentence_perplexities: Optional per-sentence perplexity values.
        spans: Optional precomputed sentence spans.

    Returns:
        Flat dict of all features with 'burst_' prefix.
    """
    if spans is None:
        spans = segment_sentences(text)

    lengths = sentence_lengths(text, spans)
    len_feats = length_burstiness(lengths)

    feats = {f'burst_{k}': v for k, v in len_feats.items()}

    if sentence_perplexities is not None and len(sentence_perplexities) == len(spans):
        ppl_feats = perplexity_burstiness(sentence_perplexities)
        feats.update({f'burst_{k}': v for k, v in ppl_feats.items()})

    return feats


def sentence_context_burstiness(
    sentence_perplexities: List[float],
    window: int = 2
) -> List[Dict]:
    """
    Compute burstiness in a sliding window around each sentence.

    Args:
        sentence_perplexities: List of perplexity values per sentence.
        window: Number of sentences on each side to include (default: 2).

    Returns:
        List of dicts per sentence with local CV/IQR:
        {
            'sentence_idx': int,
            'local_ppl_cv': float,
            'local_ppl_iqr': float,
        }
    """
    results = []
    n = len(sentence_perplexities)
    for i in range(n):
        start = max(0, i - window)
        end = min(n, i + window + 1)
        window_ppls = sentence_perplexities[start:end]
        arr = np.array(window_ppls, dtype=float)

        mean = float(arr.mean()) if len(arr) > 0 else 0.0
        std = float(arr.std()) if len(arr) > 0 else 0.0

        results.append({
            'sentence_idx': i,
            'local_ppl_cv': float(std / mean) if mean > 0 else 0.0,
            'local_ppl_iqr': float(np.percentile(arr, 75) - np.percentile(arr, 25)) if len(arr) > 0 else 0.0,
        })
    return results