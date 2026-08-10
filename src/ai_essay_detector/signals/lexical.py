"""
Lexical & phrasal fingerprint features:
- Type-Token Ratio (TTR)
- MTLD (Measure of Textual Lexical Diversity)
- AI Transition Phrase Frequency (stock phrases LLMs overuse)
"""

from typing import List, Tuple, Dict, Set
import re
from collections import Counter

from .segment import segment_sentences


# Stock AI transition phrases (expand based on observation)
AI_TRANSITION_PHRASES = [
    "moreover", "furthermore", "additionally", "consequently",
    "therefore", "thus", "hence", "accordingly",
    "in conclusion", "to conclude", "ultimately",
    "this experience taught me", "i learned that", "it taught me",
    "as i reflect", "looking back", "in retrospect",
    "it is important to note", "notably", "significantly",
    "delve", "tapestry", "landscape", "realm", "journey",
    "embark", "navigate", "unveil", "unravel", "foster",
    "crucial", "pivotal", "instrumental", "paramount",
    "holistic", "multifaceted", "nuanced", "profound",
]

# Compile regex for phrase matching (case-insensitive, word boundaries)
_PHRASE_PATTERNS = [re.compile(rf'\b{re.escape(phrase)}\b', re.IGNORECASE) for phrase in AI_TRANSITION_PHRASES]


def _tokenize_words(text: str) -> List[str]:
    """Extract word tokens from text (lowercase)."""
    return re.findall(r'\b\w+\b', text.lower())


def type_token_ratio(text: str) -> float:
    """
    Simple TTR: unique word types / total word tokens.

    Args:
        text: Input text.

    Returns:
        TTR value between 0 and 1.
    """
    words = _tokenize_words(text)
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def mtld_score(text: str, threshold: float = 0.72) -> float:
    """
    Compute MTLD (Measure of Textual Lexical Diversity).

    MTLD calculates the mean length of sequential word strings that maintain
    a TTR above a threshold (default 0.72). Higher MTLD = more lexical diversity.

    Args:
        text: Input text.
        threshold: TTR threshold (default 0.72 per McCarthy & Jarvis 2010).

    Returns:
        MTLD score (typically 0-100+, higher = more diverse).
    """
    words = _tokenize_words(text)
    n_words = len(words)

    if n_words == 0:
        return 0.0

    def count_factors(word_list: List[str]) -> float:
        """Count factors (segments) in one direction."""
        if not word_list:
            return 0.0

        factors = 0.0
        types = set()
        token_count = 0

        for word in word_list:
            types.add(word)
            token_count += 1
            ttr = len(types) / token_count

            if ttr < threshold:
                factors += 1.0
                types = {word}
                token_count = 1

        # Add partial factor for remaining words
        if token_count > 0:
            # Partial factor contribution
            current_ttr = len(types) / token_count
            if current_ttr >= threshold:
                factors += 1.0
            else:
                # Proportional contribution
                factors += current_ttr / threshold

        return factors

    # Forward pass
    forward_factors = count_factors(words)

    # Backward pass
    backward_factors = count_factors(list(reversed(words)))

    # Average factors
    avg_factors = (forward_factors + backward_factors) / 2.0

    if avg_factors == 0:
        return 0.0

    # MTLD = total words / average factors
    mtld = n_words / avg_factors
    return float(mtld)


def ai_phrase_count(text: str) -> int:
    """
    Count occurrences of stock AI transition phrases.

    Args:
        text: Input text.

    Returns:
        Total count of matched AI phrases.
    """
    count = 0
    for pattern in _PHRASE_PATTERNS:
        count += len(pattern.findall(text))
    return count


def ai_phrase_count_per_sentence(text: str, spans: List[Tuple[str, int, int]] = None) -> List[int]:
    """
    AI phrase count per sentence.

    Args:
        text: Input text.
        spans: Optional precomputed sentence spans.

    Returns:
        List of AI phrase counts per sentence.
    """
    if spans is None:
        spans = segment_sentences(text)
    counts = []
    for _, start, end in spans:
        sent_text = text[start:end]
        counts.append(ai_phrase_count(sent_text))
    return counts


def lexical_diversity_per_sentence(text: str, spans: List[Tuple[str, int, int]] = None) -> List[float]:
    """
    TTR per sentence (MTLD needs longer text, so we use TTR for per-sentence).

    Args:
        text: Input text.
        spans: Optional precomputed sentence spans.

    Returns:
        List of TTR values per sentence.
    """
    if spans is None:
        spans = segment_sentences(text)
    ttrs = []
    for _, start, end in spans:
        sent_text = text[start:end]
        ttrs.append(type_token_ratio(sent_text))
    return ttrs


def compute_lexical_features(text: str, spans: List[Tuple[str, int, int]] = None) -> Dict[str, float]:
    """
    Essay-level lexical features.

    Args:
        text: Full essay text.
        spans: Optional precomputed sentence spans.

    Returns:
        Flat dict with:
        - 'lex_ttr': Type-token ratio for full essay
        - 'lex_mtld': MTLD score for full essay
        - 'lex_ai_phrase_count': Total AI phrase count
        - 'lex_ai_phrase_rate': AI phrases per sentence
    """
    if spans is None:
        spans = segment_sentences(text)

    return {
        'lex_ttr': type_token_ratio(text),
        'lex_mtld': mtld_score(text),
        'lex_ai_phrase_count': ai_phrase_count(text),
        'lex_ai_phrase_rate': ai_phrase_count(text) / max(1, len(spans)),  # per sentence
    }


def compute_lexical_features_per_sentence(text: str, spans: List[Tuple[str, int, int]] = None) -> List[Dict]:
    """
    Per-sentence lexical features for classifier.

    Args:
        text: Full essay text.
        spans: Optional precomputed sentence spans.

    Returns:
        List of dicts per sentence:
        {
            'sentence_idx': int,
            'lex_ttr': float,
            'lex_ai_phrase_count': int,
        }
    """
    if spans is None:
        spans = segment_sentences(text)

    ttrs = lexical_diversity_per_sentence(text, spans)
    phrase_counts = ai_phrase_count_per_sentence(text, spans)

    results = []
    for i, (ttr, phrases) in enumerate(zip(ttrs, phrase_counts)):
        results.append({
            'sentence_idx': i,
            'lex_ttr': ttr,
            'lex_ai_phrase_count': phrases,
        })
    return results