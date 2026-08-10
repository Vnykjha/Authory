"""
Sentence segmentation using spaCy.
Returns sentences with character offsets for UI highlighting.
"""

import spacy
from typing import List, Tuple

_nlp = None


def get_nlp() -> spacy.Language:
    """
    Load and return the spaCy NLP pipeline with sentencizer.
    Uses lazy loading to avoid import-time overhead.
    """
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
        # Use sentencizer for faster sentence boundary detection
        # (parser is slower and not needed for just boundaries)
        if "sentencizer" not in _nlp.pipe_names:
            _nlp.add_pipe("sentencizer")
    return _nlp


def segment_sentences(text: str) -> List[Tuple[str, int, int]]:
    """
    Split text into sentences with character offsets.

    Args:
        text: Input text to segment.

    Returns:
        List of (sentence_text, start_char, end_char) tuples.
        end_char is exclusive (like Python slicing).
    """
    if not text or not text.strip():
        return []

    doc = get_nlp()(text)
    spans = []
    for sent in doc.sents:
        # sent.text preserves original whitespace within the sentence
        # sent.start_char and sent.end_char are character offsets in the original text
        spans.append((sent.text, sent.start_char, sent.end_char))
    return spans