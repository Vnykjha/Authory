# Module 4: Burstiness & Lexical / Phrasal Signals
**Day 7 — Tasks 7.2–7.3**  
**Estimated time:** 2–3 hours  
**Depends on:** Module 2 (`segment_sentences`)  
**Can run in parallel with:** Module 3 (Cross-Perplexity)

---

## Objective
Implement burstiness (sentence-length & perplexity variance) and lexical/phrasal fingerprint features — the stylometric signals that complement perplexity.

---

## Inputs Required
- `src/signals/segment.py` with `segment_sentences`
- (Optional) `src/signals/perplexity.py` for per-sentence perplexity values

---

## Outputs Produced
| Path | Description |
|------|-------------|
| `src/signals/burstiness.py` | Burstiness features (sentence length + perplexity variance) |
| `src/signals/lexical.py` | Lexical diversity (TTR, MTLD) + AI transition phrase counts |
| Updated `src/signals/__init__.py` | Export new functions/classes |

---

## Step-by-Step Tasks

### 7.2 Burstiness (`src/signals/burstiness.py`)

**Concept:** Human writing varies in sentence length and "surprise" (perplexity); AI writing stays in a narrow band.

```python
# src/signals/burstiness.py
from typing import List, Tuple
import numpy as np
from .segment import segment_sentences

def sentence_lengths(text: str, spans: List[Tuple[int, int]] = None) -> List[int]:
    """Character length of each sentence."""
    if spans is None:
        spans = segment_sentences(text)
    return [end - start for _, start, end in spans]

def perplexity_burstiness(perplexities: List[float]) -> dict:
    """
    Compute burstiness metrics from per-sentence perplexities.
    Returns dict with:
    - 'ppl_mean': mean perplexity
    - 'ppl_std': std deviation
    - 'ppl_cv': coefficient of variation (std/mean) — primary burstiness signal
    - 'ppl_iqr': interquartile range
    """
    arr = np.array(perplexities)
    return {
        'ppl_mean': float(arr.mean()),
        'ppl_std': float(arr.std()),
        'ppl_cv': float(arr.std() / arr.mean()) if arr.mean() > 0 else 0.0,
        'ppl_iqr': float(np.percentile(arr, 75) - np.percentile(arr, 25)),
    }

def length_burstiness(lengths: List[int]) -> dict:
    """
    Compute burstiness metrics from sentence lengths (characters or words).
    Returns same structure as perplexity_burstiness.
    """
    arr = np.array(lengths, dtype=float)
    return {
        'len_mean': float(arr.mean()),
        'len_std': float(arr.std()),
        'len_cv': float(arr.std() / arr.mean()) if arr.mean() > 0 else 0.0,
        'len_iqr': float(np.percentile(arr, 75) - np.percentile(arr, 25)),
    }

def compute_burstiness_features(
    text: str,
    sentence_perplexities: List[float] = None,
    spans: List[Tuple[int, int]] = None
) -> dict:
    """
    Combined burstiness features for an essay.
    If sentence_perplexities provided, includes perplexity burstiness.
    Always includes length burstiness.
    Returns flat dict of all features.
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
```

**Per-sentence context window:** For the classifier, also compute burstiness over a sliding window of ±2 sentences around each target sentence (helps localize AI spans).

```python
def sentence_context_burstiness(
    sentence_perplexities: List[float],
    window: int = 2
) -> List[dict]:
    """
    Compute burstiness in a window around each sentence.
    Returns list of dicts per sentence with local CV/IQR.
    """
    results = []
    n = len(sentence_perplexities)
    for i in range(n):
        start = max(0, i - window)
        end = min(n, i + window + 1)
        window_ppls = sentence_perplexities[start:end]
        arr = np.array(window_ppls)
        results.append({
            'sentence_idx': i,
            'local_ppl_cv': float(arr.std() / arr.mean()) if arr.mean() > 0 else 0.0,
            'local_ppl_iqr': float(np.percentile(arr, 75) - np.percentile(arr, 25)),
        })
    return results
```

---

### 7.3 Lexical & Phrasal Fingerprints (`src/signals/lexical.py`)

**Features:**
1. **Type-Token Ratio (TTR)** — unique words / total words
2. **MTLD (Measure of Textual Lexical Diversity)** — more robust than TTR for varying lengths
3. **AI Transition Phrase Frequency** — count of stock phrases LLMs overuse

```python
# src/signals/lexical.py
from typing import List, Tuple, Set
import re
from collections import Counter
import textstat  # for MTLD
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

def type_token_ratio(text: str) -> float:
    """Simple TTR: unique word types / total word tokens."""
    words = re.findall(r'\b\w+\b', text.lower())
    if not words:
        return 0.0
    return len(set(words)) / len(words)

def mtld_score(text: str) -> float:
    """MTLD via textstat (handles edge cases)."""
    try:
        return textstat.mtld(text)
    except:
        return 0.0

def ai_phrase_count(text: str) -> int:
    """Count occurrences of stock AI transition phrases."""
    count = 0
    for pattern in _PHRASE_PATTERNS:
        count += len(pattern.findall(text))
    return count

def ai_phrase_count_per_sentence(text: str, spans: List[Tuple[int, int]] = None) -> List[int]:
    """AI phrase count per sentence."""
    if spans is None:
        spans = segment_sentences(text)
    counts = []
    for _, start, end in spans:
        sent_text = text[start:end]
        counts.append(ai_phrase_count(sent_text))
    return counts

def lexical_diversity_per_sentence(text: str, spans: List[Tuple[int, int]] = None) -> List[float]:
    """TTR per sentence (MTLD needs longer text)."""
    if spans is None:
        spans = segment_sentences(text)
    ttrs = []
    for _, start, end in spans:
        sent_text = text[start:end]
        ttrs.append(type_token_ratio(sent_text))
    return ttrs

def compute_lexical_features(text: str, spans: List[Tuple[int, int]] = None) -> dict:
    """
    Essay-level lexical features.
    Returns flat dict.
    """
    if spans is None:
        spans = segment_sentences(text)
    
    return {
        'lex_ttr': type_token_ratio(text),
        'lex_mtld': mtld_score(text),
        'lex_ai_phrase_count': ai_phrase_count(text),
        'lex_ai_phrase_rate': ai_phrase_count(text) / max(1, len(spans)),  # per sentence
    }

def compute_lexical_features_per_sentence(text: str, spans: List[Tuple[int, int]] = None) -> List[dict]:
    """
    Per-sentence lexical features for classifier.
    Returns list of dicts per sentence.
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
```

---

### Quick Test
```python
from src.signals.burstiness import compute_burstiness_features, sentence_context_burstiness
from src.signals.lexical import compute_lexical_features, compute_lexical_features_per_sentence
from src.signals.segment import segment_sentences

test = """I have always been fascinated by the stars. When I was five, my father bought me a small telescope.
We would spend hours on the backyard deck. Moreover, this experience taught me the value of patience.
As I reflect on those nights, I realize they shaped my path. Furthermore, the stars still guide me."""

spans = segment_sentences(test)
ppls = [25.3, 30.1, 22.8, 35.4, 28.9, 31.2]  # dummy perplexities

burst = compute_burstiness_features(test, ppls, spans)
print("Burstiness:", burst)

local_burst = sentence_context_burstiness(ppls)
print("Local burstiness:", local_burst)

lex = compute_lexical_features(test, spans)
print("Lexical (essay):", lex)

lex_sent = compute_lexical_features_per_sentence(test, spans)
print("Lexical (per sentence):", lex_sent)
```

---

## Definition of Done (Module 4)
- [ ] `burstiness.py`: `compute_burstiness_features()` returns essay-level length + perplexity burstiness (CV, IQR, mean, std)
- [ ] `burstiness.py`: `sentence_context_burstiness()` returns per-sentence local burstiness
- [ ] `lexical.py`: `compute_lexical_features()` returns essay-level TTR, MTLD, AI phrase count/rate
- [ ] `lexical.py`: `compute_lexical_features_per_sentence()` returns per-sentence TTR + AI phrase count
- [ ] Test script runs and outputs sensible numbers
- [ ] No feature pipeline or classifier code yet

---

## Handoff to Next Modules
- **Module 5** (Feature Pipeline) needs: both modules' functions working
- Can run in parallel with Module 3