# Module 2: Signal Extraction Scaffold — Perplexity & Segmentation
**Day 6 — Tasks 6.7–6.8**  
**Estimated time:** 2–3 hours  
**Can run in parallel with:** Module 1 (after Module 1's 6.1 done)

---

## Objective
Build the single-model perplexity signal extractor and sentence segmenter with character offsets — the foundation for all downstream features.

---

## Inputs Required (from Module 1)
- `pyproject.toml` installed with `transformers`, `torch`, `spacy`
- `en_core_web_sm` spaCy model downloaded
- (Optional) Sample essay text for manual testing

---

## Outputs Produced
| Path | Description |
|------|-------------|
| `src/signals/perplexity.py` | GPT-2 loader + per-sentence perplexity & token-rank computation |
| `src/signals/segment.py` | spaCy-based sentence splitter returning (text, start_idx, end_idx) |
| `src/signals/__init__.py` | Package exports |
| Quick test script (inline or `scripts/test_perplexity.py`) | Verifies end-to-end on a sample essay |

---

## Step-by-Step Tasks

### 6.7 Single-Model Perplexity (`src/signals/perplexity.py`)

**Requirements:**
- Load GPT-2 (small, 124M params) from Hugging Face `transformers`
- CPU inference only (no GPU required)
- Process essay in chunks (max 1024 tokens) with sliding window stride 512
- For each token: compute log-probability under model + rank in predicted distribution
- Aggregate to **per-sentence** metrics using sentence boundaries from segmenter

**Function signatures:**
```python
# src/signals/perplexity.py
from typing import List, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class PerplexityComputer:
    def __init__(self, model_name: str = "gpt2", device: str = "cpu", max_length: int = 1024, stride: int = 512):
        ...

    def compute_token_logprobs_and_ranks(self, text: str) -> List[Tuple[float, int]]:
        """
        Returns list of (logprob, rank) for each token in text.
        Rank = position in model's predicted next-token distribution (1 = top prediction).
        """
        ...

    def sentence_perplexity(self, sentence_text: str, token_logprobs: List[float], token_start: int, token_end: int) -> float:
        """
        Compute perplexity for a sentence given precomputed token logprobs.
        """
        ...

    def process_essay(self, essay_text: str, sentence_spans: List[Tuple[int, int]]) -> List[dict]:
        """
        Full pipeline: token logprobs/ranks → per-sentence features.
        Returns list of dicts per sentence:
        {
            'sentence_idx': int,
            'text': str,
            'start_char': int,
            'end_char': int,
            'perplexity': float,
            'mean_token_logprob': float,
            'mean_token_rank': float,
            'pct_rank_1': float,  # fraction of tokens where rank == 1
            'pct_rank_le_5': float,
            'pct_rank_le_10': float,
        }
        """
        ...
```

**Implementation notes:**
- Use `model(**inputs, labels=inputs.input_ids)` to get loss, then `exp(loss)` = perplexity
- For ranks: `logits = model(**inputs).logits`; `probs = softmax(logits, dim=-1)`; `rank = (probs.argsort(descending=True) == target_id).nonzero()`
- Batch sentences where possible for speed
- Handle OOM by falling back to smaller batch/chunk size

---

### 6.8 Sentence Segmentation (`src/signals/segment.py`)

**Requirements:**
- Use spaCy `en_core_web_sm` for sentence boundary detection
- Return **character offsets** (not token indices) for UI highlighting
- Preserve original whitespace/newlines

```python
# src/signals/segment.py
import spacy
from typing import List, Tuple

_nlp = None

def get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
        _nlp.add_pipe("sentencizer")  # faster than parser for just boundaries
    return _nlp

def segment_sentences(text: str) -> List[Tuple[str, int, int]]:
    """
    Split text into sentences with character offsets.
    Returns list of (sentence_text, start_char, end_char).
    end_char is exclusive (like Python slicing).
    """
    doc = get_nlp()(text)
    spans = []
    for sent in doc.sents:
        spans.append((sent.text, sent.start_char, sent.end_char))
    return spans
```

---

### Quick Test Script (inline verification)
```python
# Run this after writing both modules
from src.signals.segment import segment_sentences
from src.signals.perplexity import PerplexityComputer

test_essay = """I have always been fascinated by the stars. When I was five, my father bought me a small telescope. 
We would spend hours on the backyard deck, tracing constellations and inventing names for the ones we couldn't find.
That curiosity never left me. It guided my choice of physics in high school and now drives my research in astrophysics."""

spans = segment_sentences(test_essay)
print(f"Found {len(spans)} sentences:")
for i, (text, start, end) in enumerate(spans):
    print(f"  [{i}] ({start}-{end}): {text[:60]}...")

pc = PerplexityComputer()
results = pc.process_essay(test_essay, spans)
for r in results:
    print(f"Sentence {r['sentence_idx']}: ppl={r['perplexity']:.2f}, mean_rank={r['mean_token_rank']:.1f}, rank1={r['pct_rank_1']:.1%}")
```

**Expected output:** 4 sentences, perplexity ~15–50, mean rank > 1, some rank-1 tokens.

---

## Definition of Done (Module 2)
- [ ] `src/signals/perplexity.py` loads GPT-2 and computes per-sentence perplexity + token ranks
- [ ] `src/signals/segment.py` returns sentences with correct character offsets
- [ ] Test script runs without errors on sample essay
- [ ] Per-sentence features include: perplexity, mean_logprob, mean_rank, pct_rank_1, pct_rank_le_5, pct_rank_le_10
- [ ] No cross-perplexity, burstiness, or lexical code yet

---

## Handoff to Next Modules
- **Module 3** (Cross-Perplexity) needs: `PerplexityComputer` class working
- **Module 4** (Burstiness/Lexical) needs: `segment_sentences` working
- **Module 5** (Feature Pipeline) needs: both modules done