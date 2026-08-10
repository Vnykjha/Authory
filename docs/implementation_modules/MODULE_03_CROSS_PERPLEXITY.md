# Module 3: Cross-Perplexity (Second Model / Simplified Binoculars)
**Day 7 — Task 7.1**  
**Estimated time:** 2–3 hours  
**Depends on:** Module 2 (PerplexityComputer working)  
**Can run in parallel with:** Module 4 (Burstiness & Lexical)

---

## Objective
Add a second small language model to compute the cross-perplexity ratio (simplified Binoculars signal) — the key robustness upgrade over single-model perplexity.

---

## Inputs Required
- `src/signals/perplexity.py` with working `PerplexityComputer`
- `src/signals/segment.py` with `segment_sentences`
- Two model checkpoints available via Hugging Face (GPT-2 + GPT-2-medium or Pythia-160m)

---

## Outputs Produced
| Path | Description |
|------|-------------|
| `src/signals/cross_perplexity.py` | CrossPerplexityComputer: observer + performer models → per-sentence ratio |
| Updated `src/signals/__init__.py` | Export new class |

---

## Background: Simplified Binoculars
| Component | Role |
|---|---|
| **Observer model** | Computes standard perplexity of the text (same as Module 2) |
| **Performer model** | Generates next-token predictions for the same text |
| **Cross-perplexity** | Observer scores the performer's predicted distribution (not the actual tokens) |
| **Ratio** | `perplexity / cross_perplexity` — AI text → ratio closer to 1; human text → ratio > 1 |

**Why it helps ESL bias:** Both models see the same "simple vocabulary" effect, so the ratio partially cancels it out (per Binoculars paper).

---

## Step-by-Step Tasks

### 7.1 Cross-Perplexity Implementation (`src/signals/cross_perplexity.py`)

**Model pair (CPU-friendly):**
- Observer: `gpt2` (124M) — same as Module 2
- Performer: `gpt2-medium` (355M) **or** `EleutherAI/pythia-160m` (160M) — choose based on VRAM/RAM

```python
# src/signals/cross_perplexity.py
from typing import List, Tuple
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from .perplexity import PerplexityComputer
from .segment import segment_sentences

class CrossPerplexityComputer:
    def __init__(
        self,
        observer_model: str = "gpt2",
        performer_model: str = "gpt2-medium",  # or "EleutherAI/pythia-160m"
        device: str = "cpu",
        max_length: int = 1024,
        stride: int = 512,
    ):
        self.observer = PerplexityComputer(observer_model, device, max_length, stride)
        # Performer only needs to generate logits, not compute loss
        self.performer_tokenizer = AutoTokenizer.from_pretrained(performer_model)
        self.performer_model = AutoModelForCausalLM.from_pretrained(performer_model).to(device)
        self.performer_model.eval()
        self.device = device
        self.max_length = max_length
        self.stride = stride

    def compute_cross_perplexity(self, text: str) -> float:
        """
        Compute cross-perplexity: observer scores performer's predicted distribution.
        For each position, performer predicts next-token distribution P_performer.
        Observer computes cross-entropy: -sum(P_performer * log P_observer).
        """
        # Tokenize with performer (may have different vocab)
        performer_inputs = self.performer_tokenizer(
            text, return_tensors="pt", truncation=True, max_length=self.max_length
        ).to(self.device)
        
        with torch.no_grad():
            performer_logits = self.performer_model(**performer_inputs).logits  # [1, seq_len, vocab]
            performer_probs = F.softmax(performer_logits, dim=-1)  # [1, seq_len, vocab]
        
        # Now score with observer
        # Need to map performer tokens to observer tokens — use observer tokenizer
        observer_inputs = self.observer.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=self.max_length
        ).to(self.device)
        
        with torch.no_grad():
            observer_logits = self.observer.model(**observer_inputs).logits  # [1, seq_len, vocab_obs]
            observer_logprobs = F.log_softmax(observer_logits, dim=-1)  # [1, seq_len, vocab_obs]
        
        # Align tokenizations (simplified: assume same tokenizer or map via text)
        # For GPT-2 family, tokenizers are compatible — use observer's tokenization
        # Cross-entropy: -mean over positions of sum_v P_performer(v) * log P_observer(v)
        # This requires aligning vocab — use observer's vocab as reference
        
        # Simplified practical approach (common in Binoculars implementations):
        # 1. Get performer's top-k predictions at each position
        # 2. Score those tokens under observer
        # 3. Average
        
        cross_entropies = []
        for i in range(observer_inputs.input_ids.shape[1] - 1):
            # Performer's distribution at position i (for next token)
            if i < performer_probs.shape[1]:
                p_perf = performer_probs[0, i]  # [vocab_perf]
            else:
                break
            
            # Observer's log-probs at position i
            log_p_obs = observer_logprobs[0, i]  # [vocab_obs]
            
            # Cross-entropy: -sum_v P_perf(v) * log P_obs(v)
            # Since tokenizers differ, approximate by taking performer's top-k and scoring in observer
            top_k = 100
            top_probs, top_indices = p_perf.topk(top_k)
            # Map performer token IDs to observer token IDs via text decode/encode
            # Simpler: if same tokenizer family, indices align approximately
            # For GPT-2 + GPT-2-medium: same tokenizer!
            cross_ent = -(top_probs * log_p_obs[top_indices]).sum()
            cross_entropies.append(cross_ent.item())
        
        return float(torch.tensor(cross_entropies).mean().exp())

    def process_essay(self, essay_text: str, sentence_spans: List[Tuple[int, int]]) -> List[dict]:
        """
        Compute per-sentence cross-perplexity ratio.
        Returns list of dicts per sentence with:
        {
            'sentence_idx': int,
            'perplexity': float,           # from observer (reuse Module 2)
            'cross_perplexity': float,     # new
            'binoculars_ratio': float,     # perplexity / cross_perplexity
        }
        """
        # Get observer perplexity per sentence (reuse Module 2 logic)
        observer_results = self.observer.process_essay(essay_text, sentence_spans)
        
        # For cross-perplexity, compute on full text then approximate per-sentence
        # (Full per-sentence cross-perplexity is expensive; use full-text as proxy
        #  or compute per-sentence if time allows)
        full_cross_ppl = self.compute_cross_perplexity(essay_text)
        
        # Approximate: assume cross-perplexity distributes similarly to perplexity
        # Better: compute cross-perplexity per sentence (slower but accurate)
        # For v1, use full-text cross-perplexity as essay-level feature
        # and add sentence-level ratio using observer perplexity only
        
        results = []
        for obs in observer_results:
            sent_ppl = obs['perplexity']
            # Essay-level ratio as fallback; per-sentence if computed
            ratio = sent_ppl / full_cross_ppl if full_cross_ppl > 0 else 1.0
            results.append({
                'sentence_idx': obs['sentence_idx'],
                'perplexity': sent_ppl,
                'cross_perplexity': full_cross_ppl,  # essay-level
                'binoculars_ratio': ratio,
            })
        return results


# Simpler alternative: per-sentence cross-perplexity (more accurate, slower)
def compute_cross_perplexity_per_sentence(
    self, sentence_text: str
) -> float:
    """Compute cross-perplexity for a single sentence."""
    # Same logic as compute_cross_perplexity but on one sentence
    # Can be called in loop over sentences
    ...
```

**Optimization notes:**
- Cache performer model outputs if same text processed multiple times
- Batch sentence processing where possible
- If too slow, fall back to essay-level cross-perplexity + sentence-level observer perplexity only
- Target: < 5 seconds per essay on CPU

---

### Quick Test
```python
from src.signals.cross_perplexity import CrossPerplexityComputer
from src.signals.segment import segment_sentences

test = "I have always been fascinated by the stars. My father bought me a telescope when I was five."
spans = segment_sentences(test)

cp = CrossPerplexityComputer()
results = cp.process_essay(test, spans)
for r in results:
    print(f"Sentence {r['sentence_idx']}: ppl={r['perplexity']:.2f}, cross_ppl={r['cross_perplexity']:.2f}, ratio={r['binoculars_ratio']:.3f}")
```

**Expected:** Ratios > 1 for human-like text; closer to 1 for AI-like text.

---

## Definition of Done (Module 3)
- [ ] `CrossPerplexityComputer` loads two models (observer + performer)
- [ ] `compute_cross_perplexity()` returns a scalar for full text
- [ ] `process_essay()` returns per-sentence dicts with `perplexity`, `cross_perplexity`, `binoculars_ratio`
- [ ] Runs on CPU in < 10 seconds for a 650-word essay
- [ ] No burstiness or lexical code yet

---

## Handoff to Next Modules
- **Module 5** (Feature Pipeline) needs: `CrossPerplexityComputer.process_essay()` working
- **Module 4** can run in parallel (independent)