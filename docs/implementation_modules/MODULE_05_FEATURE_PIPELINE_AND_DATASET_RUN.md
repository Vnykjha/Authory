# Module 5: Feature Extraction Pipeline & Full Dataset Run
**Day 7 — Tasks 7.4–7.6**  
**Estimated time:** 3–4 hours  
**Depends on:** Modules 2, 3, 4 (all signal extractors working)  
**Inputs:** `data/` with essays, `docs/dataset.md` with topic info

---

## Objective
Orchestrate all signal extractors into a single pipeline that produces `data/features.csv` — one row per sentence with all features + labels + metadata for training.

---

## Inputs Required
- `src/signals/perplexity.py` — `PerplexityComputer.process_essay()`
- `src/signals/cross_perplexity.py` — `CrossPerplexityComputer.process_essay()`
- `src/signals/burstiness.py` — `compute_burstiness_features()`, `sentence_context_burstiness()`
- `src/signals/lexical.py` — `compute_lexical_features()`, `compute_lexical_features_per_sentence()`
- `src/signals/segment.py` — `segment_sentences()`
- `data/human_native/*.txt`, `data/human_esl/*.txt`, `data/ai_generated/*.txt`, `data/hybrid/*.txt`
- `docs/dataset.md` — for topic mapping

---

## Outputs Produced
| Path | Description |
|------|-------------|
| `src/signals/extract.py` | `FeatureExtractor` class orchestrating all signals |
| `scripts/extract_features.py` | CLI script: processes all essays → `data/features.csv` |
| `data/features.csv` | One row per sentence, all features + labels |
| `data/essay_metadata.csv` | One row per essay: ID, category, topic, source, split assignment |

---

## Step-by-Step Tasks

### 7.4 Feature Extraction Orchestrator (`src/signals/extract.py`)

```python
# src/signals/extract.py
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
import numpy as np
from .segment import segment_sentences
from .perplexity import PerplexityComputer
from .cross_perplexity import CrossPerplexityComputer
from .burstiness import compute_burstiness_features, sentence_context_burstiness
from .lexical import compute_lexical_features, compute_lexical_features_per_sentence

@dataclass
class SentenceFeatures:
    """Per-sentence feature record for CSV."""
    # Identifiers
    essay_id: str
    sentence_idx: int
    text: str
    start_char: int
    end_char: int
    
    # Labels (for training)
    label: str  # 'human_native', 'human_esl', 'ai_generated', 'hybrid'
    source_category: str  # same as label for now
    topic: str
    
    # Single-model perplexity features
    ppl_perplexity: float
    ppl_mean_logprob: float
    ppl_mean_rank: float
    ppl_pct_rank_1: float
    ppl_pct_rank_le_5: float
    ppl_pct_rank_le_10: float
    
    # Cross-perplexity features
    cp_perplexity: float
    cp_cross_perplexity: float
    cp_binoculars_ratio: float
    
    # Burstiness (essay-level)
    burst_len_mean: float
    burst_len_std: float
    burst_len_cv: float
    burst_len_iqr: float
    burst_ppl_mean: float
    burst_ppl_std: float
    burst_ppl_cv: float
    burst_ppl_iqr: float
    
    # Burstiness (local context window ±2 sentences)
    local_burst_ppl_cv: float
    local_burst_ppl_iqr: float
    
    # Lexical (essay-level)
    lex_ttr: float
    lex_mtld: float
    lex_ai_phrase_count: int
    lex_ai_phrase_rate: float
    
    # Lexical (per-sentence)
    lex_sent_ttr: float
    lex_sent_ai_phrase_count: int

class FeatureExtractor:
    def __init__(
        self,
        perplexity_model: str = "gpt2",
        cross_perf_model: str = "gpt2-medium",
        device: str = "cpu",
    ):
        self.ppl_computer = PerplexityComputer(perplexity_model, device)
        self.cp_computer = CrossPerplexityComputer(
            observer_model=perplexity_model,
            performer_model=cross_perf_model,
            device=device,
        )
    
    def extract_essay(self, essay_text: str, essay_id: str, label: str, topic: str) -> List[SentenceFeatures]:
        """Extract all features for one essay."""
        # 1. Segment
        spans = segment_sentences(essay_text)
        sentence_texts = [text for text, _, _ in spans]
        
        # 2. Single-model perplexity
        ppl_results = self.ppl_computer.process_essay(essay_text, spans)
        sentence_perplexities = [r['perplexity'] for r in ppl_results]
        
        # 3. Cross-perplexity
        cp_results = self.cp_computer.process_essay(essay_text, spans)
        
        # 4. Burstiness (essay-level)
        burst_feats = compute_burstiness_features(essay_text, sentence_perplexities, spans)
        
        # 5. Local burstiness (per-sentence)
        local_burst = sentence_context_burstiness(sentence_perplexities, window=2)
        local_burst_map = {r['sentence_idx']: r for r in local_burst}
        
        # 6. Lexical (essay-level)
        lex_feats = compute_lexical_features(essay_text, spans)
        
        # 7. Lexical (per-sentence)
        lex_sent = compute_lexical_features_per_sentence(essay_text, spans)
        lex_sent_map = {r['sentence_idx']: r for r in lex_sent}
        
        # 8. Assemble per-sentence records
        records = []
        for i, (sent_text, start, end) in enumerate(spans):
            ppl = ppl_results[i]
            cp = cp_results[i]
            lb = local_burst_map.get(i, {})
            ls = lex_sent_map.get(i, {})
            
            rec = SentenceFeatures(
                essay_id=essay_id,
                sentence_idx=i,
                text=sent_text,
                start_char=start,
                end_char=end,
                label=label,
                source_category=label,
                topic=topic,
                
                # Perplexity
                ppl_perplexity=ppl['perplexity'],
                ppl_mean_logprob=ppl['mean_token_logprob'],
                ppl_mean_rank=ppl['mean_token_rank'],
                ppl_pct_rank_1=ppl['pct_rank_1'],
                ppl_pct_rank_le_5=ppl['pct_rank_le_5'],
                ppl_pct_rank_le_10=ppl['pct_rank_le_10'],
                
                # Cross-perplexity
                cp_perplexity=cp['perplexity'],
                cp_cross_perplexity=cp['cross_perplexity'],
                cp_binoculars_ratio=cp['binoculars_ratio'],
                
                # Burstiness (essay-level)
                burst_len_mean=burst_feats.get('burst_len_mean', 0),
                burst_len_std=burst_feats.get('burst_len_std', 0),
                burst_len_cv=burst_feats.get('burst_len_cv', 0),
                burst_len_iqr=burst_feats.get('burst_len_iqr', 0),
                burst_ppl_mean=burst_feats.get('burst_ppl_mean', 0),
                burst_ppl_std=burst_feats.get('burst_ppl_std', 0),
                burst_ppl_cv=burst_feats.get('burst_ppl_cv', 0),
                burst_ppl_iqr=burst_feats.get('burst_ppl_iqr', 0),
                
                # Local burstiness
                local_burst_ppl_cv=lb.get('local_ppl_cv', 0),
                local_burst_ppl_iqr=lb.get('local_ppl_iqr', 0),
                
                # Lexical (essay-level)
                lex_ttr=lex_feats.get('lex_ttr', 0),
                lex_mtld=lex_feats.get('lex_mtld', 0),
                lex_ai_phrase_count=lex_feats.get('lex_ai_phrase_count', 0),
                lex_ai_phrase_rate=lex_feats.get('lex_ai_phrase_rate', 0),
                
                # Lexical (per-sentence)
                lex_sent_ttr=ls.get('lex_ttr', 0),
                lex_sent_ai_phrase_count=ls.get('lex_ai_phrase_count', 0),
            )
            records.append(rec)
        
        return records
```

---

### 7.5 Dataset Extraction Script (`scripts/extract_features.py`)

```python
# scripts/extract_features.py
import csv
import json
from pathlib import Path
from typing import Dict, List
import pandas as pd
from src.signals.extract import FeatureExtractor, SentenceFeatures

# Topic mapping from docs/dataset.md — fill in based on your dataset
ESSAY_TOPICS = {
    # "human_native/essay_001.txt": "overcoming_adversity",
    # "ai_generated/essay_042.txt": "stem_passion",
    # ... populate from your dataset.md
}

CATEGORY_DIRS = {
    'human_native': 'data/human_native',
    'human_esl': 'data/human_esl',
    'ai_generated': 'data/ai_generated',
    'hybrid': 'data/hybrid',
}

def load_topic_map() -> Dict[str, str]:
    """Load topic mapping from dataset.md or a JSON sidecar."""
    # For v1, hardcode or parse dataset.md
    # Better: create data/topics.json during Module 1
    topic_file = Path('data/topics.json')
    if topic_file.exists():
        return json.loads(topic_file.read_text())
    # Fallback: infer from filename or directory
    return {}

def main():
    topic_map = load_topic_map()
    extractor = FeatureExtractor()
    
    all_records: List[SentenceFeatures] = []
    essay_metadata = []
    
    for category, dir_path in CATEGORY_DIRS.items():
        dir_path = Path(dir_path)
        if not dir_path.exists():
            print(f"Warning: {dir_path} not found")
            continue
        
        essay_files = sorted(dir_path.glob('*.txt'))
        print(f"Processing {len(essay_files)} essays from {category}...")
        
        for essay_file in essay_files:
            essay_id = essay_file.stem
            text = essay_file.read_text(encoding='utf-8')
            topic = topic_map.get(f"{category}/{essay_file.name}", "unknown")
            
            try:
                records = extractor.extract_essay(text, essay_id, category, topic)
                all_records.extend(records)
                essay_metadata.append({
                    'essay_id': essay_id,
                    'category': category,
                    'topic': topic,
                    'sentence_count': len(records),
                    'char_count': len(text),
                })
            except Exception as e:
                print(f"  Error processing {essay_file}: {e}")
                continue
    
    # Write features.csv
    print(f"Writing {len(all_records)} sentence records to data/features.csv...")
    df = pd.DataFrame([asdict(r) for r in all_records])
    df.to_csv('data/features.csv', index=False)
    
    # Write essay metadata
    meta_df = pd.DataFrame(essay_metadata)
    meta_df.to_csv('data/essay_metadata.csv', index=False)
    
    print("Done!")
    print(f"Feature columns: {list(df.columns)}")
    print(f"Label distribution:\n{df['label'].value_counts()}")

if __name__ == '__main__':
    main()
```

**Create `data/topics.json`** (populate during Module 1 or manually now):
```json
{
  "human_native/essay_001.txt": "overcoming_adversity",
  "human_native/essay_002.txt": "stem_passion",
  ...
}
```

---

### 7.6 Validate Feature Distributions (EDA)

Run after extraction:
```python
# Quick validation script (can be inline in extract_features.py or separate)
import pandas as pd
import numpy as np

df = pd.read_csv('data/features.csv')

print("=== Basic Checks ===")
print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")
print(f"NaN count:\n{df.isnull().sum().sum()}")

print("\n=== Label Distribution ===")
print(df['label'].value_counts())

print("\n=== Key Feature Stats by Label ===")
key_features = [
    'ppl_perplexity', 'ppl_mean_rank', 'ppl_pct_rank_1',
    'cp_binoculars_ratio', 'burst_ppl_cv', 'burst_len_cv',
    'lex_ttr', 'lex_mtld', 'lex_ai_phrase_rate',
    'lex_sent_ttr', 'local_burst_ppl_cv'
]
for feat in key_features:
    if feat in df.columns:
        print(f"\n{feat}:")
        print(df.groupby('label')[feat].describe())

print("\n=== Class Separation Check (Cohen's d for top features) ===")
# Compare human_native vs ai_generated
human = df[df['label'] == 'human_native']
ai = df[df['label'] == 'ai_generated']
for feat in key_features:
    if feat in df.columns:
        d = (ai[feat].mean() - human[feat].mean()) / np.sqrt((ai[feat].var() + human[feat].var()) / 2)
        print(f"  {feat}: Cohen's d = {d:.3f}")
```

---

## Definition of Done (Module 5)
- [ ] `src/signals/extract.py` with `FeatureExtractor` class combining all signals
- [ ] `scripts/extract_features.py` runs end-to-end on all essays
- [ ] `data/features.csv` exists with one row per sentence, all features + labels + metadata
- [ ] `data/essay_metadata.csv` exists with one row per essay
- [ ] No NaNs in feature matrix (or documented NaN strategy)
- [ ] EDA shows class separation (Cohen's d > 0.5 on 2+ features)
- [ ] Extraction is idempotent (re-running produces same output)

---

## Handoff to Next Modules
- **Module 6** (Classifier) needs: `data/features.csv` + `data/essay_metadata.csv` with topic column for held-out-topic split
- **Module 7** (API) will need `FeatureExtractor` for inference