# Module 6: Classifier Training & Evaluation
**Day 8 — Tasks 8.1–8.7**  
**Estimated time:** 4–5 hours  
**Depends on:** Module 5 (`data/features.csv`, `data/essay_metadata.csv`)

---

## Objective
Train a Logistic Regression classifier (baseline) with held-out-topic evaluation, measure ESL false-positive rate, and save model artifacts for the API.

---

## Inputs Required
- `data/features.csv` — one row per sentence, all features + labels
- `data/essay_metadata.csv` — essay_id, category, topic for splitting
- `src/signals/extract.py` — `FeatureExtractor` for inference-time feature computation

---

## Outputs Produced
| Path | Description |
|------|-------------|
| `src/classifier/train.py` | Training script with held-out-topic split |
| `src/classifier/predict.py` | Inference: features → probability + top-3 plain-language reasons |
| `models/logreg.joblib` | Trained model + feature names + scaler |
| `docs/evaluation.md` (initial) | Confusion matrix, P/R/F1, ESL FPR comparison, held-out-topic results |

---

## Step-by-Step Tasks

### 8.1 Topic-Aware Train/Test Split

```python
# src/classifier/split.py
import pandas as pd
import numpy as np
from typing import Tuple, List

def held_out_topic_split(
    features_path: str = 'data/features.csv',
    metadata_path: str = 'data/essay_metadata.csv',
    test_topics: List[str] = None,
    random_seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Split by topic, not randomly.
    Returns: train_df, test_df, held_out_topics
    """
    df = pd.read_csv(features_path)
    meta = pd.read_csv(metadata_path)
    
    # Merge topic info
    df = df.merge(meta[['essay_id', 'topic']], on='essay_id', how='left')
    
    # Identify all topics
    all_topics = df['topic'].unique()
    all_topics = [t for t in all_topics if t != 'unknown']
    
    if test_topics is None:
        # Default: hold out 1 topic (or 20% of topics)
        n_test = max(1, len(all_topics) // 5)
        np.random.seed(random_seed)
        test_topics = list(np.random.choice(all_topics, n_test, replace=False))
    
    train_df = df[~df['topic'].isin(test_topics)].copy()
    test_df = df[df['topic'].isin(test_topics)].copy()
    
    # ESL essays NEVER in training (per project design)
    train_df = train_df[train_df['label'] != 'human_esl'].copy()
    # ESL goes to test set for bias measurement
    esl_df = df[df['label'] == 'human_esl'].copy()
    test_df = pd.concat([test_df, esl_df], ignore_index=True)
    
    print(f"Train: {len(train_df)} sentences from {train_df['essay_id'].nunique()} essays")
    print(f"Test (held-out topics): {len(test_df)} sentences from {test_df['essay_id'].nunique()} essays")
    print(f"Held-out topics: {test_topics}")
    print(f"ESL test sentences: {len(esl_df)}")
    
    return train_df, test_df, test_topics
```

---

### 8.2 Feature Preparation & Training (`src/classifier/train.py`)

```python
# src/classifier/train.py
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from .split import held_out_topic_split

# Feature columns (exclude identifiers, labels, text)
EXCLUDE_COLS = {
    'essay_id', 'sentence_idx', 'text', 'start_char', 'end_char',
    'label', 'source_category', 'topic'
}

def prepare_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Extract feature matrix X and labels y."""
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    X = df[feature_cols].values
    y = (df['label'] == 'ai_generated').astype(int).values  # binary: AI vs human
    # For hybrid, could use separate class or treat as AI — start with binary
    return X, y, feature_cols

def train_logreg(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    C: float = 1.0,
    class_weight: str = 'balanced',
    random_state: int = 42
) -> dict:
    """Train Logistic Regression, evaluate, return metrics + model artifacts."""
    
    X_train, y_train, feature_cols = prepare_features(train_df)
    X_test, y_test, _ = prepare_features(test_df)
    
    # Pipeline: scaler + logreg
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(
            C=C, class_weight=class_weight, random_state=random_state,
            max_iter=1000, solver='lbfgs'
        ))
    ])
    
    pipe.fit(X_train, y_train)
    
    # Predictions
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    
    # Metrics
    report = classification_report(y_test, y_pred, output_dict=True, target_names=['Human', 'AI'])
    cm = confusion_matrix(y_test, y_pred)
    
    # Per-class metrics
    tn, fp, fn, tp = cm.ravel()
    
    metrics = {
        'accuracy': report['accuracy'],
        'precision_ai': report['AI']['precision'],
        'recall_ai': report['AI']['recall'],
        'f1_ai': report['AI']['f1-score'],
        'precision_human': report['Human']['precision'],
        'recall_human': report['Human']['recall'],
        'f1_human': report['Human']['f1-score'],
        'confusion_matrix': cm.tolist(),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
    }
    
    # Feature importance (coefficients)
    coef = pipe.named_steps['clf'].coef_[0]
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'coefficient': coef,
        'abs_coef': np.abs(coef)
    }).sort_values('abs_coef', ascending=False)
    
    return {
        'model': pipe,
        'feature_cols': feature_cols,
        'metrics': metrics,
        'feature_importance': feature_importance,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_proba': y_proba,
    }

def save_model(artifacts: dict, path: str = 'models/logreg.joblib'):
    """Save model + metadata for inference."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        'pipeline': artifacts['model'],
        'feature_cols': artifacts['feature_cols'],
        'feature_importance': artifacts['feature_importance'].to_dict('records'),
    }, path)
    print(f"Model saved to {path}")

if __name__ == '__main__':
    train_df, test_df, held_out = held_out_topic_split()
    artifacts = train_logreg(train_df, test_df)
    save_model(artifacts)
    
    # Print metrics
    m = artifacts['metrics']
    print(f"\n=== Held-Out Topic Test Results ===")
    print(f"Accuracy: {m['accuracy']:.3f}")
    print(f"AI Precision: {m['precision_ai']:.3f}, Recall: {m['recall_ai']:.3f}, F1: {m['f1_ai']:.3f}")
    print(f"Human Precision: {m['precision_human']:.3f}, Recall: {m['recall_human']:.3f}, F1: {m['f1_human']:.3f}")
    print(f"Confusion Matrix:\n{np.array(m['confusion_matrix'])}")
    
    # Top features
    print("\n=== Top 15 Features by |Coefficient| ===")
    print(artifacts['feature_importance'].head(15).to_string(index=False))
```

---

### 8.3 ESL False-Positive Check (integrate into train.py or separate)

```python
# Add to train.py or separate script
def evaluate_esl_fpr(artifacts: dict, test_df: pd.DataFrame):
    """Measure false positive rate on ESL vs general human."""
    pipe = artifacts['model']
    feature_cols = artifacts['feature_cols']
    
    # ESL subset
    esl_df = test_df[test_df['label'] == 'human_esl']
    # General human (native) subset from held-out topics
    human_df = test_df[test_df['label'] == 'human_native']
    
    for name, subset in [('ESL', esl_df), ('Native Human', human_df)]:
        if len(subset) == 0:
            print(f"{name}: no data")
            continue
        X, y, _ = prepare_features(subset)
        y_pred = pipe.predict(X)
        fp = ((y_pred == 1) & (y == 0)).sum()
        tn = ((y_pred == 0) & (y == 0)).sum()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        print(f"{name} FPR: {fpr:.3f} ({fp}/{fp+tn})")
    
    return {
        'esl_fpr': fp / (fp + tn) if (fp + tn) > 0 else 0,
        'native_fpr': fp / (fp + tn) if (fp + tn) > 0 else 0,
    }
```

---

### 8.4 Inference with Explanations (`src/classifier/predict.py`)

```python
# src/classifier/predict.py
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict
from src.signals.extract import FeatureExtractor, SentenceFeatures

# Plain-language reason mapping (feature → human-readable)
REASON_MAP = {
    'ppl_perplexity': "unusually predictable word choices",
    'ppl_mean_rank': "consistently high-probability token selections",
    'ppl_pct_rank_1': "frequent use of model's top predicted word",
    'ppl_pct_rank_le_5': "word choices clustered in top predictions",
    'cp_binoculars_ratio': "text aligns closely with model's own predictions",
    'burst_ppl_cv': "sentence perplexity varies very little",
    'burst_len_cv': "sentence lengths are unusually uniform",
    'local_burst_ppl_cv': "local perplexity rhythm is overly consistent",
    'lex_ttr': "low vocabulary diversity for essay length",
    'lex_mtld': "limited lexical richness",
    'lex_ai_phrase_rate': "frequent use of stock AI transition phrases",
    'lex_sent_ai_phrase_count': "contains common AI phrasing patterns",
    'lex_sent_ttr': "sentence uses limited unique vocabulary",
}

class EssayClassifier:
    def __init__(self, model_path: str = 'models/logreg.joblib'):
        data = joblib.load(model_path)
        self.pipeline = data['pipeline']
        self.feature_cols = data['feature_cols']
        self.feature_importance = pd.DataFrame(data['feature_importance'])
        self.extractor = FeatureExtractor()
    
    def predict_sentence(self, features: Dict) -> Dict:
        """Predict AI probability for a single sentence's feature dict."""
        X = np.array([[features.get(c, 0) for c in self.feature_cols]])
        proba = self.pipeline.predict_proba(X)[0, 1]
        return {'ai_probability': float(proba)}
    
    def predict_essay(self, essay_text: str, essay_id: str = "unknown", topic: str = "unknown") -> List[Dict]:
        """Full inference: text → per-sentence predictions with explanations."""
        # Extract features
        sentence_features = self.extractor.extract_essay(essay_text, essay_id, 'unknown', topic)
        
        results = []
        for sf in sentence_features:
            # Convert to feature dict
            feat_dict = {
                'ppl_perplexity': sf.ppl_perplexity,
                'ppl_mean_logprob': sf.ppl_mean_logprob,
                'ppl_mean_rank': sf.ppl_mean_rank,
                'ppl_pct_rank_1': sf.ppl_pct_rank_1,
                'ppl_pct_rank_le_5': sf.ppl_pct_rank_5,
                'ppl_pct_rank_le_10': sf.ppl_pct_le_10,
                'cp_perplexity': sf.cp_perplexity,
                'cp_cross_perplexity': sf.cp_cross_perplexity,
                'cp_binoculars_ratio': sf.cp_binoculars_ratio,
                'burst_len_mean': sf.burst_len_mean,
                'burst_len_std': sf.burst_len_std,
                'burst_len_cv': sf.burst_len_cv,
                'burst_len_iqr': sf.burst_len_iqr,
                'burst_ppl_mean': sf.burst_ppl_mean,
                'burst_ppl_std': sf.burst_ppl_std,
                'burst_ppl_cv': sf.burst_ppl_cv,
                'burst_ppl_iqr': sf.burst_ppl_iqr,
                'local_burst_ppl_cv': sf.local_burst_ppl_cv,
                'local_burst_ppl_iqr': sf.local_burst_ppl_iqr,
                'lex_ttr': sf.lex_ttr,
                'lex_mtld': sf.lex_mtld,
                'lex_ai_phrase_count': sf.lex_ai_phrase_count,
                'lex_ai_phrase_rate': sf.lex_ai_phrase_rate,
                'lex_sent_ttr': sf.lex_sent_ttr,
                'lex_sent_ai_phrase_count': sf.lex_sent_ai_phrase_count,
            }
            
            pred = self.predict_sentence(feat_dict)
            
            # Top contributing reasons (coefficient * feature value)
            X = np.array([[feat_dict.get(c, 0) for c in self.feature_cols]])
            scaled_X = self.pipeline.named_steps['scaler'].transform(X)
            contributions = scaled_X[0] * self.pipeline.named_steps['clf'].coef_[0]
            
            # Top 3 by absolute contribution toward AI class (positive = more AI-like)
            top_idx = np.argsort(contributions)[::-1][:3]
            reasons = []
            for idx in top_idx:
                feat = self.feature_cols[idx]
                if contributions[idx] > 0.01:  # only meaningful contributions
                    reason_text = REASON_MAP.get(feat, f"unusual {feat} pattern")
                    reasons.append(reason_text)
            
            results.append({
                'sentence_idx': sf.sentence_idx,
                'text': sf.text,
                'start_char': sf.start_char,
                'end_char': sf.end_char,
                'ai_probability': pred['ai_probability'],
                'reasons': reasons[:3] if reasons else ["no strong signals"],
            })
        
        return results
    
    def summarize_essay(self, sentence_results: List[Dict]) -> Dict:
        """Qualitative summary band — never a bare percentage."""
        avg_prob = np.mean([r['ai_probability'] for r in sentence_results])
        max_prob = np.max([r['ai_probability'] for r in sentence_results])
        high_spans = sum(1 for r in sentence_results if r['ai_probability'] > 0.7)
        total = len(sentence_results)
        
        if avg_prob < 0.3:
            band = "likely human-written"
        elif avg_prob < 0.5:
            band = "mixed signals, possibly human with AI-like passages"
        elif avg_prob < 0.7:
            band = "likely AI-assisted in places"
        else:
            band = "strongly indicative of AI generation"
        
        return {
            'qualitative_band': band,
            'avg_ai_probability': float(avg_prob),
            'max_ai_probability': float(max_prob),
            'high_ai_sentences': f"{high_spans}/{total}",
        }
```

---

### 8.5 Initial `docs/evaluation.md` Structure

```markdown
# Evaluation Report

## Held-Out Topic Test Results
| Metric | Value |
|---|---|
| Accuracy | X.XXX |
| AI Precision / Recall / F1 | X.XXX / X.XXX / X.XXX |
| Human Precision / Recall / F1 | X.XXX / X.XXX / X.XXX |

### Confusion Matrix
| | Pred Human | Pred AI |
|---|---|---|
| True Human | TN | FP |
| True AI | FN | TP |

## ESL False-Positive Rate Comparison
| Subset | False Positive Rate | Count |
|---|---|---|
| Native Human (held-out topics) | X.XXX | N |
| ESL Human (never in training) | X.XXX | N |

**Mechanism hypothesis:** ESL writing shows lower lexical diversity (TTR, MTLD) and lower perplexity due to simpler vocabulary and more standard constructions — the same statistical signature that single-model perplexity flags as AI-like. Cross-perplexity ratio partially mitigates this but gap remains.

## Top Contributing Features
| Feature | Coefficient | Direction |
|---|---|---|

## 3 Confidently-Wrong Examples
*(To be filled after manual inspection of test predictions)*

---

*This document will be finalized on Day 10 with wrong examples and full analysis.*
```

---

## Definition of Done (Module 6)
- [ ] `held_out_topic_split()` correctly separates by topic, excludes ESL from training
- [ ] Logistic Regression trained with balanced class weights, saved to `models/logreg.joblib`
- [ ] Held-out-topic metrics printed: accuracy, P/R/F1 per class, confusion matrix
- [ ] ESL FPR vs. native human FPR measured and logged
- [ ] `EssayClassifier.predict_essay()` returns per-sentence: probability, reasons, offsets
- [ ] `summarize_essay()` returns qualitative band (no bare percentage)
- [ ] Initial `docs/evaluation.md` created with metrics tables

---

## Handoff to Next Modules
- **Module 7** (Backend API) needs: `EssayClassifier` class loadable from `models/logreg.joblib`
- **Module 8/9** (Frontend) need: API endpoint returning sentence results with reasons
- **Module 10** (Final Evaluation) needs: `docs/evaluation.md` initialized