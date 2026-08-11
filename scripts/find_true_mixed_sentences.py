import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import joblib

data = joblib.load('models/logreg.joblib')
pipe = data['pipeline']
feature_cols = data['feature_cols']

df = pd.read_csv('data/features.csv')
print(f"Total dataset sentences: {len(df)}")

# Predict on all sentences in features.csv
X = df[feature_cols].fillna(0).values
probs = pipe.predict_proba(X)[:, 1]
df['pred_prob'] = probs

# Group by essay_id and check min/max probability within each essay
essay_grouped = df.groupby('essay_id').agg(
    label=('label', 'first'),
    topic=('topic', 'first'),
    num_sents=('sentence_idx', 'count'),
    min_prob=('pred_prob', 'min'),
    max_prob=('pred_prob', 'max'),
    std_prob=('pred_prob', 'std')
).reset_index()

print("\n=== ESSAYS WITH SENTENCE HIGHLIGHT VARIANCE (min < 0.4, max > 0.6) ===")
mixed_essays = essay_grouped[(essay_grouped['min_prob'] < 0.40) & (essay_grouped['max_prob'] > 0.60)].sort_values('std_prob', ascending=False)
print(mixed_essays.to_string())

if len(mixed_essays) > 0:
    target_id = mixed_essays.iloc[0]['essay_id']
    sample_sents = df[df['essay_id'] == target_id]
    print(f"\nDetailed Sentence Breakdown for Essay '{target_id}' ({sample_sents.iloc[0]['label']}):")
    for idx, r in sample_sents.iterrows():
        p = r['pred_prob']
        tag = "RED (AI)" if p >= 0.70 else ("AMBER" if p >= 0.30 else "BLUE (Human)")
        print(f"  [{tag} - {p*100:.1f}%] (Sentence #{r['sentence_idx']+1}): {r['text']}")
