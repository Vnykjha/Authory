import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

from src.classifier.split import held_out_topic_split

df = pd.read_csv('data/features.csv')
train_df, test_df, _ = held_out_topic_split()

# Define purely sentence-level feature columns
SENTENCE_FEATURE_COLS = [
    'ppl_perplexity',
    'ppl_mean_logprob',
    'ppl_mean_rank',
    'ppl_pct_rank_1',
    'ppl_pct_rank_le_5',
    'ppl_pct_rank_le_10',
    'cp_perplexity',
    'cp_cross_perplexity',
    'cp_binoculars_ratio',
    'local_burst_ppl_cv',
    'lex_sent_ttr',
    'lex_sent_ai_phrase_count',
]

X_train = train_df[SENTENCE_FEATURE_COLS].fillna(0).values
y_train = (train_df['label'] == 'ai_generated').astype(int).values

X_test = test_df[SENTENCE_FEATURE_COLS].fillna(0).values
y_test = (test_df['label'] == 'ai_generated').astype(int).values

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(C=1.0, class_weight='balanced', random_state=42, max_iter=1000))
])

pipe.fit(X_train, y_train)

acc = pipe.score(X_test, y_test)
print(f"Sentence-level classifier test accuracy: {acc*100:.2f}%")

# Save temporary sentence-level pipeline test
joblib.dump({
    'pipeline': pipe,
    'feature_cols': SENTENCE_FEATURE_COLS,
    'feature_importance': [{'feature': c, 'coefficient': co, 'abs_coef': abs(co)} for c, co in zip(SENTENCE_FEATURE_COLS, pipe.named_steps['clf'].coef_[0])]
}, 'models/logreg_sentence_only.joblib')

print("Saved models/logreg_sentence_only.joblib")
