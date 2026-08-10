import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.classifier.split import held_out_topic_split
from src.classifier.train import prepare_features

data = joblib.load('models/logreg.joblib')
pipe = data['pipeline']
feature_cols = data['feature_cols']

train_df, test_df, held_out = held_out_topic_split()
X_test = test_df[feature_cols].fillna(0).values
probs = pipe.predict_proba(X_test)[:, 1]
test_df['pred_prob'] = probs

print("=== 1. ESL False Positive Case Study ===")
esl_fp = test_df[(test_df['label'] == 'human_esl') & (test_df['pred_prob'] > 0.75)].sort_values('pred_prob', ascending=False).head(1)
for idx, r in esl_fp.iterrows():
    print(f"Essay ID: {r['essay_id']}")
    print(f"Text: \"{r['text']}\"")
    print(f"Predicted AI Prob: {r['pred_prob']:.4f}")
    print(f"Key Features: ppl_pct_rank_1={r.get('ppl_pct_rank_1', 0):.3f}, burst_len_cv={r.get('burst_len_cv', 0):.3f}, lex_ttr={r.get('lex_ttr', 0):.3f}, lex_ai_phrase_rate={r.get('lex_ai_phrase_rate', 0):.3f}")

print("\n=== 2. Native Human False Positive Case Study ===")
native_fp = test_df[(test_df['label'] == 'human_native') & (test_df['pred_prob'] > 0.4)].sort_values('pred_prob', ascending=False).head(1)
if len(native_fp) == 0:
    # If no native FP > 0.4 on held-out test set, get top prob native sentence
    native_fp = test_df[test_df['label'] == 'human_native'].sort_values('pred_prob', ascending=False).head(1)
for idx, r in native_fp.iterrows():
    print(f"Essay ID: {r['essay_id']}")
    print(f"Text: \"{r['text']}\"")
    print(f"Predicted AI Prob: {r['pred_prob']:.4f}")
    print(f"Key Features: ppl_pct_rank_1={r.get('ppl_pct_rank_1', 0):.3f}, burst_len_cv={r.get('burst_len_cv', 0):.3f}, lex_ttr={r.get('lex_ttr', 0):.3f}")

print("\n=== 3. False Negative / Hybrid Case Study ===")
ai_fn = test_df[(test_df['label'] == 'ai_generated')].sort_values('pred_prob', ascending=True).head(1)
for idx, r in ai_fn.iterrows():
    print(f"Essay ID: {r['essay_id']}")
    print(f"Text: \"{r['text']}\"")
    print(f"Predicted AI Prob: {r['pred_prob']:.4f}")
    print(f"Key Features: ppl_pct_rank_1={r.get('ppl_pct_rank_1', 0):.3f}, burst_len_cv={r.get('burst_len_cv', 0):.3f}, lex_ttr={r.get('lex_ttr', 0):.3f}")
