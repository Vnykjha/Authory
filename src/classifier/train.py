"""
Classifier training script (Module 06).
Trains Logistic Regression on signal features with held-out-topic evaluation,
evaluates ESL false-positive rate gap, and saves artifacts to models/logreg.joblib.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Tuple, List, Dict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

from .split import held_out_topic_split

# Features that belong strictly to sentence-level prediction
# Exclude global essay-level aggregates so per-sentence signals drive highlighting
EXCLUDE_COLS = {
    'essay_id', 'sentence_idx', 'text', 'start_char', 'end_char',
    'label', 'source_category', 'topic',

    # Exclude all global essay-level aggregates
    'burst_ppl_mean', 'burst_ppl_std', 'burst_ppl_cv', 'burst_ppl_iqr',
    'burst_len_mean', 'burst_len_std', 'burst_len_cv', 'burst_len_iqr',
    'lex_ttr', 'lex_mtld', 'lex_ai_phrase_count', 'lex_ai_phrase_rate',
}


def prepare_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Extract numeric feature matrix X and binary target vector y.
    Target y: 1 for 'ai_generated', 0 for human ('human_native' or 'human_esl').
    """
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]

    X = df[feature_cols].fillna(0).values
    y = (df['label'] == 'ai_generated').astype(int).values

    return X, y, feature_cols


def train_logreg(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    C: float = 1.0,
    class_weight: str = 'balanced',
    random_state: int = 42
) -> Dict:
    """
    Train Logistic Regression pipeline (StandardScaler + LogisticRegression),
    compute evaluation metrics, and extract feature importances.
    """
    X_train, y_train, feature_cols = prepare_features(train_df)
    X_test, y_test, _ = prepare_features(test_df)

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(
            C=C,
            class_weight=class_weight,
            random_state=random_state,
            max_iter=1000,
            solver='lbfgs'
        ))
    ])

    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, output_dict=True, target_names=['Human', 'AI'])
    cm = confusion_matrix(y_test, y_pred)

    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

    metrics = {
        'accuracy': report['accuracy'],
        'precision_ai': report.get('AI', {}).get('precision', 0.0),
        'recall_ai': report.get('AI', {}).get('recall', 0.0),
        'f1_ai': report.get('AI', {}).get('f1-score', 0.0),
        'precision_human': report.get('Human', {}).get('precision', 0.0),
        'recall_human': report.get('Human', {}).get('recall', 0.0),
        'f1_human': report.get('Human', {}).get('f1-score', 0.0),
        'confusion_matrix': cm.tolist(),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
    }

    coef = pipe.named_steps['clf'].coef_[0]
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'coefficient': coef,
        'abs_coef': np.abs(coef)
    }).sort_values('abs_coef', ascending=False)

    return {
        'model': pipe,
        'feature_cols': feature_cols,
        'feature_importance': feature_importance,
        'metrics': metrics,
    }


def main():
    print("=" * 60)
    print("MODULE 06: CLASSIFIER TRAINING & EVALUATION")
    print("=" * 60)

    train_df, test_df, held_out = held_out_topic_split()

    print(f"Train set: {len(train_df)} sentences from {train_df['essay_id'].nunique()} essays")
    print(f"Test set: {len(test_df)} sentences from {test_df['essay_id'].nunique()} essays")
    print(f"Held-out test topics: {held_out}")

    results = train_logreg(train_df, test_df)
    pipe = results['model']
    feature_cols = results['feature_cols']

    # Save trained model pipeline artifact
    model_dir = Path('models')
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / 'logreg.joblib'
    joblib.dump({
        'pipeline': pipe,
        'feature_cols': feature_cols,
        'feature_importance': results['feature_importance'].to_dict(orient='records'),
        'metrics': results['metrics'],
    }, model_path)
    print(f"Model successfully saved to {model_path}\n")

    # Evaluate held-out topic test metrics
    metrics = results['metrics']
    print(f"=== Held-Out Topic Test Results ({held_out}) ===")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"AI Precision: {metrics['precision_ai']:.3f} | Recall: {metrics['recall_ai']:.3f} | F1: {metrics['f1_ai']:.3f}")
    print(f"Human Precision: {metrics['precision_human']:.3f} | Recall: {metrics['recall_human']:.3f} | F1: {metrics['f1_human']:.3f}")
    print(f"Confusion Matrix (TN, FP / FN, TP):")
    print(f"[[{metrics['tn']}  {metrics['fp']}]\n [{metrics['fn']}  {metrics['tp']}]]\n")

    # Evaluate ESL FPR gap
    esl_df = test_df[test_df['label'] == 'human_esl']
    native_df = test_df[test_df['label'] == 'human_native']

    if len(esl_df) > 0 and len(native_df) > 0:
        X_esl = esl_df[feature_cols].fillna(0).values
        X_native = native_df[feature_cols].fillna(0).values

        esl_preds = pipe.predict(X_esl)
        native_preds = pipe.predict(X_native)

        esl_fpr = (esl_preds == 1).mean()
        native_fpr = (native_preds == 1).mean()

        print("=== ESL False-Positive Rate Comparison ===")
        print(f"Native Human FPR: {native_fpr*100:.1f}% ({sum(native_preds==1)}/{len(native_preds)})")
        print(f"ESL Human FPR:    {esl_fpr*100:.1f}% ({sum(esl_preds==1)}/{len(esl_preds)})\n")

    print("=== Top 15 Features by |Coefficient| ===")
    print(results['feature_importance'].head(15).to_string(index=False))


if __name__ == '__main__':
    main()
