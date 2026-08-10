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


# Non-feature metadata and text columns to exclude from training matrix X
EXCLUDE_COLS = {
    'essay_id', 'sentence_idx', 'text', 'start_char', 'end_char',
    'label', 'source_category', 'topic'
}


def prepare_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Extract numeric feature matrix X and binary target vector y.
    Target y: 1 for 'ai_generated', 0 for human ('human_native' or 'human_esl').
    """
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    # Filter out any non-numeric columns if present
    feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]

    X = df[feature_cols].fillna(0).values

    # Binary label: 1 = AI, 0 = Human (hybrid passages treated as AI if labeled so, else by label)
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
        'metrics': metrics,
        'feature_importance': feature_importance,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_proba': y_proba,
    }


def evaluate_esl_fpr(pipe: Pipeline, feature_cols: List[str], df: pd.DataFrame) -> Dict[str, float]:
    """
    Evaluate false-positive rate on ESL human essays vs native human essays.
    """
    results = {}
    for cat_name, label_val in [('native', 'human_native'), ('esl', 'human_esl')]:
        sub_df = df[df['label'] == label_val]
        if len(sub_df) == 0:
            results[f'{cat_name}_fpr'] = 0.0
            results[f'{cat_name}_count'] = 0
            continue

        X = sub_df[feature_cols].fillna(0).values
        preds = pipe.predict(X)
        fp = (preds == 1).sum()
        total = len(preds)
        fpr = float(fp / total) if total > 0 else 0.0

        results[f'{cat_name}_fpr'] = fpr
        results[f'{cat_name}_count'] = total
        results[f'{cat_name}_fp'] = int(fp)

    return results


def save_model(artifacts: Dict, path: str = 'models/logreg.joblib'):
    """Save trained pipeline, feature columns, and importances to joblib file."""
    model_path = Path(path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        'pipeline': artifacts['model'],
        'feature_cols': artifacts['feature_cols'],
        'feature_importance': artifacts['feature_importance'].to_dict('records'),
    }, model_path)
    print(f"Model successfully saved to {model_path}")


def main():
    print("=" * 60)
    print("MODULE 06: CLASSIFIER TRAINING & EVALUATION")
    print("=" * 60)

    train_df, test_df, held_out = held_out_topic_split()
    artifacts = train_logreg(train_df, test_df)

    esl_metrics = evaluate_esl_fpr(artifacts['model'], artifacts['feature_cols'], test_df)

    save_model(artifacts)

    m = artifacts['metrics']
    print(f"\n=== Held-Out Topic Test Results ({held_out}) ===")
    print(f"Accuracy: {m['accuracy']:.3f}")
    print(f"AI Precision: {m['precision_ai']:.3f} | Recall: {m['recall_ai']:.3f} | F1: {m['f1_ai']:.3f}")
    print(f"Human Precision: {m['precision_human']:.3f} | Recall: {m['recall_human']:.3f} | F1: {m['f1_human']:.3f}")
    print(f"Confusion Matrix (TN, FP / FN, TP):\n{np.array(m['confusion_matrix'])}")

    print(f"\n=== ESL False-Positive Rate Comparison ===")
    print(f"Native Human FPR: {esl_metrics.get('native_fpr', 0):.1%} ({esl_metrics.get('native_fp', 0)}/{esl_metrics.get('native_count', 0)})")
    print(f"ESL Human FPR:    {esl_metrics.get('esl_fpr', 0):.1%} ({esl_metrics.get('esl_fp', 0)}/{esl_metrics.get('esl_count', 0)})")

    print("\n=== Top 15 Features by |Coefficient| ===")
    print(artifacts['feature_importance'].head(15).to_string(index=False))


if __name__ == '__main__':
    main()
