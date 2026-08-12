"""
Topic-aware dataset splitting for classifier evaluation.
Splits dataset by topic rather than random sentence assignment to prevent data leakage.
Excludes ESL human essays from training set to measure false-positive bias in testing.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional, Union


def held_out_topic_split(
    features_path: Union[str, pd.DataFrame] = 'data/features.csv',
    metadata_path: str = 'data/essay_metadata.csv',
    test_topics: Optional[List[str]] = None,
    random_seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    if isinstance(features_path, pd.DataFrame):
        df = features_path.copy()
    else:
        df = pd.read_csv(features_path)

    # Ensure topic column exists; merge from metadata if missing in features.csv
    if 'topic' not in df.columns or df['topic'].isnull().all():
        if Path(metadata_path).exists():
            meta = pd.read_csv(metadata_path)
            if 'topic' in meta.columns:
                df = df.drop(columns=['topic'], errors='ignore')
                df = df.merge(meta[['essay_id', 'topic']], on='essay_id', how='left')

    if 'topic' not in df.columns:
        df['topic'] = 'general'

    df['topic'] = df['topic'].fillna('unknown')

    # Identify candidate topics for held-out evaluation
    all_topics = sorted([t for t in df['topic'].unique() if t not in ['unknown', 'general']])
    if not all_topics:
        all_topics = sorted(df['topic'].unique().tolist())

    if test_topics is None:
        n_test = max(1, len(all_topics) // 5)
        np.random.seed(random_seed)
        test_topics = list(np.random.choice(all_topics, n_test, replace=False))

    # Split main dataset by topic
    train_df = df[~df['topic'].isin(test_topics)].copy()
    test_df = df[df['topic'].isin(test_topics)].copy()

    # Rule: ESL essays are NEVER included in training set (per evaluation methodology)
    train_df = train_df[train_df['label'] != 'human_esl'].copy()

    # Place all ESL essays into test set for false-positive bias measurement
    esl_df = df[df['label'] == 'human_esl'].copy()

    # Avoid duplicating ESL essays if their topic was already in test_topics
    esl_not_in_test = esl_df[~esl_df['topic'].isin(test_topics)]
    test_df = pd.concat([test_df, esl_not_in_test], ignore_index=True)

    print(f"Train set: {len(train_df)} sentences from {train_df['essay_id'].nunique()} essays")
    print(f"Test set: {len(test_df)} sentences from {test_df['essay_id'].nunique()} essays")
    print(f"Held-out test topics: {test_topics}")
    print(f"ESL test sentences: {len(esl_df)}")

    return train_df, test_df, test_topics
