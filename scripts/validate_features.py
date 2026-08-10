"""
Feature distribution validation and EDA script (Module 05, Task 7.6).
Checks features.csv for NaNs, distribution statistics, and Cohen's d class separation.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def main():
    features_path = Path("data/features.csv")
    if not features_path.exists():
        print(f"Error: {features_path} does not exist yet.")
        return

    df = pd.read_csv(features_path)

    print("=" * 60)
    print("MODULE 05: FEATURE DISTRIBUTION VALIDATION")
    print("=" * 60)

    print("\n1. Basic Checks:")
    print(f"  Total sentence records: {len(df)}")
    print(f"  Total columns: {len(df.columns)}")
    nan_count = df.isnull().sum().sum()
    print(f"  Total NaN count: {nan_count}")
    if nan_count > 0:
        print("  NaN breakdown by column:")
        print(df.isnull().sum()[df.isnull().sum() > 0])

    print("\n2. Label Distribution:")
    print(df['label'].value_counts())

    print("\n3. Unique Essays per Category:")
    if 'essay_id' in df.columns:
        print(df.groupby('label')['essay_id'].nunique())

    key_features = [
        'ppl_perplexity',
        'ppl_mean_rank',
        'ppl_pct_rank_1',
        'cp_binoculars_ratio',
        'burst_ppl_cv',
        'burst_len_cv',
        'lex_ttr',
        'lex_mtld',
        'lex_ai_phrase_rate',
        'lex_sent_ttr',
        'local_burst_ppl_cv'
    ]

    available_features = [f for f in key_features if f in df.columns]

    print("\n4. Key Feature Stats by Label (Mean ± Std):")
    print("-" * 60)
    for feat in available_features:
        print(f"\nFeature: {feat}")
        stats = df.groupby('label')[feat].agg(['mean', 'std', 'median'])
        print(stats.to_string())

    print("\n5. Class Separation Check (Cohen's d: AI vs Human Native):")
    print("-" * 60)
    human = df[df['label'] == 'human_native']
    ai = df[df['label'] == 'ai_generated']

    for feat in available_features:
        h_vals = human[feat].dropna()
        a_vals = ai[feat].dropna()

        if len(h_vals) > 0 and len(a_vals) > 0:
            h_var = h_vals.var()
            a_var = a_vals.var()
            pooled_std = np.sqrt((h_var + a_var) / 2) if (h_var + a_var) > 0 else 1.0
            d = (a_vals.mean() - h_vals.mean()) / pooled_std
            print(f"  {feat:25s}: Cohen's d = {d:+.3f}")

    print("\n" + "=" * 60)
    print("Validation script execution completed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
