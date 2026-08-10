"""
Feature extraction script for full dataset.
Processes all essays and outputs data/features.csv + data/essay_metadata.csv
"""

import csv
import json
from pathlib import Path
from typing import Dict, List
import pandas as pd

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ai_essay_detector.signals.extract import FeatureExtractor, SentenceFeatures


# Topic mapping - populate based on your dataset
ESSAY_TOPICS = {}

CATEGORY_DIRS = {
    'human_native': 'data/human_native',
    'human_esl': 'data/human_esl',
    'ai_generated': 'data/ai_generated',
    'hybrid': 'data/hybrid',
}


def load_topic_map() -> Dict[str, str]:
    """Load topic mapping from data/topics.json if it exists."""
    topic_file = Path('data/topics.json')
    if topic_file.exists():
        return json.loads(topic_file.read_text())
    return {}


def main():
    topic_map = load_topic_map()
    extractor = FeatureExtractor(device='cpu')

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
    df = pd.DataFrame([vars(r) for r in all_records])
    df.to_csv('data/features.csv', index=False)

    # Write essay metadata
    meta_df = pd.DataFrame(essay_metadata)
    meta_df.to_csv('data/essay_metadata.csv', index=False)

    print("Done!")
    print(f"Feature columns: {list(df.columns)}")
    print(f"Label distribution:\n{df['label'].value_counts()}")


if __name__ == '__main__':
    main()