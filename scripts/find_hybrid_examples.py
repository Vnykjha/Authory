import sys, os, glob
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.classifier.predict import EssayClassifier

clf = EssayClassifier('models/logreg.joblib')
files = sorted(glob.glob('data/hybrid/*.txt'))

print(f"Scanning {len(files)} hybrid essays for mixed sentence signals...\n")

mixed_examples = []

for filepath in files:
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read().strip()
    results = clf.predict_essay(text)
    probs = [r['ai_probability'] for r in results]
    high_ai = sum(1 for p in probs if p >= 0.60)
    low_ai = sum(1 for p in probs if p < 0.40)

    if high_ai > 0 and low_ai > 0:
        mixed_examples.append((filepath, text, results, high_ai, low_ai))

print(f"Found {len(mixed_examples)} essays with mixed sentence signals!\n")

for filepath, text, results, high_ai, low_ai in mixed_examples[:3]:
    filename = os.path.basename(filepath)
    print(f"==================================================")
    print(f"HYBRID EXAMPLE: {filename}")
    print(f"Stats: {len(results)} sentences ({low_ai} Human-like BLUE, {high_ai} AI-like RED)")
    print(f"==================================================")
    print(text)
    print("\nSentence Breakdown:")
    for r in results:
        prob = r['ai_probability']
        color = "RED (AI)" if prob >= 0.60 else ("AMBER" if prob >= 0.40 else "BLUE (Human)")
        print(f"  [{color} - {prob*100:.0f}%] {r['text']}")
    print("\n\n")
