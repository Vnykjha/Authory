import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.classifier.predict import EssayClassifier

clf = EssayClassifier('models/logreg.joblib')

# Essay with high sentence length contrast and mixed sentence perplexity
essay_text = """My hands were covered in engine grease when the old 1974 Honda Civic finally sputtered to life on a cold October morning. Uncle Ray laughed out loud. We had spent three long Saturdays in his cramped garage rebuilding that rusty carburetor from scratch. Moreover, this experience taught me the value of patience, meticulous attention to detail, and careful observation. Furthermore, it is essential to delve into the multifaceted tapestry of mechanical engineering principles to optimize energy efficiency."""

results = clf.predict_essay(essay_text)

print("=== MIXED HIGHLIGHT TEST ===")
for r in results:
    p = r['ai_probability']
    color = "RED (High AI)" if p >= 0.70 else ("AMBER (Moderate)" if p >= 0.30 else "BLUE (Human)")
    print(f"[{color} - {p*100:.1f}%] (Sentence #{r['sentence_idx']+1}): {r['text']}")
