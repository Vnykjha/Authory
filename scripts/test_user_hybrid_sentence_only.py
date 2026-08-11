import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.classifier.predict import EssayClassifier

clf = EssayClassifier('models/logreg_sentence_only.joblib')

text = """My hands were covered in engine grease when the 1974 Honda Civic finally sputtered to life on a cold Saturday morning in my uncle's cramped garage. Uncle Ray laughed out loud, wiping sweat from his forehead with a dirty rag. I wasn't thinking about college applications back then; I just wanted to figure out why the fuel pump kept vapor locking whenever the engine got hot.

Moreover, this experience taught me the value of patience, meticulous attention to detail, and careful observation. Furthermore, it is essential to delve into the multifaceted tapestry of mechanical engineering principles to optimize energy efficiency and thermal dynamics. In conclusion, my journey in mechanics represents a harmonious synthesis of empirical curiosity and dedicated problem-solving."""

results = clf.predict_essay(text)

print("=== SENTENCE-ONLY MODEL ON HYBRID ESSAY ===")
for r in results:
    p = r['ai_probability']
    color = "RED (High AI)" if p >= 0.70 else ("AMBER (Moderate)" if p >= 0.30 else "BLUE (Human)")
    print(f"[{color} - {p*100:.1f}%] (Sentence #{r['sentence_idx']+1}): {r['text']}")
