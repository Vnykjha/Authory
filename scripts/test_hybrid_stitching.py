import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.classifier.predict import EssayClassifier

clf = EssayClassifier('models/logreg.joblib')

text = """My hands were covered in engine grease when the 1974 Honda Civic finally sputtered to life. My uncle laughed, wiping sweat from his forehead with a rag that was dirtier than the carburetor we had just spent three Saturdays rebuilding. Moreover, this experience taught me the value of patience, meticulous attention to detail, and careful observation. I wasn't thinking about college applications or career paths back then; I just wanted to figure out why the fuel pump kept vapor locking whenever the engine got hot. Furthermore, it is essential to delve into the multifaceted tapestry of mechanical engineering principles."""

results = clf.predict_essay(text)

print("=== CUSTOM HYBRID ESSAY TEST ===")
for r in results:
    p = r['ai_probability']
    color = "RED (High AI)" if p >= 0.70 else ("AMBER (Moderate)" if p >= 0.30 else "BLUE (Human)")
    print(f"[{color} - {p*100:.1f}%] {r['text']}")
