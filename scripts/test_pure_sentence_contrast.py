import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.classifier.predict import EssayClassifier

clf = EssayClassifier('models/logreg.joblib')

# Pure sentence contrast without stock phrase trigger words
essay_text = """My hands were covered in engine grease when the old 1974 Honda Civic finally sputtered to life on a cold October morning in my uncle's cramped garage. Uncle Ray laughed out loud, wiping sweat from his forehead with a dirty rag. The grey-cowled wood rail is a species of bird in the family Rallidae, the rails. It lives primarily in the forests, mangroves, and swamps of Central and South America. We finally packed up our wrenches and washed our hands under the outdoor spigot as the sun went down."""

results = clf.predict_essay(essay_text)

print("=== PURE SENTENCE CONTRAST TEST ===")
for r in results:
    p = r['ai_probability']
    color = "RED (High AI)" if p >= 0.70 else ("AMBER (Moderate)" if p >= 0.30 else "BLUE (Human)")
    print(f"[{color} - {p*100:.1f}%] (Sentence #{r['sentence_idx']+1}): {r['text']}")
