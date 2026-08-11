import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.classifier.predict import EssayClassifier

clf = EssayClassifier('models/logreg.joblib')

# Longer essay text combining informal personal story + formal AI sentences
text = """My hands were covered in engine grease when the 1974 Honda Civic finally sputtered to life on a cold October morning in my uncle's backyard. Uncle Ray laughed out loud, wiping sweat from his forehead with a rag that was dirtier than the carburetor we had just spent three Saturdays rebuilding from scratch. I wasn't thinking about college applications or career paths back then. I just wanted to figure out why the fuel pump kept vapor locking whenever the engine got hot. Furthermore, it is essential to delve into the multifaceted tapestry of mechanical engineering principles to optimize energy efficiency and thermal dynamics. In conclusion, my journey in mechanics represents a harmonious synthesis of empirical curiosity and dedicated problem-solving. We finally cleaned up our tools as the sun went down behind the trees."""

results = clf.predict_essay(text)

print("=== LONGER ESSAY HIGHLIGHT CONTRAST TEST ===")
for r in results:
    p = r['ai_probability']
    color = "RED (High AI)" if p >= 0.70 else ("AMBER (Moderate)" if p >= 0.30 else "BLUE (Human)")
    print(f"[{color} - {p*100:.1f}%] (Sentence #{r['sentence_idx']+1}): {r['text']}")
