import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.classifier.predict import EssayClassifier

clf = EssayClassifier('models/logreg.joblib')

text1 = "My uncle laughed, wiping sweat from his forehead with a rag that was dirtier than the carburetor we had just spent three Saturdays rebuilding. The species as a whole is usually found at elevations from sea level to 2,000 metres. We finally got the engine running right before sunset."

results = clf.predict_essay(text1)

print("=== TEST CONTRASTING SENTENCES ===")
for r in results:
    p = r['ai_probability']
    color = "RED (AI)" if p >= 0.70 else ("AMBER" if p >= 0.30 else "BLUE (Human)")
    print(f"[{color} - {p*100:.1f}%] (Sentence #{r['sentence_idx']+1}): {r['text']}")
