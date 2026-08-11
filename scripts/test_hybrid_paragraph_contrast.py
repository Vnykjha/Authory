import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.classifier.predict import EssayClassifier

clf = EssayClassifier('models/logreg.joblib')

# Hybrid essay combining human personal essay paragraph + AI polished transition paragraph
hybrid_text = """I argue in favor of keeping the Electoral College. That is the way we did the voting for years and why would you want to change that. The outcome of that might be a disaster because we haven't used the popular vote to decide a president. The first reason why I think we should keep it is because when we vote for the people that represent us they have lots of experience in the politics area. They have spent years learning all about politics so the most thoughtful voters should be the ones to decide the election.

University education is a hotly debated topic, with many people having different opinions on its purpose. Some believe that it is solely to prepare students for employment, while others think it has a wider range of functions. Firstly, university education provides students with the opportunity to develop their knowledge and skills in a specific field. Students are encouraged to think critically, analyse information, and develop their own opinions."""

results = clf.predict_essay(hybrid_text)

print("=== HYBRID PARAGRAPH HIGHLIGHT BREAKDOWN ===")
for r in results:
    p = r['ai_probability']
    color = "RED (High AI)" if p >= 0.70 else ("AMBER (Moderate)" if p >= 0.30 else "BLUE (Human)")
    print(f"[{color} - {p*100:.1f}%] (Sentence #{r['sentence_idx']+1}): {r['text']}")
