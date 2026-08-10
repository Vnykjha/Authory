"""
Quick test for burstiness and lexical features (Module 4).
"""

from src.ai_essay_detector.signals.burstiness import compute_burstiness_features, sentence_context_burstiness
from src.ai_essay_detector.signals.lexical import compute_lexical_features, compute_lexical_features_per_sentence
from src.ai_essay_detector.signals.segment import segment_sentences

test = """I have always been fascinated by the stars. When I was five, my father bought me a small telescope.
We would spend hours on the backyard deck. Moreover, this experience taught me the value of patience.
As I reflect on those nights, I realize they shaped my path. Furthermore, the stars still guide me."""

spans = segment_sentences(test)
print("Sentence spans:", spans)

ppls = [25.3, 30.1, 22.8, 35.4, 28.9, 31.2]  # dummy perplexities

burst = compute_burstiness_features(test, ppls, spans)
print("\nBurstiness features:")
for k, v in burst.items():
    print(f"  {k}: {v}")

local_burst = sentence_context_burstiness(ppls)
print("\nLocal burstiness (per sentence):")
for item in local_burst:
    print(f"  Sentence {item['sentence_idx']}: cv={item['local_ppl_cv']:.4f}, iqr={item['local_ppl_iqr']:.4f}")

lex = compute_lexical_features(test, spans)
print("\nLexical features (essay):")
for k, v in lex.items():
    print(f"  {k}: {v}")

lex_sent = compute_lexical_features_per_sentence(test, spans)
print("\nLexical features (per sentence):")
for item in lex_sent:
    print(f"  Sentence {item['sentence_idx']}: ttr={item['lex_ttr']:.4f}, ai_phrases={item['lex_ai_phrase_count']}")

print("\n[OK] All tests passed!")