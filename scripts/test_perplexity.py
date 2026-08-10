#!/usr/bin/env python
"""
Test script for Module 2: Perplexity & Segmentation.
Verifies end-to-end functionality on a sample essay.
"""

from ai_essay_detector.signals.segment import segment_sentences
from ai_essay_detector.signals.perplexity import PerplexityComputer


def main():
    test_essay = """I have always been fascinated by the stars. When I was five, my father bought me a small telescope.
We would spend hours on the backyard deck, tracing constellations and inventing names for the ones we couldn't find.
That curiosity never left me. It guided my choice of physics in high school and now drives my research in astrophysics."""

    print("=" * 60)
    print("Testing Module 2: Perplexity & Segmentation")
    print("=" * 60)

    # Test sentence segmentation
    print("\n1. Testing sentence segmentation...")
    spans = segment_sentences(test_essay)
    print(f"Found {len(spans)} sentences:")
    for i, (text, start, end) in enumerate(spans):
        print(f"  [{i}] ({start}-{end}): {text[:80]}...")

    # Verify character offsets are correct
    print("\n2. Verifying character offsets...")
    for i, (text, start, end) in enumerate(spans):
        extracted = test_essay[start:end]
        match = "OK" if extracted == text else "FAIL"
        print(f"  [{i}] {match} Offset match: {extracted == text}")
        if extracted != text:
            print(f"    Expected: {repr(text)}")
            print(f"    Got:      {repr(extracted)}")

    # Test perplexity computation
    print("\n3. Testing perplexity computation...")
    print("Loading GPT-2 model (this may take a moment on first run)...")
    pc = PerplexityComputer()

    print("Processing essay...")
    results = pc.process_essay(test_essay, spans)

    print("\n4. Per-sentence results:")
    print("-" * 60)
    for r in results:
        print(f"Sentence {r['sentence_idx']}:")
        print(f"  Text: {r['text'][:60]}...")
        print(f"  Perplexity: {r['perplexity']:.2f}")
        print(f"  Mean token logprob: {r['mean_token_logprob']:.4f}")
        print(f"  Mean token rank: {r['mean_token_rank']:.2f}")
        print(f"  Pct rank=1: {r['pct_rank_1']:.1%}")
        print(f"  Pct rank<=5: {r['pct_rank_le_5']:.1%}")
        print(f"  Pct rank<=10: {r['pct_rank_le_10']:.1%}")
        print()

    # Summary validation
    print("5. Validation summary:")
    print("-" * 60)
    assert len(spans) == 5, f"Expected 5 sentences, got {len(spans)}"
    assert len(results) == 5, f"Expected 5 results, got {len(results)}"

    for r in results:
        assert "perplexity" in r
        assert "mean_token_logprob" in r
        assert "mean_token_rank" in r
        assert "pct_rank_1" in r
        assert "pct_rank_le_5" in r
        assert "pct_rank_le_10" in r
        assert r["perplexity"] > 0
        assert 0 <= r["pct_rank_1"] <= 1
        assert 0 <= r["pct_rank_le_5"] <= 1
        assert 0 <= r["pct_rank_le_10"] <= 1

    print("OK All assertions passed!")
    print("OK Module 2 implementation complete!")


if __name__ == "__main__":
    main()