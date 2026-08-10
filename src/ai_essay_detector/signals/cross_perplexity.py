"""
Cross-perplexity computation (simplified Binoculars signal).
Uses two models: observer (computes standard perplexity) and performer (generates predictions).
Cross-perplexity = observer scores performer's predicted distribution.
Ratio = perplexity / cross_perplexity — AI text → ratio closer to 1; human text → ratio > 1.
"""

from typing import List, Tuple, Dict
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from .perplexity import PerplexityComputer
from .segment import segment_sentences


class CrossPerplexityComputer:
    """
    Computes cross-perplexity ratio (simplified Binoculars) using two language models.

    Observer model: Computes standard perplexity of the text (same as Module 2)
    Performer model: Generates next-token predictions for the same text
    Cross-perplexity: Observer scores the performer's predicted distribution
    Ratio: perplexity / cross_perplexity
    """

    def __init__(
        self,
        observer_model: str = "gpt2",
        performer_model: str = "gpt2-medium",  # Same tokenizer family as gpt2
        device: str = "cpu",
        max_length: int = 1024,
        stride: int = 512,
    ):
        """
        Initialize the cross-perplexity computer.

        Args:
            observer_model: Hugging Face model for observer (default: "gpt2" = 124M)
            performer_model: Hugging Face model for performer (default: "gpt2-medium" = 355M)
            device: Device to run inference on ("cpu" or "cuda")
            max_length: Maximum sequence length for model input
            stride: Sliding window stride for long texts
        """
        self.observer = PerplexityComputer(observer_model, device, max_length, stride)
        self.device = torch.device(device)
        self.max_length = max_length
        self.stride = stride

        # Load performer model and tokenizer
        # gpt2 and gpt2-medium share the same tokenizer, simplifying alignment
        self.performer_tokenizer = AutoTokenizer.from_pretrained(performer_model)
        self.performer_tokenizer.pad_token = self.performer_tokenizer.eos_token

        self.performer_model = AutoModelForCausalLM.from_pretrained(performer_model).to(self.device)
        self.performer_model.eval()

    def compute_cross_perplexity(self, text: str) -> float:
        """
        Compute cross-perplexity for full text: observer scores performer's predicted distribution.

        For each position, performer predicts next-token distribution P_performer.
        Observer computes cross-entropy: -sum(P_performer * log P_observer).
        Returns exp(mean cross-entropy).

        Args:
            text: Input text to analyze.

        Returns:
            Cross-perplexity value (scalar).
        """
        if not text or not text.strip():
            return float("inf")

        # Tokenize with performer
        performer_inputs = self.performer_tokenizer(
            text, return_tensors="pt", truncation=True, max_length=self.max_length
        ).to(self.device)

        with torch.no_grad():
            performer_logits = self.performer_model(**performer_inputs).logits  # [1, seq_len, vocab]
            performer_probs = F.softmax(performer_logits, dim=-1)  # [1, seq_len, vocab]

        # Tokenize with observer (same tokenizer family, so compatible)
        observer_inputs = self.observer.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=self.max_length
        ).to(self.device)

        with torch.no_grad():
            observer_logits = self.observer.model(**observer_inputs).logits  # [1, seq_len, vocab]
            observer_logprobs = F.log_softmax(observer_logits, dim=-1)  # [1, seq_len, vocab]

        # Since GPT-2 and GPT-2-medium share the same tokenizer, vocab indices align
        # Cross-entropy at each position: -sum_v P_performer(v) * log P_observer(v)
        # We compute this for each position (except last, which has no next token)

        seq_len = min(performer_probs.shape[1], observer_logprobs.shape[1])
        if seq_len <= 1:
            return float("inf")

        # Vectorized cross-entropy computation: -sum_v (P_perf * log P_obs)
        p_perf = performer_probs[0, :seq_len - 1]  # [seq_len-1, vocab]
        log_p_obs = observer_logprobs[0, :seq_len - 1]  # [seq_len-1, vocab]

        cross_entropies = -(p_perf * log_p_obs).sum(dim=-1)  # [seq_len-1]
        mean_cross_entropy = cross_entropies.mean().item()
        return float(torch.exp(torch.tensor(mean_cross_entropy)).item())

    def compute_cross_perplexity_per_sentence(self, sentence_text: str) -> float:
        """
        Compute cross-perplexity for a single sentence.

        Args:
            sentence_text: Single sentence text.

        Returns:
            Cross-perplexity value for the sentence.
        """
        return self.compute_cross_perplexity(sentence_text)

    def process_essay(
        self,
        essay_text: str,
        sentence_spans: List[Tuple[str, int, int]],
        observer_results: Optional[List[Dict]] = None,
    ) -> List[Dict]:
        """
        Full pipeline: compute per-sentence cross-perplexity ratio.

        Args:
            essay_text: Full essay text.
            sentence_spans: List of (sentence_text, start_char, end_char) from segmenter.
            observer_results: Optional precomputed observer perplexity results.

        Returns:
            List of dicts per sentence.
        """
        if not essay_text or not essay_text.strip():
            return []

        # Get observer perplexity per sentence if not provided
        if observer_results is None:
            observer_results = self.observer.process_essay(essay_text, sentence_spans)

        # Compute essay-level cross-perplexity (more efficient than per-sentence)
        # For v1, use essay-level cross-perplexity as a global feature
        # and compute per-sentence ratio using sentence-level observer perplexity
        essay_cross_ppl = self.compute_cross_perplexity(essay_text)

        # Optionally compute per-sentence cross-perplexity for more granular features
        # (Commented out for performance; uncomment if needed)
        # sent_cross_ppls = []
        # for sent_text, _, _ in sentence_spans:
        #     sent_cross_ppls.append(self.compute_cross_perplexity_per_sentence(sent_text))

        results = []
        for obs in observer_results:
            sent_ppl = obs["perplexity"]

            # Use essay-level cross-perplexity for ratio computation
            # This is a practical approximation; true Binoculars uses per-sentence
            ratio = sent_ppl / essay_cross_ppl if essay_cross_ppl > 0 else 1.0

            results.append({
                "sentence_idx": obs["sentence_idx"],
                "text": obs["text"],
                "start_char": obs["start_char"],
                "end_char": obs["end_char"],
                "perplexity": sent_ppl,
                "cross_perplexity": essay_cross_ppl,
                "binoculars_ratio": ratio,
            })

        return results


# Convenience function for quick testing
def quick_test():
    """Quick test of CrossPerplexityComputer."""
    from .segment import segment_sentences

    test = "I have always been fascinated by the stars. My father bought me a telescope when I was five."
    spans = segment_sentences(test)

    print("Testing CrossPerplexityComputer...")
    cp = CrossPerplexityComputer()
    results = cp.process_essay(test, spans)

    for r in results:
        print(f"Sentence {r['sentence_idx']}: ppl={r['perplexity']:.2f}, cross_ppl={r['cross_perplexity']:.2f}, ratio={r['binoculars_ratio']:.3f}")

    return results


if __name__ == "__main__":
    quick_test()