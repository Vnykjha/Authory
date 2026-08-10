"""
Single-model perplexity computation using GPT-2.
Computes per-token log-probabilities and ranks, aggregates to per-sentence metrics.
"""

from typing import List, Tuple, Dict
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


class PerplexityComputer:
    """
    Computes perplexity and token-rank statistics using a causal LM (default: GPT-2 small).

    Processes text in sliding windows to handle long sequences.
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        device: str = "cpu",
        max_length: int = 1024,
        stride: int = 512,
    ):
        """
        Initialize the perplexity computer.

        Args:
            model_name: Hugging Face model identifier (default: "gpt2" = 124M params)
            device: Device to run inference on ("cpu" or "cuda")
            max_length: Maximum sequence length for model input
            stride: Sliding window stride for long texts
        """
        self.model_name = model_name
        self.device = torch.device(device)
        self.max_length = max_length
        self.stride = stride

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def compute_token_logprobs_and_ranks(self, text: str) -> List[Tuple[float, int]]:
        """
        Compute log-probability and rank for each token in the text.

        Args:
            text: Input text to analyze.

        Returns:
            List of (logprob, rank) tuples for each token.
            Rank = position in model's predicted next-token distribution (1 = top prediction).
        """
        if not text or not text.strip():
            return []

        # Tokenize the full text
        encodings = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=False,
            add_special_tokens=False,
        )
        input_ids = encodings.input_ids[0]  # Shape: [seq_len]
        seq_len = input_ids.size(0)

        if seq_len == 0:
            return []

        # Process in sliding windows
        all_logprobs = []
        all_ranks = []

        for i in range(0, seq_len, self.stride):
            # Define window boundaries
            begin_loc = max(i + self.stride - self.max_length, 0)
            end_loc = min(i + self.stride, seq_len)
            trg_len = end_loc - i  # tokens in target portion (may be < stride at end)

            # Input chunk for model
            input_chunk = input_ids[begin_loc:end_loc].unsqueeze(0).to(self.device)
            target_chunk = input_chunk.clone()

            # Mask tokens not in target portion for loss computation
            # We only compute loss on the target tokens (last trg_len tokens)
            target_chunk[:, :-trg_len] = -100

            with torch.no_grad():
                # Forward pass with labels to get loss (perplexity)
                outputs = self.model(input_chunk, labels=target_chunk)
                # loss is averaged over non-masked tokens
                # We need per-token logprobs, so do a separate forward pass

                # Get logits for all positions
                logits = self.model(input_chunk).logits  # [1, chunk_len, vocab_size]

                # Get log-probs for target tokens only
                # logits are for predicting next token, so shift by 1
                # For position j in chunk, logits[j] predicts token j+1
                shift_logits = logits[:, :-1, :]  # [1, chunk_len-1, vocab_size]
                shift_labels = input_chunk[:, 1:]  # [1, chunk_len-1]

                # Compute log-probs
                log_probs = F.log_softmax(shift_logits, dim=-1)  # [1, chunk_len-1, vocab_size]

                # Gather log-probs for actual tokens
                token_logprobs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)  # [1, chunk_len-1]
                token_logprobs = token_logprobs[0].cpu().tolist()

                # Vectorized rank calculation: rank = 1 + count of vocabulary tokens with logit > target token logit
                target_logits = shift_logits.gather(2, shift_labels.unsqueeze(-1))  # [1, chunk_len-1, 1]
                ranks_tensor = (shift_logits > target_logits).sum(dim=-1) + 1  # [1, chunk_len-1]
                ranks = ranks_tensor[0].cpu().tolist()

            # Only keep the target portion (last trg_len tokens)
            # The first chunk has no overlap, subsequent chunks overlap by stride
            if i == 0:
                # First window: keep all tokens
                all_logprobs.extend(token_logprobs)
                all_ranks.extend(ranks)
            else:
                # Subsequent windows: only keep the last trg_len tokens (non-overlapping part)
                # But we need to be careful: the token_logprobs correspond to shift_labels
                # which are input_chunk[:, 1:], so length is chunk_len - 1
                # The target portion is the last trg_len positions of shift_labels
                keep_start = len(token_logprobs) - trg_len
                all_logprobs.extend(token_logprobs[keep_start:])
                all_ranks.extend(ranks[keep_start:])

            if end_loc == seq_len:
                break

        return list(zip(all_logprobs, all_ranks))

    def sentence_perplexity(
        self,
        sentence_text: str,
        token_logprobs: List[float],
        token_start: int,
        token_end: int,
    ) -> float:
        """
        Compute perplexity for a sentence given precomputed token logprobs.

        Args:
            sentence_text: The sentence text (for reference).
            token_logprobs: List of log-probabilities for all tokens in the essay.
            token_start: Start token index (inclusive) for this sentence.
            token_end: End token index (exclusive) for this sentence.

        Returns:
            Perplexity value (exp of negative mean log-prob).
        """
        if token_start >= token_end or token_start >= len(token_logprobs):
            return float("inf")

        sent_logprobs = token_logprobs[token_start:token_end]
        if not sent_logprobs:
            return float("inf")

        mean_logprob = sum(sent_logprobs) / len(sent_logprobs)
        return float(torch.exp(torch.tensor(-mean_logprob)).item())

    def _get_token_spans(self, text: str, sentence_spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Map character-level sentence spans to token indices.

        Args:
            text: Full essay text.
            sentence_spans: List of (start_char, end_char) for each sentence.

        Returns:
            List of (token_start, token_end) for each sentence.
        """
        # Tokenize with offset mapping
        encodings = self.tokenizer(
            text,
            return_offsets_mapping=True,
            add_special_tokens=False,
        )
        offset_mapping = encodings.offset_mapping  # List of (start_char, end_char) per token

        token_spans = []
        for sent_start, sent_end in sentence_spans:
            token_start = None
            token_end = None

            for idx, (tok_start, tok_end) in enumerate(offset_mapping):
                # Token overlaps with sentence start
                if token_start is None and tok_start >= sent_start and tok_end <= sent_end:
                    token_start = idx
                # Token overlaps with sentence end
                if tok_start >= sent_start and tok_end <= sent_end:
                    token_end = idx + 1  # exclusive

            # Handle edge cases
            if token_start is None:
                token_start = 0
            if token_end is None or token_end <= token_start:
                token_end = token_start + 1

            token_spans.append((token_start, token_end))

        return token_spans

    def process_essay(
        self,
        essay_text: str,
        sentence_spans: List[Tuple[str, int, int]],
    ) -> List[Dict]:
        """
        Full pipeline: token logprobs/ranks → per-sentence features.

        Args:
            essay_text: Full essay text.
            sentence_spans: List of (sentence_text, start_char, end_char) from segmenter.

        Returns:
            List of dicts per sentence with features:
            {
                'sentence_idx': int,
                'text': str,
                'start_char': int,
                'end_char': int,
                'perplexity': float,
                'mean_token_logprob': float,
                'mean_token_rank': float,
                'pct_rank_1': float,
                'pct_rank_le_5': float,
                'pct_rank_le_10': float,
            }
        """
        if not essay_text or not essay_text.strip():
            return []

        # Extract just the character spans
        char_spans = [(start, end) for _, start, end in sentence_spans]

        # Get token logprobs and ranks for entire essay
        token_logprobs_ranks = self.compute_token_logprobs_and_ranks(essay_text)
        if not token_logprobs_ranks:
            # Return empty features for each sentence
            return [
                {
                    "sentence_idx": i,
                    "text": sent_text,
                    "start_char": start,
                    "end_char": end,
                    "perplexity": float("inf"),
                    "mean_token_logprob": 0.0,
                    "mean_token_rank": 0.0,
                    "pct_rank_1": 0.0,
                    "pct_rank_le_5": 0.0,
                    "pct_rank_le_10": 0.0,
                }
                for i, (sent_text, start, end) in enumerate(sentence_spans)
            ]

        token_logprobs = [lr[0] for lr in token_logprobs_ranks]
        token_ranks = [lr[1] for lr in token_logprobs_ranks]

        # Map sentence character spans to token indices
        token_spans = self._get_token_spans(essay_text, char_spans)

        # Compute per-sentence features
        results = []
        for i, (sent_text, start_char, end_char) in enumerate(sentence_spans):
            token_start, token_end = token_spans[i]

            # Clamp to valid range
            token_start = max(0, min(token_start, len(token_logprobs)))
            token_end = max(token_start, min(token_end, len(token_logprobs)))

            sent_logprobs = token_logprobs[token_start:token_end]
            sent_ranks = token_ranks[token_start:token_end]

            if not sent_logprobs:
                results.append({
                    "sentence_idx": i,
                    "text": sent_text,
                    "start_char": start_char,
                    "end_char": end_char,
                    "perplexity": float("inf"),
                    "mean_token_logprob": 0.0,
                    "mean_token_rank": 0.0,
                    "pct_rank_1": 0.0,
                    "pct_rank_le_5": 0.0,
                    "pct_rank_le_10": 0.0,
                })
                continue

            mean_logprob = sum(sent_logprobs) / len(sent_logprobs)
            perplexity = float(torch.exp(torch.tensor(-mean_logprob)).item())
            mean_rank = sum(sent_ranks) / len(sent_ranks)

            pct_rank_1 = sum(1 for r in sent_ranks if r == 1) / len(sent_ranks)
            pct_rank_le_5 = sum(1 for r in sent_ranks if r <= 5) / len(sent_ranks)
            pct_rank_le_10 = sum(1 for r in sent_ranks if r <= 10) / len(sent_ranks)

            results.append({
                "sentence_idx": i,
                "text": sent_text,
                "start_char": start_char,
                "end_char": end_char,
                "perplexity": perplexity,
                "mean_token_logprob": mean_logprob,
                "mean_token_rank": mean_rank,
                "pct_rank_1": pct_rank_1,
                "pct_rank_le_5": pct_rank_le_5,
                "pct_rank_le_10": pct_rank_le_10,
            })

        return results