"""
Feature extraction orchestrator.
Combines all signal extractors into a single pipeline producing per-sentence feature records.
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
import numpy as np

from .segment import segment_sentences
from .perplexity import PerplexityComputer
from .cross_perplexity import CrossPerplexityComputer
from .burstiness import compute_burstiness_features, sentence_context_burstiness
from .lexical import compute_lexical_features, compute_lexical_features_per_sentence


@dataclass
class SentenceFeatures:
    """Per-sentence feature record for CSV."""
    # Identifiers
    essay_id: str
    sentence_idx: int
    text: str
    start_char: int
    end_char: int

    # Labels (for training)
    label: str  # 'human_native', 'human_esl', 'ai_generated', 'hybrid'
    source_category: str  # same as label for now
    topic: str

    # Single-model perplexity features
    ppl_perplexity: float
    ppl_mean_logprob: float
    ppl_mean_rank: float
    ppl_pct_rank_1: float
    ppl_pct_rank_le_5: float
    ppl_pct_rank_le_10: float

    # Cross-perplexity features
    cp_perplexity: float
    cp_cross_perplexity: float
    cp_binoculars_ratio: float

    # Burstiness (essay-level)
    burst_len_mean: float
    burst_len_std: float
    burst_len_cv: float
    burst_len_iqr: float
    burst_ppl_mean: float
    burst_ppl_std: float
    burst_ppl_cv: float
    burst_ppl_iqr: float

    # Burstiness (local context window ±2 sentences)
    local_burst_ppl_cv: float
    local_burst_ppl_iqr: float
    local_ppl_mean: float
    local_ppl_min: float
    local_ppl_max: float
    local_ppl_delta_prev: float

    # Lexical (essay-level)
    lex_ttr: float
    lex_mtld: float
    lex_ai_phrase_count: int
    lex_ai_phrase_rate: float

    # Lexical (per-sentence)
    lex_sent_ttr: float
    lex_sent_ai_phrase_count: int


class FeatureExtractor:
    def __init__(
        self,
        perplexity_model: str = "gpt2",
        cross_perf_model: str = "gpt2-medium",
        device: str = "cpu",
    ):
        self.cp_computer = CrossPerplexityComputer(
            observer_model=perplexity_model,
            performer_model=cross_perf_model,
            device=device,
        )
        self.ppl_computer = self.cp_computer.observer

    def extract_essay(self, essay_text: str, essay_id: str, label: str, topic: str) -> List[SentenceFeatures]:
        """Extract all features for one essay."""
        # 1. Segment
        spans = segment_sentences(essay_text)
        sentence_texts = [text for text, _, _ in spans]

        # 2. Single-model perplexity
        ppl_results = self.ppl_computer.process_essay(essay_text, spans)
        sentence_perplexities = [r['perplexity'] for r in ppl_results]

        # 3. Cross-perplexity (reusing pre-computed single-model perplexity)
        cp_results = self.cp_computer.process_essay(essay_text, spans, observer_results=ppl_results)

        # 4. Burstiness (essay-level)
        burst_feats = compute_burstiness_features(essay_text, sentence_perplexities, spans)

        # 5. Local burstiness (per-sentence)
        local_burst = sentence_context_burstiness(sentence_perplexities, window=2)
        local_burst_map = {r['sentence_idx']: r for r in local_burst}

        # 6. Lexical (essay-level)
        lex_feats = compute_lexical_features(essay_text, spans)

        # 7. Lexical (per-sentence)
        lex_sent = compute_lexical_features_per_sentence(essay_text, spans)
        lex_sent_map = {r['sentence_idx']: r for r in lex_sent}

        # 8. Assemble per-sentence records
        records = []
        for i, (sent_text, start, end) in enumerate(spans):
            ppl = ppl_results[i]
            cp = cp_results[i]
            lb = local_burst_map.get(i, {})
            ls = lex_sent_map.get(i, {})

            rec = SentenceFeatures(
                essay_id=essay_id,
                sentence_idx=i,
                text=sent_text,
                start_char=start,
                end_char=end,
                label=label,
                source_category=label,
                topic=topic,

                # Perplexity
                ppl_perplexity=ppl['perplexity'],
                ppl_mean_logprob=ppl['mean_token_logprob'],
                ppl_mean_rank=ppl['mean_token_rank'],
                ppl_pct_rank_1=ppl['pct_rank_1'],
                ppl_pct_rank_le_5=ppl['pct_rank_le_5'],
                ppl_pct_rank_le_10=ppl['pct_rank_le_10'],

                # Cross-perplexity
                cp_perplexity=cp['perplexity'],
                cp_cross_perplexity=cp['cross_perplexity'],
                cp_binoculars_ratio=cp['binoculars_ratio'],

                # Burstiness (essay-level)
                burst_len_mean=burst_feats.get('burst_len_mean', 0),
                burst_len_std=burst_feats.get('burst_len_std', 0),
                burst_len_cv=burst_feats.get('burst_len_cv', 0),
                burst_len_iqr=burst_feats.get('burst_len_iqr', 0),
                burst_ppl_mean=burst_feats.get('burst_ppl_mean', 0),
                burst_ppl_std=burst_feats.get('burst_ppl_std', 0),
                burst_ppl_cv=burst_feats.get('burst_ppl_cv', 0),
                burst_ppl_iqr=burst_feats.get('burst_ppl_iqr', 0),

                # Local burstiness & context window
                local_burst_ppl_cv=lb.get('local_ppl_cv', 0),
                local_burst_ppl_iqr=lb.get('local_ppl_iqr', 0),
                local_ppl_mean=lb.get('local_ppl_mean', 0),
                local_ppl_min=lb.get('local_ppl_min', 0),
                local_ppl_max=lb.get('local_ppl_max', 0),
                local_ppl_delta_prev=lb.get('local_ppl_delta_prev', 0),

                # Lexical (essay-level)
                lex_ttr=lex_feats.get('lex_ttr', 0),
                lex_mtld=lex_feats.get('lex_mtld', 0),
                lex_ai_phrase_count=lex_feats.get('lex_ai_phrase_count', 0),
                lex_ai_phrase_rate=lex_feats.get('lex_ai_phrase_rate', 0),

                # Lexical (per-sentence)
                lex_sent_ttr=ls.get('lex_ttr', 0),
                lex_sent_ai_phrase_count=ls.get('lex_ai_phrase_count', 0),
            )
            records.append(rec)

        return records