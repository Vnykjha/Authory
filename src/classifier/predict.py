"""
Classifier inference and explanation engine (Module 06).
Computes per-sentence AI probabilities, top 3 plain-language reasons using model coefficients,
and qualitative essay-level summaries (no bare percentages).
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional

from src.ai_essay_detector.signals.extract import FeatureExtractor, SentenceFeatures


# Plain-language reason mapping for model features
REASON_MAP = {
    'ppl_perplexity': "unusually predictable word choices (low perplexity)",
    'ppl_mean_logprob': "consistently high model likelihood across tokens",
    'ppl_mean_rank': "words strongly match top-predicted model tokens",
    'ppl_pct_rank_1': "frequent use of model's #1 predicted word",
    'ppl_pct_rank_le_5': "word choices heavily clustered in top 5 predictions",
    'ppl_pct_rank_le_10': "token sequence follows standard AI generation distribution",
    'cp_binoculars_ratio': "text structure aligns closely with model's self-prediction ratio",
    'cp_cross_perplexity': "low cross-model perplexity divergence",
    'burst_ppl_cv': "sentence perplexity variance is narrow (low burstiness)",
    'burst_len_cv': "sentence length variation is unnaturally uniform",
    'burst_ppl_iqr': "perplexity spread across paragraphs is tightly bounded",
    'local_burst_ppl_cv': "local sentence rhythm lacks natural human variance",
    'local_ppl_mean': "low average perplexity in surrounding sentence context window",
    'local_ppl_min': "local sentence cluster contains highly predictable AI phrasing",
    'local_ppl_max': "uniformly low perplexity ceiling across neighboring sentences",
    'local_ppl_delta_prev': "abrupt shift in predictability across adjacent sentence boundary",
    'lex_ttr': "low overall vocabulary diversity for essay length",
    'lex_mtld': "limited lexical richness and repetitive phrasing",
    'lex_ai_phrase_count': "contains stock AI transition phrases",
    'lex_ai_phrase_rate': "frequent density of stock AI phrases (e.g., 'moreover', 'delve', 'tapestry')",
    'lex_sent_ai_phrase_count': "sentence contains characteristic AI transition terms",
    'lex_sent_ttr': "sentence uses repetitive vocabulary",
}


class EssayClassifier:
    """
    Inference class that wraps trained model pipeline and feature extractor.
    """

    def __init__(self, model_path: str = 'models/logreg.joblib', device: str = 'cpu'):
        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}. Train model first via python -m src.classifier.train")

        data = joblib.load(model_file)
        self.pipeline = data['pipeline']
        self.feature_cols = data['feature_cols']
        self.feature_importance = pd.DataFrame(data['feature_importance'])
        self.extractor = FeatureExtractor(device=device)

    def predict_features(self, feat_dict: Dict[str, float]) -> float:
        """Predict AI probability for a feature dictionary."""
        X = np.array([[feat_dict.get(c, 0.0) for c in self.feature_cols]])
        proba = self.pipeline.predict_proba(X)[0, 1]
        return float(proba)

    def extract_top_reasons(self, feat_dict: Dict[str, float], top_n: int = 3) -> List[str]:
        """
        Extract top plain-language reasons based on feature contribution:
        contribution = scaled_feature_value * logistic_regression_coefficient
        """
        X = np.array([[feat_dict.get(c, 0.0) for c in self.feature_cols]])
        scaler = self.pipeline.named_steps['scaler']
        clf = self.pipeline.named_steps['clf']

        scaled_X = scaler.transform(X)[0]
        coefficients = clf.coef_[0]

        contributions = scaled_X * coefficients

        # Sort indices by highest positive contribution toward AI classification
        sorted_idx = np.argsort(contributions)[::-1]

        reasons = []
        for idx in sorted_idx:
            feat_name = self.feature_cols[idx]
            contrib = contributions[idx]

            # Only include features that contribute positively towards AI prediction
            if contrib > 0.01:
                reason_text = REASON_MAP.get(feat_name, f"unusual {feat_name} statistical pattern")
                if reason_text not in reasons:
                    reasons.append(reason_text)

            if len(reasons) >= top_n:
                break

        if not reasons:
            reasons = ["standard natural language variation"]

        return reasons

    def predict_essay(
        self,
        essay_text: str,
        essay_id: str = "user_input",
        topic: str = "unknown"
    ) -> List[Dict]:
        """
        Process essay text, extract sentence features, and return per-sentence predictions & reasons.
        """
        sentence_features = self.extractor.extract_essay(essay_text, essay_id, 'unknown', topic)

        results = []
        for sf in sentence_features:
            feat_dict = {
                'ppl_perplexity': sf.ppl_perplexity,
                'ppl_mean_logprob': sf.ppl_mean_logprob,
                'ppl_mean_rank': sf.ppl_mean_rank,
                'ppl_pct_rank_1': sf.ppl_pct_rank_1,
                'ppl_pct_rank_le_5': sf.ppl_pct_rank_le_5,
                'ppl_pct_rank_le_10': sf.ppl_pct_rank_le_10,
                'cp_perplexity': sf.cp_perplexity,
                'cp_cross_perplexity': sf.cp_cross_perplexity,
                'cp_binoculars_ratio': sf.cp_binoculars_ratio,
                'burst_len_mean': sf.burst_len_mean,
                'burst_len_std': sf.burst_len_std,
                'burst_len_cv': sf.burst_len_cv,
                'burst_len_iqr': sf.burst_len_iqr,
                'burst_ppl_mean': sf.burst_ppl_mean,
                'burst_ppl_std': sf.burst_ppl_std,
                'burst_ppl_cv': sf.burst_ppl_cv,
                'burst_ppl_iqr': sf.burst_ppl_iqr,
                'local_burst_ppl_cv': sf.local_burst_ppl_cv,
                'local_burst_ppl_iqr': sf.local_burst_ppl_iqr,
                'local_ppl_mean': getattr(sf, 'local_ppl_mean', 0.0),
                'local_ppl_min': getattr(sf, 'local_ppl_min', 0.0),
                'local_ppl_max': getattr(sf, 'local_ppl_max', 0.0),
                'local_ppl_delta_prev': getattr(sf, 'local_ppl_delta_prev', 0.0),
                'lex_ttr': sf.lex_ttr,
                'lex_mtld': sf.lex_mtld,
                'lex_ai_phrase_count': sf.lex_ai_phrase_count,
                'lex_ai_phrase_rate': sf.lex_ai_phrase_rate,
                'lex_sent_ttr': sf.lex_sent_ttr,
                'lex_sent_ai_phrase_count': sf.lex_sent_ai_phrase_count,
            }

            ai_proba = self.predict_features(feat_dict)
            reasons = self.extract_top_reasons(feat_dict)

            results.append({
                'sentence_idx': sf.sentence_idx,
                'text': sf.text,
                'start_char': sf.start_char,
                'end_char': sf.end_char,
                'ai_probability': round(ai_proba, 4),
                'reasons': reasons,
            })

        return results

    def summarize_essay(self, sentence_results: List[Dict]) -> Dict:
        """
        Produce qualitative essay-level classification summary.
        Per product requirement: never present a bare percentage as the main verdict.
        """
        if not sentence_results:
            return {
                'qualitative_band': 'insufficient text',
                'summary_description': 'No sentences found to analyze.',
                'avg_ai_probability': 0.0,
                'max_ai_probability': 0.0,
                'high_ai_sentences_count': 0,
                'total_sentences': 0,
            }

        probs = [r['ai_probability'] for r in sentence_results]
        avg_prob = float(np.mean(probs))
        max_prob = float(np.max(probs))
        high_ai_count = sum(1 for p in probs if p >= 0.70)
        total = len(sentence_results)

        if avg_prob < 0.30 and high_ai_count == 0:
            band = "Likely Human-Written"
            desc = "Statistical features show natural variation in vocabulary, perplexity rhythm, and sentence structure typical of human writing."
        elif avg_prob < 0.50 and high_ai_count <= max(1, total // 5):
            band = "Mixed Signals / Predominantly Human"
            desc = "The overall text appears largely human-written, with a few passages exhibiting uniform patterns."
        elif avg_prob < 0.70 or high_ai_count <= total // 2:
            band = "Likely AI-Assisted in Places"
            desc = "Contains several paragraphs with strong AI-like characteristics, suggesting hybrid or heavily edited AI text."
        else:
            band = "Strongly Indicative of AI Generation"
            desc = "The text consistently exhibits high predictability, uniform perplexity, and low structural variation typical of AI generation."

        return {
            'qualitative_band': band,
            'summary_description': desc,
            'avg_ai_probability': round(avg_prob, 4),
            'max_ai_probability': round(max_prob, 4),
            'high_ai_sentences_count': high_ai_count,
            'total_sentences': total,
        }
