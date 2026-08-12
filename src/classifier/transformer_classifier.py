"""
Context-Aware Sentence Transformer Classifier.
Uses fine-tuned transformer sequence classification (DeBERTa-v3/DistilRoBERTa) over 3-sentence sliding windows
to detect subtle AI phrasing, transition syntax, and style shifts.
"""

import os
import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification


DEFAULT_MODEL_NAME = "distilbert-base-uncased"  # Fast, lightweight default transformer


class ContextTransformerClassifier:
    """
    Context-aware transformer sequence classifier for sentences in essays.
    Evaluates sentence S_i with local context [S_{i-1}, S_i, S_{i+1}] to produce contextual AI probability.
    """

    def __init__(self, model_name_or_path: str = DEFAULT_MODEL_NAME, device: str = "cpu"):
        self.device = torch.device(device)
        self.model_name = model_name_or_path

        print(f"Loading Context Transformer Classifier ({self.model_name})...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=2
        ).to(self.device)
        self.model.eval()

    def get_sentence_contexts(self, sentence_texts: List[str]) -> List[str]:
        """
        Build 3-sentence sliding windows around each sentence.
        Format: "Previous sentence... Target sentence... Next sentence..."
        """
        n = len(sentence_texts)
        contexts = []
        for i in range(n):
            prev_s = sentence_texts[i - 1] if i > 0 else ""
            curr_s = sentence_texts[i]
            next_s = sentence_texts[i + 1] if i < n - 1 else ""

            if prev_s and next_s:
                ctx = f"{prev_s} {curr_s} {next_s}"
            elif prev_s:
                ctx = f"{prev_s} {curr_s}"
            elif next_s:
                ctx = f"{curr_s} {next_s}"
            else:
                ctx = curr_s

            contexts.append(ctx)
        return contexts

    @torch.no_grad()
    def predict_sentence_probabilities(self, sentence_texts: List[str]) -> List[float]:
        """
        Extract per-sentence AI probabilities using sliding-window context embeddings.
        """
        if not sentence_texts:
            return []

        contexts = self.get_sentence_contexts(sentence_texts)
        
        inputs = self.tokenizer(
            contexts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        ).to(self.device)

        outputs = self.model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy().tolist()

        return [round(float(p), 4) for p in probs]

    def save_checkpoint(self, save_dir: str):
        """Save fine-tuned transformer model weights."""
        out_dir = Path(save_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(out_dir)
        self.tokenizer.save_pretrained(out_dir)
        print(f"Saved Transformer Classifier checkpoint to {out_dir}")
