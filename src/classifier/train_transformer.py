"""
Fine-tuning script for local Context Transformer Classifier (PyTorch + HuggingFace).
Trains sequence classification head on 3-sentence sliding windows using local PyTorch training loop.
Saves fine-tuned checkpoint to models/transformer_sent_clf.
"""

import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Tuple
import pandas as pd
import numpy as np
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.classifier.split import held_out_topic_split

MODEL_NAME = "distilbert-base-uncased"
SAVE_DIR = Path("models/transformer_sent_clf")


class SentenceContextDataset(Dataset):
    """PyTorch Dataset for sentence context sliding windows."""

    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def build_sliding_window_contexts(df: pd.DataFrame) -> List[str]:
    """Group by essay_id and construct 3-sentence sliding window texts."""
    contexts = []
    for essay_id, group in df.groupby('essay_id', sort=False):
        sents = group['text'].tolist()
        n = len(sents)
        for i in range(n):
            prev_s = sents[i - 1] if i > 0 else ""
            curr_s = sents[i]
            next_s = sents[i + 1] if i < n - 1 else ""

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


def main():
    print("=" * 60)
    print("FINE-TUNING LOCAL CONTEXT TRANSFORMER CLASSIFIER")
    print("=" * 60)

    features_path = Path("data/features.csv")
    if not features_path.exists():
        print(f"Features file not found at {features_path}. Run prepare_data.py first.")
        return

    df = pd.read_csv(features_path)
    train_df, test_df, held_out = held_out_topic_split(df)

    print(f"Train sentences: {len(train_df)} | Test sentences: {len(test_df)}")

    # Extract 3-sentence sliding window contexts and binary labels
    train_contexts = build_sliding_window_contexts(train_df)
    test_contexts = build_sliding_window_contexts(test_df)

    train_labels = (train_df['label'] == 'ai_generated').astype(int).tolist()
    test_labels = (test_df['label'] == 'ai_generated').astype(int).tolist()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(device)

    train_dataset = SentenceContextDataset(train_contexts, train_labels, tokenizer)
    test_dataset = SentenceContextDataset(test_contexts, test_labels, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    epochs = 2
    optimizer = AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)

    print("Beginning PyTorch sequence classifier training...", flush=True)
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

            if (step + 1) % 25 == 0:
                print(f"Epoch {epoch + 1}/{epochs} | Step {step + 1}/{len(train_loader)} | Loss: {loss.item():.4f}", flush=True)

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs} Complete - Avg Loss: {avg_loss:.4f}", flush=True)

    # Evaluate on test set
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total if total > 0 else 0.0
    print(f"\nLocal Transformer Test Accuracy: {accuracy * 100:.2f}% ({correct}/{total})", flush=True)

    # Save fine-tuned checkpoint locally
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)
    print(f"Successfully saved fine-tuned Transformer model to {SAVE_DIR}", flush=True)


if __name__ == '__main__':
    main()
