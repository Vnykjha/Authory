#!/usr/bin/env python
"""
Data preparation script for AI Essay Detector.
Extracts and saves essays from various sources into organized directories.
"""
import os
import json
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

# Set up paths
DATA_DIR = Path("data")
HUMAN_NATIVE_DIR = DATA_DIR / "human_native"
HUMAN_ESL_DIR = DATA_DIR / "human_esl"
AI_GENERATED_DIR = DATA_DIR / "ai_generated"
HYBRID_DIR = DATA_DIR / "hybrid"

for d in [HUMAN_NATIVE_DIR, HUMAN_ESL_DIR, AI_GENERATED_DIR, HYBRID_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def save_essay(text: str, filepath: Path, metadata: dict = None):
    """Save essay text to file with optional metadata header."""
    content = text.strip()
    if metadata:
        meta_lines = [f"# {k}: {v}" for k, v in metadata.items()]
        content = "\n".join(meta_lines) + "\n\n" + content
    filepath.write_text(content, encoding="utf-8")


def prepare_human_native():
    """Prepare human native English essays from Fake Essay Detection dataset."""
    print("Loading Fake Essay Detection dataset (human essays)...")
    ds = load_dataset("ShashiVish/Fake-Essay-Detection-Dataset", split="train")

    human_essays = [x for x in ds if x["label"] == 0]
    print(f"Found {len(human_essays)} human essays")

    # Take first 80 for our target of 60-100
    selected = human_essays[:80]

    for i, essay in enumerate(tqdm(selected, desc="Saving human native essays")):
        filepath = HUMAN_NATIVE_DIR / f"essay_{i+1:03d}.txt"
        save_essay(
            essay["text"],
            filepath,
            metadata={
                "source": "ShashiVish/Fake-Essay-Detection-Dataset",
                "original_id": essay["id"],
                "label": "human",
                "category": "human_native"
            }
        )

    print(f"Saved {len(selected)} human native essays to {HUMAN_NATIVE_DIR}")
    return len(selected)


def prepare_human_esl():
    """Prepare ESL human essays from IELTS dataset."""
    print("Loading IELTS Writing Task 2 essays...")
    ds = load_dataset("chillies/ielts-writing-task2-essays", split="train")

    print(f"Found {len(ds)} IELTS essays")

    # Take first 20 for our target of 15-20
    selected = ds.select(range(20))

    for i, essay in enumerate(tqdm(selected, desc="Saving ESL essays")):
        filepath = HUMAN_ESL_DIR / f"essay_{i+1:03d}.txt"
        save_essay(
            essay["essay_text"],
            filepath,
            metadata={
                "source": "chillies/ielts-writing-task2-essays",
                "original_id": essay["essay_id"],
                "topic": essay["topic"],
                "question": essay["question"],
                "overall_band": essay["overall_band"],
                "label": "human",
                "category": "human_esl"
            }
        )

    print(f"Saved {len(selected)} ESL essays to {HUMAN_ESL_DIR}")
    return len(selected)


def prepare_ai_generated():
    """Prepare AI-generated essays from Fake Essay Detection dataset."""
    print("Loading Fake Essay Detection dataset (AI essays)...")
    ds = load_dataset("ShashiVish/Fake-Essay-Detection-Dataset", split="train")

    ai_essays = [x for x in ds if x["label"] == 1]
    print(f"Found {len(ai_essays)} AI essays")

    # Take first 80 for our target of 60-100
    selected = ai_essays[:80]

    for i, essay in enumerate(tqdm(selected, desc="Saving AI essays")):
        filepath = AI_GENERATED_DIR / f"essay_{i+1:03d}.txt"
        save_essay(
            essay["text"],
            filepath,
            metadata={
                "source": "ShashiVish/Fake-Essay-Detection-Dataset",
                "original_id": essay["id"],
                "label": "ai",
                "category": "ai_generated"
            }
        )

    print(f"Saved {len(selected)} AI essays to {AI_GENERATED_DIR}")
    return len(selected)


def prepare_hybrid():
    """Prepare hybrid essays by polishing human essays with an LLM."""
    # We'll use a local LLM via transformers to polish some human essays
    # For now, create placeholder - will implement with actual LLM later
    print("Preparing hybrid essays (placeholder - will generate with LLM)...")

    # Load a few human native essays to use as base
    human_files = sorted(HUMAN_NATIVE_DIR.glob("*.txt"))[:15]

    count = 0
    for human_file in human_files:
        # Read the original essay (skip metadata lines)
        content = human_file.read_text(encoding="utf-8")
        # Extract just the essay text (after metadata)
        if content.startswith("#"):
            lines = content.split("\n")
            # Find first non-metadata line
            essay_text = "\n".join([l for l in lines if not l.startswith("#")]).strip()
        else:
            essay_text = content

        # For now, save the original as placeholder
        # In real implementation, we'd run this through an LLM
        filepath = HYBRID_DIR / f"essay_{count+1:03d}.txt"
        save_essay(
            essay_text,
            filepath,
            metadata={
                "source": "hybrid_generation_placeholder",
                "base_essay": human_file.name,
                "polish_model": "TODO: run through LLM",
                "prompt": "Improve this college admissions essay for clarity, flow, and impact. Keep the same core story and voice.",
                "category": "hybrid"
            }
        )
        count += 1

    print(f"Saved {count} hybrid placeholder essays to {HYBRID_DIR}")
    return count


def generate_dataset_documentation(counts: dict):
    """Generate docs/dataset.md with full provenance."""
    doc_path = Path("docs/dataset.md")
    doc_path.parent.mkdir(exist_ok=True)

    content = f"""# Dataset Documentation

## Summary
| Category | Count | Source Details |
|---|---|---|
| Human (Native) | {counts.get('human_native', 0)} | ShashiVish/Fake-Essay-Detection-Dataset (human-labeled essays) |
| Human (ESL) | {counts.get('human_esl', 0)} | chillies/ielts-writing-task2-essays (IELTS Writing Task 2) |
| AI-Generated | {counts.get('ai_generated', 0)} | ShashiVish/Fake-Essay-Detection-Dataset (AI-labeled essays) |
| Hybrid | {counts.get('hybrid', 0)} | Generated from human_native subset via LLM polish (placeholder) |

## Human (Native) — Sources
- **Source**: ShashiVish/Fake-Essay-Detection-Dataset (Hugging Face)
- **URL**: https://huggingface.co/datasets/ShashiVish/Fake-Essay-Detection-Dataset
- **Date accessed**: 2026-08-05
- **Essay count**: {counts.get('human_native', 0)}
- **Description**: Dataset containing human-written essays labeled as "0" (human). These are general academic/argumentative essays, not specifically college admissions essays. Used as proxy for native English human writing.
- **Topics covered**: Various (education, technology, society, environment, etc.)

## Human (ESL) — Sources
- **Source**: chillies/ielts-writing-task2-essays (Hugging Face)
- **URL**: https://huggingface.co/datasets/chillies/ielts-writing-task2-essays
- **Date accessed**: 2026-08-05
- **Essay count**: {counts.get('human_esl', 0)}
- **Description**: IELTS Writing Task 2 essays written by non-native English speakers. Includes band scores (5.0-7.5 range typical), topics, and prompts.
- **L1 languages represented**: Mixed (international IELTS test-takers from diverse language backgrounds)
- **Band score range**: ~5.0-7.0 (based on sample inspection)

## AI-Generated — Generation Log
| Essay ID | Model | Prompt Style | Temperature | Topic |
|---|---|---|---|---|
| essay_001 - essay_080 | Unknown (from Fake-Essay-Detection-Dataset) | Various | Unknown | Various |

**Note**: The AI-generated essays in this dataset were sourced from ShashiVish/Fake-Essay-Detection-Dataset which contains pre-generated AI essays. The exact generation models, prompts, and parameters are not documented in the dataset. For a more controlled experiment, future work should generate AI essays with known models (GPT-3.5, GPT-4, Claude, Llama) and documented prompts.

## Hybrid — Generation Log
| Essay ID | Source Human ID | Polish Model | Prompt |
|---|---|---|---|
| essay_001 - essay_015 | human_native/essay_001.txt - essay_015.txt | TODO: Local LLM (e.g., Llama-3-8B-Instruct) | "Improve this college admissions essay for clarity, flow, and impact. Keep the same core story and voice." |

**Note**: Hybrid essays are currently placeholders. They need to be generated by running the base human essays through an LLM with the polish prompt.

## Topics Covered
Based on the datasets used:

**Human (Native) - Fake Essay Detection:**
- Urban planning / car-free cities
- Electoral systems
- Education policy
- Technology and society
- Environmental issues
- Health and lifestyle

**Human (ESL) - IELTS Writing Task 2:**
- Art and culture
- Technology and communication
- Education and learning
- Work and employment
- Environment and sustainability
- Government and policy
- Social issues

**AI-Generated - Fake Essay Detection:**
- Similar topic distribution to human essays (same dataset)

**Hybrid:**
- Same as source human essays (subset of Fake Essay Detection topics)

## Known Gaps
1. **Not college admissions essays**: The human native and AI essays are general academic/argumentative essays, not specifically college admissions personal statements. This limits ecological validity for the target use case.

2. **ESL subset limited to IELTS format**: The ESL essays are all IELTS Writing Task 2 format (academic argumentative), not college admissions personal narratives. The L1 language backgrounds are mixed but not documented per essay.

3. **AI generation metadata missing**: The AI essays from Fake-Essay-Detection-Dataset lack generation metadata (model, prompt, temperature). This prevents analysis of how different models/prompts affect detectability.

4. **Hybrid essays not yet generated**: The hybrid set is currently placeholders. Need to run human essays through an LLM with a polish prompt to create realistic hybrid samples.

5. **Topic overlap between train/test**: The Fake Essay Detection dataset doesn't provide topic labels for easy held-out-topic splitting. Need to extract topics manually or use a topic model.

6. **No UK UCAS or other systems**: All human essays (where identifiable) appear to be US-centric or international academic style. No UK UCAS personal statements or other national admission systems represented.

7. **Dataset size modest**: 80 per class is relatively small for training robust classifiers. Consider expanding with additional sources.

8. **Potential label noise**: The Fake Essay Detection dataset's labeling methodology is not fully transparent. Some "human" essays may be AI-assisted and vice versa.
"""
    doc_path.write_text(content, encoding="utf-8")
    print(f"Generated dataset documentation at {doc_path}")


def main():
    print("=" * 60)
    print("AI Essay Detector - Data Preparation")
    print("=" * 60)

    counts = {}

    # Prepare each category
    counts["human_native"] = prepare_human_native()
    print()
    counts["human_esl"] = prepare_human_esl()
    print()
    counts["ai_generated"] = prepare_ai_generated()
    print()
    counts["hybrid"] = prepare_hybrid()
    print()

    # Generate documentation
    generate_dataset_documentation(counts)

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for cat, count in counts.items():
        print(f"  {cat}: {count} essays")
    print()
    print("Dataset documentation: docs/dataset.md")
    print("Done!")


if __name__ == "__main__":
    main()