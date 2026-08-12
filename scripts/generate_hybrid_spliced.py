"""
Script to create hybrid spliced essays (mixing human and AI paragraphs/sentences).
This produces ground-truth labeled hybrid essays for training context-aware classifiers.
"""

import os
import random
from pathlib import Path
from typing import List, Tuple

HUMAN_NATIVE_DIR = Path("data/human_native")
HUMAN_ESL_DIR = Path("data/human_esl")
AI_DIR = Path("data/ai_generated")
HYBRID_DIR = Path("data/hybrid")

random.seed(42)


def read_essay(filepath: Path) -> Tuple[str, str]:
    """Read essay text and topic metadata."""
    content = filepath.read_text(encoding="utf-8")
    lines = content.split("\n")
    topic = "unknown"
    essay_lines = []
    for l in lines:
        if l.startswith("# topic:"):
            topic = l.split(":", 1)[1].strip()
        elif not l.startswith("#"):
            essay_lines.append(l)
    return "\n".join(essay_lines).strip(), topic


def split_paragraphs(text: str) -> List[str]:
    """Split text into non-empty paragraphs."""
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paras:
        paras = [p.strip() for p in text.split("\n") if p.strip()]
    return paras


def main():
    HYBRID_DIR.mkdir(parents=True, exist_ok=True)
    
    human_files = sorted(list(HUMAN_NATIVE_DIR.glob("*.txt")) + list(HUMAN_ESL_DIR.glob("*.txt")))
    ai_files = sorted(list(AI_DIR.glob("*.txt")))

    print(f"Found {len(human_files)} human essays and {len(ai_files)} AI essays.")

    num_hybrids = min(30, len(human_files), len(ai_files))
    print(f"Creating {num_hybrids} spliced hybrid essays...")

    for i in range(num_hybrids):
        h_file = human_files[i % len(human_files)]
        a_file = ai_files[i % len(ai_files)]

        h_text, h_topic = read_essay(h_file)
        a_text, _ = read_essay(a_file)

        h_paras = split_paragraphs(h_text)
        a_paras = split_paragraphs(a_text)

        # Interleave paragraphs: e.g. Human P1, AI P1, Human P2, AI P2
        spliced_paras = []
        max_p = max(len(h_paras), len(a_paras))
        for p_idx in range(max_p):
            if p_idx % 2 == 0:
                if p_idx < len(h_paras):
                    spliced_paras.append(h_paras[p_idx])
                elif p_idx < len(a_paras):
                    spliced_paras.append(a_paras[p_idx])
            else:
                if p_idx < len(a_paras):
                    spliced_paras.append(a_paras[p_idx])
                elif p_idx < len(h_paras):
                    spliced_paras.append(h_paras[p_idx])

        spliced_text = "\n\n".join(spliced_paras)

        out_path = HYBRID_DIR / f"hybrid_spliced_{i+1:03d}.txt"
        header = f"""# category: hybrid
# source_human: {h_file.name}
# source_ai: {a_file.name}
# topic: {h_topic}

"""
        out_path.write_text(header + spliced_text, encoding="utf-8")

    print(f"Successfully generated {num_hybrids} hybrid spliced essays in {HYBRID_DIR}")


if __name__ == "__main__":
    main()
