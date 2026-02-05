"""
BIO format I/O for IDNer2k and SOTA comparison.
Format: one line per token = "token<TAB>tag", blank line = sentence boundary.
"""

from pathlib import Path
from typing import List, Tuple


def load_bio_file(file_path: Path) -> List[Tuple[List[str], List[str]]]:
    """
    Load BIO-format NER dataset from file.

    Args:
        file_path: Path to BIO file (token\\ttag per line, blank line = new sentence).

    Returns:
        List of (tokens, tags) per sentence.
    """
    file_path = Path(file_path)
    sentences = []
    tokens, tags = [], []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if tokens:
                    sentences.append((tokens, tags))
                    tokens, tags = [], []
            else:
                parts = line.split()
                if len(parts) == 2:
                    tokens.append(parts[0])
                    tags.append(parts[1])
        if tokens:
            sentences.append((tokens, tags))
    return sentences


def save_predictions_bio(
    file_path: Path,
    sentences: List[Tuple[List[str], List[str], List[str]]],
    *,
    gold_included: bool = True,
) -> None:
    """
    Save predictions in BIO format.

    Args:
        file_path: Output path.
        sentences: List of (tokens, gold_tags, pred_tags). If gold_included is False,
                   each element is (tokens, pred_tags) and file is token\\tpred_tag only.
        gold_included: If True, write token\\tgold\\tpred; else token\\tpred.
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        for item in sentences:
            if gold_included and len(item) == 3:
                toks, gold, pred = item
                for t, g, p in zip(toks, gold, pred):
                    f.write(f"{t}\t{g}\t{p}\n")
            else:
                toks = item[0]
                pred = item[1]
                for t, p in zip(toks, pred):
                    f.write(f"{t}\t{p}\n")
            f.write("\n")


def save_gold_bio(
    file_path: Path, sentences: List[Tuple[List[str], List[str]]]
) -> None:
    """Save gold-only BIO (tokens + tags) for use as reference."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        for tokens, tags in sentences:
            for t, g in zip(tokens, tags):
                f.write(f"{t}\t{g}\n")
            f.write("\n")
