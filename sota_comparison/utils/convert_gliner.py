"""
BIO <-> GLiNER format conversion for IDNer2k.
GLiNER expects: list of dicts with "tokens" (list of str) and "entities" (list of {start, end, label}).
Indices are token-level, inclusive.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple

from .bio_io import load_bio_file


BIO_TO_LABEL = {
    "B-PER": "PER",
    "I-PER": "PER",
    "B-ORG": "ORG",
    "I-ORG": "ORG",
    "B-LOC": "LOC",
    "I-LOC": "LOC",
}
ENTITY_TYPES = ["PER", "ORG", "LOC"]

LABEL_TO_GLINER_TRAIN = {"PER": "person", "ORG": "organization", "LOC": "location"}


def bio_sentences_to_gliner_train_data(
    sentences: List[Tuple[List[str], List[str]]],
) -> List[Dict[str, Any]]:
    """
    Convert BIO sentences to GLiNER train_model format.

    Returns:
        List of {"tokenized_text": [...], "ner": [[start, end, label], ...]}
        with token-level inclusive indices and lowercase labels (person, organization, location).
    """
    out = []
    for tokens, tags in sentences:
        ner = []
        i = 0
        while i < len(tags):
            tag = tags[i]
            if tag.startswith("B-"):
                label = tag[2:]
                start = i
                i += 1
                while i < len(tags) and tags[i] == f"I-{label}":
                    i += 1
                end = i - 1
                gliner_label = LABEL_TO_GLINER_TRAIN.get(label, label.lower())
                ner.append([start, end, gliner_label])
            else:
                i += 1
        out.append({"tokenized_text": tokens, "ner": ner})
    return out


def bio_sentences_to_gliner_data(
    sentences: List[Tuple[List[str], List[str]]],
) -> List[Dict[str, Any]]:
    """
    Convert BIO sentences to GLiNER training format.

    Args:
        sentences: List of (tokens, tags) in BIO.

    Returns:
        List of {"tokens": [...], "entities": [{"start": int, "end": int, "label": str}]}.
        start, end are token indices (0-based, inclusive).
    """
    out = []
    for tokens, tags in sentences:
        entities = []
        i = 0
        while i < len(tags):
            tag = tags[i]
            if tag.startswith("B-"):
                label = tag[2:]
                start = i
                i += 1
                while i < len(tags) and tags[i] == f"I-{label}":
                    i += 1
                end = i - 1
                entities.append({"start": start, "end": end, "label": label})
            else:
                i += 1
        out.append({"tokens": tokens, "entities": entities})
    return out


def gliner_predictions_to_bio(
    tokens_per_sentence: List[List[str]],
    entities_per_sentence: List[List[Dict[str, Any]]],
) -> List[List[str]]:
    """
    Convert GLiNER span predictions to token-level BIO tags.

    Args:
        tokens_per_sentence: List of token lists (one per sentence).
        entities_per_sentence: List of entity lists; each entity has "start", "end" (token indices, inclusive), "label".

    Returns:
        List of tag sequences (BIO) per sentence.
    """
    pred_tags = []
    for tokens, entities in zip(tokens_per_sentence, entities_per_sentence):
        n = len(tokens)
        tags = ["O"] * n
        for ent in entities:
            start = ent.get("start", 0)
            end = ent.get("end", start)
            label = ent.get("label", "O").upper()
            if label in ("PER", "ORG", "LOC"):
                start = max(0, min(start, n - 1))
                end = max(start, min(end, n - 1))
                tags[start] = f"B-{label}"
                for j in range(start + 1, end + 1):
                    tags[j] = f"I-{label}"
        pred_tags.append(tags)
    return pred_tags


def load_bio_and_convert_to_gliner(bio_path: Path) -> List[Dict[str, Any]]:
    """Load BIO file and return GLiNER-format data."""
    sentences = load_bio_file(bio_path)
    return bio_sentences_to_gliner_data(sentences)


def gliner_entities_from_text_spans(
    text: str,
    entities: List[Dict[str, Any]],
    tokens: List[str],
) -> List[Dict[str, Any]]:
    """
    Map GLiNER entities (character spans: text, start_char, end_char) to token indices.
    GLiNER predict_entities often returns {"text", "label", "start", "end"} in characters.
    We need token-level start/end for BIO. Build character offsets for tokens, then map.
    """
    offset = 0
    token_spans = []
    for t in tokens:
        start = offset
        idx = text.find(t, offset)
        if idx >= 0:
            start = idx
            offset = idx + len(t)
        else:
            offset = start + len(t) + 1  # +1 for space
        token_spans.append((start, offset))
    result = []
    for ent in entities:
        c_start = ent.get("start", 0)
        c_end = ent.get("end", c_start)
        label = ent.get("label", "O")
        tok_start, tok_end = None, None
        for i, (s, e) in enumerate(token_spans):
            if s <= c_start < e or (tok_start is None and s <= c_start):
                tok_start = i
            if s < c_end <= e or (tok_end is None and c_end <= e):
                tok_end = i
                break
        if tok_start is None:
            tok_start = 0
        if tok_end is None:
            tok_end = len(tokens) - 1
        result.append({"start": tok_start, "end": tok_end, "label": label})
    return result
