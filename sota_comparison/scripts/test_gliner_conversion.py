#!/usr/bin/env python3
"""
Test GLiNER conversion without a model: gold BIO -> char spans -> mock predictions -> BIO -> evaluate.
If F1 is high, conversion pipeline and evaluator are correct. Run: python scripts/test_gliner_conversion.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.bio_io import load_bio_file
from utils.eval_ner import evaluate_bio_predictions


def gold_bio_to_char_spans(tokens, tags, token_spans):
    """Convert gold BIO to list of {start, end, text, label} as GLiNER would return. token_spans = _token_char_spans(tokens, text)."""
    entities = []
    i = 0
    while i < len(tags):
        tag = tags[i]
        if tag.startswith("B-"):
            label = tag[2:]
            j = i + 1
            while j < len(tags) and tags[j] == f"I-{label}":
                j += 1
            start_char = token_spans[i][0]
            end_char = token_spans[j - 1][1]
            ent_text = " ".join(tokens[i:j])
            gliner_label = (
                "person"
                if label == "PER"
                else "organization" if label == "ORG" else "location"
            )
            entities.append(
                {
                    "start": start_char,
                    "end": end_char,
                    "text": ent_text,
                    "label": gliner_label,
                }
            )
            i = j
        else:
            i += 1
    return entities


def gold_to_gliner_mock(sentences):
    """For each sentence, build gold entities (char spans) as GLiNER would return."""
    from run_gliner_idner2k import _token_char_spans

    mock_entities_per_sent = []
    for tokens, tags in sentences:
        text = " ".join(tokens)
        token_spans = _token_char_spans(tokens, text)
        mock_entities_per_sent.append(gold_bio_to_char_spans(tokens, tags, token_spans))
    return mock_entities_per_sent


def main():
    sys.path.insert(0, str(ROOT / "scripts"))
    data_root = ROOT / ".." / "data" / "idner2k"
    test_file = data_root.resolve() / "test_bio.txt"
    if not test_file.exists():
        print("SKIP: test_bio.txt not found at", test_file)
        return 0
    sentences = load_bio_file(test_file)[:10]
    if not sentences:
        print("SKIP: no sentences")
        return 0

    from run_gliner_idner2k import (
        _token_char_spans,
        _char_span_to_token_span,
        _normalize_gliner_label,
        predict_and_convert_to_bio,
    )

    mock_entities = gold_to_gliner_mock(sentences)
    call_idx = [0]

    class MockModel:
        def predict_entities(self, text, labels, threshold=0.5):
            i = call_idx[0]
            call_idx[0] += 1
            if i < len(mock_entities):
                return mock_entities[i]
            return []

    model = MockModel()
    pred_tags = predict_and_convert_to_bio(
        model, sentences, ["person", "organization", "location"], threshold=0.3
    )

    report = evaluate_bio_predictions(sentences, pred_tags)
    w = report.get("weighted avg", {})
    f1 = w.get("f1-score", 0)
    prec = w.get("precision", 0)
    rec = w.get("recall", 0)

    print("Test: gold -> mock GLiNER spans -> convert to BIO -> evaluate")
    print(f"  Weighted F1:  {f1:.4f}")
    print(f"  Precision:    {prec:.4f}")
    print(f"  Recall:       {rec:.4f}")

    if f1 < 0.9:
        print("  FAIL: expected F1 ~1.0 when mock equals gold (check conversion)")
        return 1
    print("  OK: conversion pipeline works.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
