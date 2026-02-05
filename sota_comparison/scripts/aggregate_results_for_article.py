#!/usr/bin/env python3
"""
Aggregate SOTA summary JSONs into a single table for the article.
Reads outputs/gliner/summary_test.json, outputs/nusabert/summary_test.json,
adds TAMSA and IndoBERT-CRF from manuscript Table 2, prints full table per docs/ARTICLE_TABLE.md.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"

ARTICLE_TAMSA = (
    "TAMSA (Ours)",
    92.54,
    92.19,
    93.03,
    "Fine-tuned on IDNer2k; token-aware multi-source attention.",
)
ARTICLE_INDOBERT = (
    "IndoBERT-CRF",
    91.24,
    91.55,
    91.07,
    "Baseline; same splits and metrics.",
)
ARTICLE_NOTES = {
    "GLiNER": "Zero-shot on IDNer2k test; span-based, predictions converted to BIO for evaluation.",
    "NusaBERT-NER": "Inference on IDNer2k test; token-level; same evaluation protocol.",
}


def main():
    rows = []
    for name, subdir in [
        ("GLiNER", "gliner"),
        ("NusaBERT-NER", "nusabert"),
    ]:
        p = OUTPUTS / subdir / "summary_test.json"
        if not p.exists():
            rows.append(
                (
                    name,
                    None,
                    None,
                    None,
                    ARTICLE_NOTES.get(name, "No summary (run not executed)"),
                )
            )
            continue
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        w = data.get("weighted_avg", {})
        if not w:
            rows.append(
                (
                    name,
                    None,
                    None,
                    None,
                    ARTICLE_NOTES.get(name, data.get("note", "No weighted_avg")),
                )
            )
            continue
        f1 = w.get("f1-score")
        prec = w.get("precision")
        rec = w.get("recall")
        if f1 is not None:
            rows.append(
                (
                    name,
                    f1 * 100,
                    prec * 100 if prec is not None else None,
                    rec * 100 if rec is not None else None,
                    ARTICLE_NOTES.get(name, ""),
                )
            )
        else:
            rows.append(
                (name, None, None, None, ARTICLE_NOTES.get(name, data.get("note", "")))
            )

    rows.append(ARTICLE_TAMSA)
    rows.append(ARTICLE_INDOBERT)

    def f1_sort_key(r):
        _, f1, _, _, _ = r
        return (f1 is None, f1 if f1 is not None else 0.0)

    rows.sort(key=f1_sort_key)

    print("Table for article (weighted avg, same dataset/splits/metrics):")
    print()
    print(
        "| Method         | F1 (%) | Precision (%) | Recall (%) | Notes                                                                               |"
    )
    print(
        "|----------------|--------|----------------|------------|-------------------------------------------------------------------------------------|"
    )
    for name, f1, prec, rec, notes in rows:
        f1_s = f"{f1:.2f}" if f1 is not None else "-"
        prec_s = f"{prec:.2f}" if prec is not None else "-"
        rec_s = f"{rec:.2f}" if rec is not None else "-"
        name_pad = name.ljust(14) if len(name) <= 14 else (name[:11] + "...").ljust(14)
        print(f"| {name_pad} | {f1_s:>6} | {prec_s:>14} | {rec_s:>10} | {notes} |")
    print()
    print(
        "TAMSA & IndoBERT-CRF: manuscript Table 2. GLiNER & NusaBERT-NER: run outputs. Notes: docs/ARTICLE_TABLE.md"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
