#!/usr/bin/env python3
"""
Standalone evaluator: gold BIO + pred BIO -> seqeval IOB2 weighted avg.
Usage:
  python evaluate_idner2k_standard.py --gold ../data/idner2k/test_bio.txt --pred path/to/pred_bio.txt [--output_dir dir]
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.eval_ner import (
    load_gold_and_pred_from_file,
    evaluate_bio_predictions,
    compute_metrics,
)


def main():
    p = argparse.ArgumentParser(
        description="Evaluate NER predictions (BIO) with seqeval IOB2 weighted avg."
    )
    p.add_argument("--gold", required=True, help="Path to gold BIO file (token\\ttag).")
    p.add_argument(
        "--pred", required=True, help="Path to prediction BIO file (token\\tpred_tag)."
    )
    p.add_argument(
        "--output_dir",
        default=None,
        help="Directory to write report.txt and report.json.",
    )
    args = p.parse_args()

    gold_path = Path(args.gold).resolve()
    pred_path = Path(args.pred).resolve()
    if not gold_path.exists():
        print(f"Gold file not found: {gold_path}", file=sys.stderr)
        sys.exit(1)
    if not pred_path.exists():
        print(f"Pred file not found: {pred_path}", file=sys.stderr)
        sys.exit(1)

    gold_sentences, pred_tags = load_gold_and_pred_from_file(gold_path, pred_path)
    report = evaluate_bio_predictions(
        gold_sentences,
        pred_tags,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        run_name="idner2k_standard",
    )

    w = report.get("weighted avg", {})
    f1 = w.get("f1-score", 0)
    prec = w.get("precision", 0)
    rec = w.get("recall", 0)
    print(f"Weighted F1:  {f1:.4f}")
    print(f"Precision:    {prec:.4f}")
    print(f"Recall:       {rec:.4f}")
    if args.output_dir:
        print(f"Reports saved to {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
