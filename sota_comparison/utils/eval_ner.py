"""
Standard NER evaluation: seqeval IOB2, weighted avg F1/Precision/Recall.
All SOTA and baseline results must use this for fair comparison.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

try:
    from seqeval.metrics import classification_report
    from seqeval.scheme import IOB2
except ImportError:
    classification_report = None
    IOB2 = None


def compute_metrics(
    y_true: List[List[str]],
    y_pred: List[List[str]],
    *,
    scheme=IOB2,
    digits: int = 4,
    zero_division: float = 0,
) -> Dict:
    """
    Compute entity-level metrics using seqeval (IOB2).

    Returns:
        Dict with keys "weighted avg", "micro avg", per-class, etc.
        Primary reported: report["weighted avg"]["f1-score"], precision, recall.
    """
    if classification_report is None or IOB2 is None:
        raise ImportError("seqeval is required: pip install seqeval")
    report = classification_report(
        y_true,
        y_pred,
        scheme=scheme,
        output_dict=True,
        digits=digits,
        zero_division=zero_division,
    )

    def to_native(obj):
        if hasattr(obj, "item"):
            return obj.item()
        if isinstance(obj, dict):
            return {k: to_native(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_native(x) for x in obj]
        return obj

    return to_native(report)


def evaluate_bio_predictions(
    gold_sentences: List[Tuple[List[str], List[str]]],
    pred_tags_only: List[List[str]],
    *,
    output_dir: Optional[Path] = None,
    run_name: str = "eval",
) -> Dict:
    """
    Evaluate predictions against gold BIO.

    Args:
        gold_sentences: List of (tokens, gold_tags).
        pred_tags_only: List of predicted tag sequences (same order/length as gold).
        output_dir: If set, write report.txt and report.json.
        run_name: Prefix for output files.

    Returns:
        classification_report dict; use report["weighted avg"] for paper metrics.
    """
    y_true = [tags for _, tags in gold_sentences]
    if len(y_true) != len(pred_tags_only):
        raise ValueError(
            f"Number of sentences: gold={len(y_true)}, pred={len(pred_tags_only)}"
        )
    for i, (true_seq, pred_seq) in enumerate(zip(y_true, pred_tags_only)):
        if len(true_seq) != len(pred_seq):
            raise ValueError(
                f"Sentence {i}: length mismatch gold={len(true_seq)} pred={len(pred_seq)}"
            )
    report = compute_metrics(y_true, pred_tags_only)

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        report_str = classification_report(
            y_true,
            pred_tags_only,
            scheme=IOB2,
            digits=4,
            zero_division=0,
        )
        (output_dir / f"{run_name}_report.txt").write_text(report_str, encoding="utf-8")
        import json

        with open(output_dir / f"{run_name}_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    return report


def load_gold_and_pred_from_file(
    gold_bio_path: Union[str, Path],
    pred_bio_path: Union[str, Path],
) -> Tuple[List[Tuple[List[str], List[str]]], List[List[str]]]:
    """
    Load gold BIO file and prediction BIO file (token\\tpred_tag).
    Both files: one line per token (token\\ttag), blank line = sentence boundary.
    """
    from .bio_io import load_bio_file

    gold_sentences = load_bio_file(gold_bio_path)
    pred_sentences = load_bio_file(pred_bio_path)
    if len(gold_sentences) != len(pred_sentences):
        raise ValueError(
            f"Sentence count: gold={len(gold_sentences)}, pred={len(pred_sentences)}"
        )
    for i, ((_, gtags), (_, ptags)) in enumerate(zip(gold_sentences, pred_sentences)):
        if len(gtags) != len(ptags):
            raise ValueError(
                f"Sentence {i}: length mismatch gold={len(gtags)} pred={len(ptags)}"
            )
    pred_tags_only = [tags for _, tags in pred_sentences]
    return gold_sentences, pred_tags_only
