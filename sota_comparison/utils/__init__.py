"""SOTA comparison utilities: BIO I/O, evaluation, and format conversion."""

from .bio_io import load_bio_file, save_predictions_bio, save_gold_bio
from .eval_ner import (
    evaluate_bio_predictions,
    compute_metrics,
    load_gold_and_pred_from_file,
)
from .convert_gliner import (
    bio_sentences_to_gliner_data,
    gliner_predictions_to_bio,
    load_bio_and_convert_to_gliner,
    ENTITY_TYPES,
)

__all__ = [
    "load_bio_file",
    "save_predictions_bio",
    "save_gold_bio",
    "evaluate_bio_predictions",
    "compute_metrics",
    "load_gold_and_pred_from_file",
    "bio_sentences_to_gliner_data",
    "gliner_predictions_to_bio",
    "load_bio_and_convert_to_gliner",
    "ENTITY_TYPES",
]
