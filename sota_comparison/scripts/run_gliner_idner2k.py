#!/usr/bin/env python3
"""
Run GLiNER on IDNer2k: load test set, predict with GLiNER (zero-shot or fine-tuned),
convert predictions to BIO, evaluate with standard evaluator.
Uses same dataset paths and metrics as TAMSA/baselines for fair comparison.
"""
import argparse
import json
import logging
import os
import sys
import warnings
from pathlib import Path

# Suppress transformers truncation warning when GLiNER tokenizer has no max_length.
warnings.filterwarnings(
    "ignore",
    message=".*truncate.*max_length.*no maximum length.*",
    category=UserWarning,
)
logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.bio_io import load_bio_file
from utils.eval_ner import evaluate_bio_predictions

# Map GLiNER output labels to IDNer2k PER/ORG/LOC for seqeval (same scheme as TAMSA).
GLINER_LABEL_TO_IDNER2K = {
    "PERSON": "PER",
    "ORGANIZATION": "ORG",
    "LOCATION": "LOC",
    "ORG": "ORG",
    "LOC": "LOC",
    "PER": "PER",
}
GLINER_INPUT_LABELS = ["person", "organization", "location"]


def _normalize_gliner_label(label: str) -> str:
    """Map GLiNER entity label to IDNer2k type (PER, ORG, LOC)."""
    u = (label or "").strip().upper()
    return GLINER_LABEL_TO_IDNER2K.get(u, u if u in ("PER", "ORG", "LOC") else "")


def _token_char_spans(tokens: list, text: str) -> list:
    """Build (start_char, end_char) for each token. Assumes text == ' '.join(tokens)."""
    spans = []
    pos = 0
    for t in tokens:
        start = pos
        end = pos + len(t)
        spans.append((start, end))
        pos = end
        if pos < len(text) and text[pos] == " ":
            pos += 1
    return spans


def _char_span_to_token_span(c_start: int, c_end: int, token_spans: list) -> tuple:
    """Map character span (c_start, c_end) to (start_token, end_token) inclusive. Returns (-1,-1) if invalid."""
    n = len(token_spans)
    if not n or c_end <= c_start:
        return (-1, -1)
    start_t = None
    end_t = None
    for i, (s, e) in enumerate(token_spans):
        if s < c_end and e > c_start:
            if start_t is None:
                start_t = i
            end_t = i
    if start_t is None or end_t is None:
        return (-1, -1)
    return (start_t, end_t)


def entity_text_to_token_spans(tokens: list, entity_text: str, text: str) -> tuple:
    """Find (start_idx, end_idx) token span for entity text (inclusive). Fallback when char spans not used. Returns (-1, -1) if no match."""
    def norm(s):
        return " ".join((s or "").split()).strip().lower()

    text_lower = norm(entity_text)
    if not text_lower:
        return (-1, -1)
    n = len(tokens)
    for start in range(n):
        for end in range(start, n):
            span_text = norm(" ".join(tokens[start : end + 1]))
            if span_text == text_lower:
                return (start, end)
    for start in range(n):
        for end in range(start, n):
            span_text = norm(" ".join(tokens[start : end + 1]))
            if span_text in text_lower or text_lower in span_text:
                return (start, end)
    return (-1, -1)


def _ent_get(ent, key: str, default=None):
    """Get value from entity that may be dict or object with attributes."""
    if isinstance(ent, dict):
        return ent.get(key, default)
    return getattr(ent, key, default)


def predict_and_convert_to_bio(
    model,
    sentences: list,
    labels: list,
    threshold: float = 0.5,
    progress_interval: int = 50,
    debug: bool = False,
) -> list:
    """Run GLiNER predict_entities on each sentence; use char spans (start/end) to align to tokens, then convert to BIO. Same evaluator as TAMSA (seqeval IOB2)."""
    pred_tags = []
    n = len(sentences)
    for i, (tokens, _) in enumerate(sentences):
        if progress_interval and (i + 1) % progress_interval == 0:
            print(f"  GLiNER progress: {i + 1}/{n} sentences", flush=True)
        text = " ".join(tokens)
        try:
            entities = model.predict_entities(text, labels, threshold=threshold)
        except Exception as e:
            entities = []
            if debug and i < 2:
                print(f"  [DEBUG] predict_entities exception: {e}", flush=True)
        if not isinstance(entities, list):
            entities = []
        if entities and isinstance(entities[0], list):
            entities = entities[0] if entities[0] else []
        if debug and i < 2:
            print(f"  [DEBUG] sentence {i}: text={text[:60]}...", flush=True)
            print(f"  [DEBUG] sentence {i}: n_entities={len(entities)}", flush=True)
            for ei, ent in enumerate(entities[:5]):
                lab = _ent_get(ent, "label")
                st = _ent_get(ent, "start")
                en = _ent_get(ent, "end")
                tx = _ent_get(ent, "text")
                print(
                    f"  [DEBUG]   ent[{ei}] label={lab!r} start={st} end={en} text={tx!r}",
                    flush=True,
                )
        tags = ["O"] * len(tokens)
        token_spans = _token_char_spans(tokens, text)
        for ent in entities:
            label = _normalize_gliner_label(_ent_get(ent, "label", ""))
            if label not in ("PER", "ORG", "LOC"):
                continue
            c_start = _ent_get(ent, "start")
            c_end = _ent_get(ent, "end")
            start_t, end_t = -1, -1
            if c_start is not None and c_end is not None:
                try:
                    cs, ce = int(c_start), int(c_end)
                    max_c = len(text)
                    if cs < 0:
                        cs = 0
                    if ce > max_c:
                        ce = max_c
                    if cs < ce:
                        start_t, end_t = _char_span_to_token_span(cs, ce, token_spans)
                except (TypeError, ValueError):
                    pass
            if start_t < 0 or end_t < 0:
                start_t, end_t = entity_text_to_token_spans(
                    tokens, _ent_get(ent, "text", "") or "", text
                )
            if debug and i < 2 and entities:
                print(
                    f"  [DEBUG]   -> label={label} start_t={start_t} end_t={end_t} len(tags)={len(tags)}",
                    flush=True,
                )
            if start_t >= 0 and end_t >= 0 and start_t <= end_t and end_t < len(tags):
                tags[start_t] = f"B-{label}"
                for j in range(start_t + 1, end_t + 1):
                    tags[j] = f"I-{label}"
        pred_tags.append(tags)
    return pred_tags


def main():
    p = argparse.ArgumentParser(description="Run GLiNER on IDNer2k and evaluate.")
    p.add_argument(
        "--data_root",
        default=str(ROOT / ".." / "data" / "idner2k"),
        help="Path to IDNer2k folder.",
    )
    p.add_argument(
        "--output_dir",
        default=str(ROOT / "outputs" / "gliner"),
        help="Output directory.",
    )
    p.add_argument(
        "--model_name",
        default="urchade/gliner_medium-v2.1",
        help="GLiNER model name (Hugging Face).",
    )
    p.add_argument(
        "--checkpoint",
        default=None,
        help="Path to fine-tuned GLiNER checkpoint (overrides --model_name if set).",
    )
    p.add_argument(
        "--split",
        default="test",
        choices=["train", "dev", "test"],
        help="Which split to evaluate.",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="GLiNER prediction threshold (default 0.3 for zero-shot Indonesian).",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Print raw GLiNER output for first 2 sentences to diagnose F1=0.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (from env RANDOM_SEED if not set).",
    )
    p.add_argument(
        "--max_sentences",
        type=int,
        default=None,
        help="Max sentences to predict (for quick test). Default: all.",
    )
    p.add_argument(
        "--gpu",
        type=str,
        default=None,
        help="GPU id (e.g. 0). Sets CUDA_VISIBLE_DEVICES. Default: use env or 0.",
    )
    args = p.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    seed = args.seed
    if seed is None:
        seed = int(os.environ.get("RANDOM_SEED", "42"))
    try:
        import random
        import numpy as np

        random.seed(seed)
        np.random.seed(seed)
    except Exception:
        pass

    data_root = Path(args.data_root).resolve()
    test_file = data_root / f"{args.split}_bio.txt"
    if not test_file.exists():
        print(f"Data file not found: {test_file}", file=sys.stderr)
        sys.exit(1)

    MAP_LOCATION = "cpu"
    DEVICE = None
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        MAP_LOCATION = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device: {DEVICE}")
    except Exception:
        pass

    try:
        from gliner import GLiNER
    except ImportError:
        print("GLiNER not installed. Run: pip install gliner", file=sys.stderr)
        sys.exit(1)

    model_path = args.checkpoint if args.checkpoint else args.model_name
    print(f"Loading GLiNER model: {model_path} (map_location={MAP_LOCATION})")
    try:
        model = GLiNER.from_pretrained(model_path, map_location=MAP_LOCATION)
    except FileNotFoundError as e:
        if "config" in str(e).lower() and args.checkpoint:
            cp_dir = Path(args.checkpoint).resolve()
            if cp_dir.is_dir():
                checkpoints = sorted(
                    cp_dir.glob("checkpoint-*"),
                    key=lambda p: (
                        int(p.name.split("-")[-1])
                        if p.name.split("-")[-1].isdigit()
                        else 0
                    ),
                )
                for sub in reversed(checkpoints):
                    if (sub / "gliner_config.json").exists():
                        print(f"  Using checkpoint: {sub}")
                        model = GLiNER.from_pretrained(
                            str(sub), map_location=MAP_LOCATION
                        )
                        break
                else:
                    raise
            else:
                raise
        else:
            raise
    if DEVICE is not None and DEVICE.type == "cuda":
        if hasattr(model, "to"):
            model = model.to(DEVICE)
        elif hasattr(model, "model") and hasattr(model.model, "to"):
            model.model.to(DEVICE)
        print("Model moved to GPU for inference.")

    print(f"Loading {args.split} set: {test_file}")
    sentences = load_bio_file(test_file)
    if args.max_sentences is not None:
        sentences = sentences[: args.max_sentences]
        print(f"  Using first {len(sentences)} sentences (--max_sentences)")
    else:
        print(f"  Sentences: {len(sentences)}")
    if MAP_LOCATION == "cuda":
        print("  (GLiNER runs 1 inference per sentence; using GPU for speed.)")
    else:
        print(
            "  (GLiNER runs 1 inference per sentence; 509 sentences can take several minutes on CPU.)"
        )

    labels = GLINER_INPUT_LABELS
    print(
        f"Predicting with labels: {labels} (output mapped to PER/ORG/LOC; char spans -> token BIO)"
    )
    pred_tags = predict_and_convert_to_bio(
        model,
        sentences,
        labels,
        threshold=args.threshold,
        debug=args.debug,
    )

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_file = out_dir / f"predictions_{args.split}_bio.txt"
    with open(pred_file, "w", encoding="utf-8") as f:
        for (tokens, _), tags in zip(sentences, pred_tags):
            for t, tag in zip(tokens, tags):
                f.write(f"{t}\t{tag}\n")
            f.write("\n")
    print(f"Predictions saved: {pred_file}")

    report = evaluate_bio_predictions(
        sentences,
        pred_tags,
        output_dir=out_dir,
        run_name=f"gliner_{args.split}",
    )
    w = report.get("weighted avg", {})
    f1 = w.get("f1-score", 0)
    prec = w.get("precision", 0)
    rec = w.get("recall", 0)
    print(f"Weighted F1:  {f1:.4f}")
    print(f"Precision:    {prec:.4f}")
    print(f"Recall:       {rec:.4f}")

    summary = {
        "model": "GLiNER",
        "model_name": args.model_name,
        "checkpoint": args.checkpoint,
        "split": args.split,
        "seed": seed,
        "weighted_avg": {"f1-score": f1, "precision": prec, "recall": rec},
    }
    summary_path = out_dir / f"summary_{args.split}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
