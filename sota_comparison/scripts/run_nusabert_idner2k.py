#!/usr/bin/env python3
"""
Run NusaBERT-NER on IDNer2k: load test set, tokenize, predict, align to word-level BIO,
evaluate with standard evaluator. Uses same dataset paths and metrics for fair comparison.
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.bio_io import load_bio_file
from utils.eval_ner import evaluate_bio_predictions


def main():
    p = argparse.ArgumentParser(description="Run NusaBERT-NER on IDNer2k and evaluate.")
    p.add_argument(
        "--data_root",
        default=str(ROOT / ".." / "data" / "idner2k"),
        help="Path to IDNer2k folder.",
    )
    p.add_argument(
        "--output_dir",
        default=str(ROOT / "outputs" / "nusabert"),
        help="Output directory.",
    )
    p.add_argument(
        "--model_name", default="cahya/NusaBert-ner", help="HuggingFace model name."
    )
    p.add_argument(
        "--checkpoint",
        default=None,
        help="Path to fine-tuned checkpoint (overrides --model_name if set).",
    )
    p.add_argument(
        "--split",
        default="test",
        choices=["train", "dev", "test"],
        help="Which split to evaluate.",
    )
    p.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for inference."
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (from env RANDOM_SEED if not set).",
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

    seed = args.seed or int(os.environ.get("RANDOM_SEED", "42"))
    try:
        import random
        import numpy as np
        import torch

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

    data_root = Path(args.data_root).resolve()
    test_file = data_root / f"{args.split}_bio.txt"
    if not test_file.exists():
        print(f"Data file not found: {test_file}", file=sys.stderr)
        sys.exit(1)

    try:
        from transformers import AutoTokenizer, AutoModelForTokenClassification
        import torch
    except ImportError:
        print(
            "transformers and torch required. pip install transformers torch",
            file=sys.stderr,
        )
        sys.exit(1)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model_path = Path(args.checkpoint).resolve() if args.checkpoint else args.model_name
    print(f"Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    id2label = model.config.id2label
    model = model.to(device)
    model.eval()

    print(f"Loading {args.split} set: {test_file}")
    sentences = load_bio_file(test_file)
    print(f"  Sentences: {len(sentences)}")

    pred_tags = []
    n_sent = len(sentences)
    for i, (tokens, _) in enumerate(sentences):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  NusaBERT progress: {i + 1}/{n_sent} sentences", flush=True)
        encoding = tokenizer(
            tokens,
            is_split_into_words=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        word_ids = (
            encoding.word_ids(0)
            if hasattr(encoding, "word_ids") and encoding.word_ids(0) is not None
            else None
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits
        pred_id_seq = logits[0].argmax(dim=-1).cpu().tolist()

        n_words = len(tokens)
        word_labels = ["O"] * n_words
        if word_ids is not None:
            prev_word = None
            for idx, w_id in enumerate(word_ids):
                if w_id is None:
                    continue
                if prev_word is not None and w_id == prev_word:
                    continue
                label_id = pred_id_seq[idx]
                label = id2label.get(str(label_id), id2label.get(label_id, "O"))
                if w_id < n_words:
                    word_labels[w_id] = label
                prev_word = w_id
        else:
            for i in range(min(n_words, len(pred_id_seq))):
                lid = pred_id_seq[i]
                word_labels[i] = id2label.get(str(lid), id2label.get(lid, "O"))
        pred_tags.append(word_labels)

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
        run_name=f"nusabert_{args.split}",
    )
    w = report.get("weighted avg", {})
    f1 = w.get("f1-score", 0)
    prec = w.get("precision", 0)
    rec = w.get("recall", 0)
    print(f"Weighted F1:  {f1:.4f}")
    print(f"Precision:    {prec:.4f}")
    print(f"Recall:       {rec:.4f}")

    summary = {
        "model": "NusaBERT-NER",
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
