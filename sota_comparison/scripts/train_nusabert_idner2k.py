#!/usr/bin/env python3
"""
Fine-tune NusaBERT-NER on IDNer2k train (and optional dev for validation).
Uses same splits and entity types (PER, ORG, LOC, BIO) as TAMSA for fair comparison.
Requires: transformers, torch, datasets (optional; we use simple list of examples).
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

LABELS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]


def align_labels_to_subwords(tokens, tags, word_ids, label2id):
    """Map word-level BIO tags to subword indices; use -100 for special/continuation."""
    aligned = []
    prev_word_id = None
    for i, w_id in enumerate(word_ids):
        if w_id is None:
            aligned.append(-100)
        elif w_id == prev_word_id:
            aligned.append(-100)
        else:
            if w_id < len(tags):
                tag = tags[w_id]
                aligned.append(label2id.get(tag, label2id["O"]))
            else:
                aligned.append(-100)
            prev_word_id = w_id
    return aligned


def main():
    p = argparse.ArgumentParser(description="Fine-tune NusaBERT-NER on IDNer2k.")
    p.add_argument(
        "--data_root",
        default=str(ROOT / ".." / "data" / "idner2k"),
        help="Path to IDNer2k folder (train_bio.txt, dev_bio.txt).",
    )
    p.add_argument(
        "--output_dir",
        default=str(ROOT / "outputs" / "nusabert_finetuned"),
        help="Directory to save fine-tuned model.",
    )
    p.add_argument(
        "--model_name",
        default="cahya/NusaBert-ner",
        help="Base HuggingFace model to fine-tune.",
    )
    p.add_argument(
        "--num_epochs",
        type=int,
        default=5,
        help="Number of training epochs (default 5).",
    )
    p.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate (default 5e-5).",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Per-device train batch size (default 8).",
    )
    p.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Max token length (default 128).",
    )
    p.add_argument(
        "--eval_steps",
        type=int,
        default=200,
        help="Evaluate and save every N steps (default 200).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    p.add_argument(
        "--gpu",
        type=str,
        default=None,
        help="GPU id (e.g. 0). Sets CUDA_VISIBLE_DEVICES.",
    )
    args = p.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    try:
        import random
        import numpy as np
        import torch
        from transformers import (
            AutoTokenizer,
            AutoModelForTokenClassification,
            TrainingArguments,
            Trainer,
            DataCollatorForTokenClassification,
        )
    except ImportError as e:
        print(
            "transformers and torch required. pip install transformers torch",
            file=sys.stderr,
        )
        raise SystemExit(1) from e

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    data_root = Path(args.data_root).resolve()
    train_file = data_root / "train_bio.txt"
    dev_file = data_root / "dev_bio.txt"
    if not train_file.exists():
        print(f"Train file not found: {train_file}", file=sys.stderr)
        sys.exit(1)

    print("Loading tokenizer and model:", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForTokenClassification.from_pretrained(args.model_name)

    id2label_raw = getattr(model.config, "id2label", None) or {}
    id2label = {}
    for k, v in id2label_raw.items():
        id2label[int(k)] = str(v)
    label2id = {v: k for k, v in id2label.items()}
    for lb in LABELS:
        if lb not in label2id:
            label2id[lb] = label2id.get("O", 0)
    print("Label2id (from model):", label2id)

    print("Loading train set:", train_file)
    train_sentences = load_bio_file(train_file)
    print(f"  Train sentences: {len(train_sentences)}")

    def tokenize_and_align(sentences):
        out = []
        for tokens, tags in sentences:
            enc = tokenizer(
                tokens,
                is_split_into_words=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors=None,
                padding=False,
            )
            input_ids = enc["input_ids"]
            if isinstance(input_ids[0], list):
                input_ids = input_ids[0]
            attention_mask = enc.get("attention_mask", [1] * len(input_ids))
            if isinstance(attention_mask[0], list):
                attention_mask = attention_mask[0]
            word_ids = (
                enc.word_ids(0)
                if hasattr(enc, "word_ids") and enc.word_ids(0) is not None
                else None
            )
            if word_ids is None:
                labels = [-100] * len(input_ids)
                for wi, tag in enumerate(tags):
                    if wi + 1 < len(input_ids) - 1:
                        labels[wi + 1] = label2id.get(tag, label2id.get("O", 0))
            else:
                labels = align_labels_to_subwords(tokens, tags, word_ids, label2id)
            out.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }
            )
        return out

    train_data = tokenize_and_align(train_sentences)
    eval_data = None
    if dev_file.exists():
        print("Loading dev set:", dev_file)
        dev_sentences = load_bio_file(dev_file)
        print(f"  Dev sentences: {len(dev_sentences)}")
        eval_data = tokenize_and_align(dev_sentences)
    else:
        print("No dev file; evaluation on train subset.")

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        eval_strategy="steps" if eval_data else "no",
        eval_steps=args.eval_steps if eval_data else None,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=2,
        logging_steps=min(50, args.eval_steps),
        load_best_model_at_end=bool(eval_data),
        metric_for_best_model="f1" if eval_data else None,
        greater_is_better=True,
        seed=args.seed,
    )

    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        padding=True,
        max_length=args.max_length,
        label_pad_token_id=-100,
    )

    from datasets import Dataset

    train_ds = Dataset.from_list(train_data)
    eval_ds = Dataset.from_list(eval_data) if eval_data else None

    def compute_metrics(eval_pred):
        from seqeval.metrics import f1_score, precision_score, recall_score

        preds, labels = eval_pred
        preds = np.argmax(preds, axis=2)
        true_labels = []
        pred_labels = []
        for i in range(labels.shape[0]):
            true_row = []
            pred_row = []
            for j in range(labels.shape[1]):
                if labels[i, j] != -100:
                    lid = int(labels[i, j])
                    pid = int(preds[i, j])
                    true_row.append(id2label.get(lid, "O"))
                    pred_row.append(id2label.get(pid, "O"))
            if true_row:
                true_labels.append(true_row)
                pred_labels.append(pred_row)
        if not true_labels:
            return {"f1": 0.0, "precision": 0.0, "recall": 0.0}
        return {
            "f1": f1_score(true_labels, pred_labels),
            "precision": precision_score(true_labels, pred_labels),
            "recall": recall_score(true_labels, pred_labels),
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics if eval_ds else None,
    )

    print("Starting fine-tuning...")
    print(
        f"  epochs={args.num_epochs} lr={args.learning_rate} batch_size={args.batch_size} max_length={args.max_length}"
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    model.config.save_pretrained(str(out_dir))
    print(f"Model saved to: {out_dir}")

    config_summary = {
        "model_name": args.model_name,
        "data_root": str(data_root),
        "train_samples": len(train_sentences),
        "eval_samples": len(eval_data) if eval_data else 0,
        "num_epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "seed": args.seed,
    }
    with open(out_dir / "train_config.json", "w", encoding="utf-8") as f:
        json.dump(config_summary, f, indent=2)
    print("Config saved: train_config.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
