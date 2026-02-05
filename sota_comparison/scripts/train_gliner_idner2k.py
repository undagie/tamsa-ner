#!/usr/bin/env python3
"""
Fine-tune GLiNER on IDNer2k train (and optional dev for validation).
Uses same splits and entity types (PER, ORG, LOC) as TAMSA for a fair comparison.
Requires: pip install gliner[training]

If you see: Trainer.__init__() got an unexpected keyword argument 'tokenizer'
then GLiNER is incompatible with transformers>=5.0. Use: pip install "transformers>=4.20,<5.0"
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


def _check_transformers_version():
    """Ensure transformers version is compatible with GLiNER Trainer (avoid tokenizer argument error)."""
    try:
        import transformers

        ver = getattr(transformers, "__version__", "0")
        parts = ver.split(".")[:3]
        major = int(parts[0]) if parts and parts[0].isdigit() else 0
        if major >= 5:
            print(
                "Error: transformers>=5.0 is installed. GLiNER's Trainer raises:\n"
                "  TypeError: Trainer.__init__() got an unexpected keyword argument 'tokenizer'\n"
                'Fix: pip install "transformers>=4.20,<5.0" then re-run this script.',
                file=sys.stderr,
            )
            sys.exit(1)
    except Exception:
        pass


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.bio_io import load_bio_file
from utils.convert_gliner import bio_sentences_to_gliner_train_data


def main():
    p = argparse.ArgumentParser(description="Fine-tune GLiNER on IDNer2k.")
    p.add_argument(
        "--data_root",
        default=str(ROOT / ".." / "data" / "idner2k"),
        help="Path to IDNer2k folder (train_bio.txt, dev_bio.txt).",
    )
    p.add_argument(
        "--output_dir",
        default=str(ROOT / "outputs" / "gliner_finetuned"),
        help="Directory to save fine-tuned model.",
    )
    p.add_argument(
        "--model_name",
        default="urchade/gliner_medium-v2.1",
        help="Base GLiNER model to fine-tune.",
    )
    p.add_argument(
        "--max_steps",
        type=int,
        default=2000,
        help="Max training steps (default 2000).",
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

    _check_transformers_version()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    try:
        import random
        import numpy as np
        import torch

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    except Exception:
        pass

    data_root = Path(args.data_root).resolve()
    train_file = data_root / "train_bio.txt"
    dev_file = data_root / "dev_bio.txt"
    if not train_file.exists():
        print(f"Train file not found: {train_file}", file=sys.stderr)
        sys.exit(1)

    print("Loading train set:", train_file)
    train_sentences = load_bio_file(train_file)
    train_data = bio_sentences_to_gliner_train_data(train_sentences)
    train_data = [d for d in train_data if d.get("ner")]
    n_train_dropped = len(train_sentences) - len(train_data)
    if n_train_dropped:
        print(
            f"  Dropped {n_train_dropped} train samples with no entities (GLiNER requires at least one)."
        )
    print(f"  Train samples: {len(train_data)}")
    if len(train_data) == 0:
        print("No training samples left after filtering. Aborting.", file=sys.stderr)
        sys.exit(1)

    eval_data = None
    if dev_file.exists():
        print("Loading dev set:", dev_file)
        dev_sentences = load_bio_file(dev_file)
        eval_data = bio_sentences_to_gliner_train_data(dev_sentences)
        eval_data = [d for d in eval_data if d.get("ner")]
        n_dev_dropped = len(dev_sentences) - len(eval_data)
        if n_dev_dropped:
            print(f"  Dropped {n_dev_dropped} dev samples with no entities.")
        print(f"  Dev samples: {len(eval_data)}")
    else:
        print("No dev file; using train as eval (not ideal).")
        eval_data = train_data
    if not eval_data:
        eval_data = train_data
        print("  Using train as eval (dev had no samples with entities).")

    try:
        from gliner import GLiNER
    except ImportError:
        print(
            "GLiNER with training not installed. Run: pip install 'gliner[training]'",
            file=sys.stderr,
        )
        sys.exit(1)

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading base model: {args.model_name}")
    model = GLiNER.from_pretrained(args.model_name)

    print("Starting fine-tuning...")
    print(
        f"  max_steps={args.max_steps} lr={args.learning_rate} batch_size={args.batch_size} eval_steps={args.eval_steps}"
    )
    trainer = model.train_model(
        train_dataset=train_data,
        eval_dataset=eval_data,
        output_dir=str(out_dir),
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        save_steps=args.eval_steps,
        eval_steps=args.eval_steps,
        logging_steps=min(50, args.eval_steps),
        save_total_limit=2,
        seed=args.seed,
    )
    trainer.save_model(output_dir=str(out_dir))
    model.save_pretrained(out_dir)
    print(f"Model saved to: {out_dir}")

    config_summary = {
        "model_name": args.model_name,
        "data_root": str(data_root),
        "train_samples": len(train_data),
        "eval_samples": len(eval_data),
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "seed": args.seed,
    }
    with open(out_dir / "train_config.json", "w", encoding="utf-8") as f:
        json.dump(config_summary, f, indent=2)
    print("Config saved: train_config.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
