#!/usr/bin/env python3
"""
Run all SOTA experiments: check dependencies, self-test evaluator, GLiNER zero-shot,
GLiNER fine-tune (train + eval), NusaBERT-NER, aggregation, and article table.

Order: [0] deps, [1] self-test, [2] GLiNER zero-shot, [3] GLiNER fine-tune, [4] NusaBERT-NER
inference-only, [5] NusaBERT-NER fine-tune, [6] aggregate, [7] print table.

Env options:
  GLINER_SKIP_TRAIN=1  Skip GLiNER training if checkpoint exists (eval fine-tuned only).
  NUSA_SKIP_TRAIN=1    Skip NusaBERT-NER training if checkpoint exists (eval fine-tuned only).
  GLINER_DEBUG=1       Add --debug when running GLiNER zero-shot.

Run from project root: python sota_comparison/run_all_sota.py
Or from sota_comparison: python run_all_sota.py
"""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "configs" / "idner2k_protocol.json"
OUTPUTS = ROOT / "outputs"
SCRIPTS = ROOT / "scripts"


def _get_data_root():
    local_data = ROOT / "data" / "idner2k"
    if local_data.exists() and (local_data / "test_bio.txt").exists():
        return local_data.resolve()
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, encoding="utf-8") as f:
                cfg = json.load(f)
            p = cfg.get("paths", {}).get("data_root", "")
            if p:
                return (ROOT / p).resolve()
        except Exception:
            pass
    return (ROOT / ".." / "data" / "idner2k").resolve()


DATA_ROOT = _get_data_root()


def check_dependencies():
    """Check dependencies; return True if at least seqeval is available."""
    missing = []
    try:
        import seqeval  # noqa: F401
    except ImportError:
        missing.append("seqeval (pip install seqeval)")
    if missing:
        print("Missing required dependencies:", ", ".join(missing), file=sys.stderr)
        print(
            "Run: pip install -r sota_comparison/requirements_sota.txt", file=sys.stderr
        )
        return False
    optional = []
    try:
        import gliner  # noqa: F401
    except ImportError:
        optional.append("gliner")
    try:
        import transformers  # noqa: F401
    except ImportError:
        optional.append("transformers")
    if optional:
        print("Optional (for GLiNER/NusaBERT): install", ", ".join(optional))
    return True


def run_self_test():
    """Self-test evaluator: gold = pred -> F1 1.0."""
    test_file = DATA_ROOT / "test_bio.txt"
    if not test_file.exists():
        return False
    out_dir = OUTPUTS / "self_test"
    out_dir.mkdir(parents=True, exist_ok=True)
    ret = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "evaluate_idner2k_standard.py"),
            "--gold",
            str(test_file),
            "--pred",
            str(test_file),
            "--output_dir",
            str(out_dir),
        ],
        cwd=str(ROOT),
    )
    return ret.returncode == 0


def run_script(script_name: str, *args: str) -> bool:
    """Run script in sota_comparison/scripts with given args."""
    cmd = [sys.executable, str(SCRIPTS / script_name)] + list(args)
    ret = subprocess.run(cmd, cwd=str(ROOT))
    return ret.returncode == 0


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
ARTICLE_FASTTEXT = (
    "BiLSTM-CRF + FastText [36]",
    90.85,
    92.23,
    89.52,
    "Reported in [36]; same dataset (re-annotated IDNer2k); Table 4/5.",
)
ARTICLE_NOTES = {
    "GLiNER": "Zero-shot; F1 negligible without Indonesian fine-tuning; not comparable; see text.",
    "GLiNER (fine-tuned)": "Fine-tuned on IDNer2k train; same protocol.",
    "NusaBERT-NER": "Inference-only on IDNer2k test; token-level; same evaluation protocol.",
    "NusaBERT-NER (fine-tuned)": "Fine-tuned on IDNer2k train; same protocol.",
}


def _read_summary(path: Path):
    """Read summary_test.json; return (f1_pct, prec_pct, rec_pct) or (None, None, None)."""
    if not path.exists():
        return None, None, None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None, None, None
    w = data.get("weighted_avg", {})
    if not w:
        return None, None, None
    f1 = w.get("f1-score")
    if f1 is None:
        return None, None, None
    return (
        f1 * 100,
        (w.get("precision") or 0) * 100,
        (w.get("recall") or 0) * 100,
    )


def aggregate_and_print_table():
    """Read summary JSON, merge with TAMSA/IndoBERT/[36], print table for article."""
    rows = []

    f1, prec, rec = _read_summary(OUTPUTS / "gliner" / "summary_test.json")
    if f1 is not None and f1 < 1.0:
        rows.append(("GLiNER [31]", None, None, None, ARTICLE_NOTES["GLiNER"]))
    elif f1 is not None:
        rows.append(("GLiNER [31]", f1, prec, rec, ARTICLE_NOTES["GLiNER"]))
    else:
        rows.append(("GLiNER [31]", None, None, None, "No summary (run not executed)"))

    f1, prec, rec = _read_summary(OUTPUTS / "nusabert" / "summary_test.json")
    if f1 is not None:
        rows.append(("NusaBERT-NER [34]", f1, prec, rec, ARTICLE_NOTES["NusaBERT-NER"]))
    else:
        rows.append(
            ("NusaBERT-NER [34]", None, None, None, "No summary (run not executed)")
        )

    f1, prec, rec = _read_summary(
        OUTPUTS / "nusabert_finetuned_eval" / "summary_test.json"
    )
    if f1 is not None:
        rows.append(
            (
                "NusaBERT-NER [34] (fine-tuned)",
                f1,
                prec,
                rec,
                ARTICLE_NOTES["NusaBERT-NER (fine-tuned)"],
            )
        )
    else:
        rows.append(
            (
                "NusaBERT-NER [34] (fine-tuned)",
                None,
                None,
                None,
                "No summary (run train_nusabert_idner2k.py then run with --checkpoint)",
            )
        )

    f1, prec, rec = _read_summary(
        OUTPUTS / "gliner_finetuned_eval" / "summary_test.json"
    )
    if f1 is not None:
        rows.append(
            (
                "GLiNER [31] (fine-tuned)",
                f1,
                prec,
                rec,
                ARTICLE_NOTES["GLiNER (fine-tuned)"],
            )
        )
    else:
        rows.append(
            (
                "GLiNER [31] (fine-tuned)",
                None,
                None,
                None,
                "No summary (run train_gliner_idner2k.py then run with --checkpoint)",
            )
        )

    rows.append(ARTICLE_FASTTEXT)
    rows.append(ARTICLE_INDOBERT)
    rows.append(ARTICLE_TAMSA)

    def f1_sort_key(r):
        _, f1, _, _, _ = r
        return (f1 is None, f1 if f1 is not None else 0.0)

    rows.sort(key=f1_sort_key)

    print()
    print("=" * 80)
    print(
        "ARTICLE TABLE (weighted avg, same dataset/splits/metrics; seqeval IOB2)"
    )
    print("=" * 80)
    print()
    print("| Method                     | F1 (%) | Precision (%) | Recall (%) | Notes")
    print(
        "|----------------------------|--------|----------------|------------|----------------------------------------------------------------------|"
    )
    for name, f1, prec, rec, notes in rows:
        f1_s = f"{f1:.2f}" if f1 is not None else "—"
        prec_s = f"{prec:.2f}" if prec is not None else "—"
        rec_s = f"{rec:.2f}" if rec is not None else "—"
        name_pad = (name[:24] + "..") if len(name) > 26 else name.ljust(26)
        print(f"| {name_pad} | {f1_s:>6} | {prec_s:>14} | {rec_s:>10} | {notes}")
    print()
    print(
        "Source: TAMSA, IndoBERT-CRF, BiLSTM-CRF+FastText [36] from manuscript/literature; "
        "GLiNER & NusaBERT-NER from SOTA runs. docs/ARTICLE_TABLE.md"
    )
    print()


def main():
    print("SOTA Comparison (single-command run)")
    print("ROOT:", ROOT)
    print("DATA_ROOT:", DATA_ROOT)
    print()

    OUTPUTS.mkdir(parents=True, exist_ok=True)

    if not (DATA_ROOT / "test_bio.txt").exists():
        print(
            "ERROR: IDNer2k test set not found:",
            DATA_ROOT / "test_bio.txt",
            file=sys.stderr,
        )
        sys.exit(1)

    print("[0/7] Checking dependencies...")
    if not check_dependencies():
        sys.exit(1)
    print("  OK (seqeval present)")
    print()

    print("[1/7] Self-test evaluator (gold=pred -> F1 1.0)...")
    if not run_self_test():
        print("  WARNING: self-test failed (evaluator or data path)", file=sys.stderr)
    else:
        print("  OK")
    print()

    print("[2/7] Running GLiNER zero-shot on IDNer2k test...")
    gliner_args = [
        "run_gliner_idner2k.py",
        "--data_root",
        str(DATA_ROOT),
        "--split",
        "test",
        "--output_dir",
        str(OUTPUTS / "gliner"),
        "--threshold",
        "0.3",
    ]
    if os.environ.get("GLINER_DEBUG"):
        gliner_args.append("--debug")
    run_script(*gliner_args)
    if (OUTPUTS / "gliner" / "summary_test.json").exists():
        print("  OK")
    else:
        print("  SKIP or FAIL (install gliner for full run)")
    print()

    gliner_ft_dir = OUTPUTS / "gliner_finetuned"
    gliner_ft_eval_dir = OUTPUTS / "gliner_finetuned_eval"
    skip_train = os.environ.get("GLINER_SKIP_TRAIN", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    if skip_train and (gliner_ft_dir / "gliner_config.json").exists():
        print(
            "[3/7] GLiNER fine-tuned: skip training (GLINER_SKIP_TRAIN=1, checkpoint exists)..."
        )
    else:
        print("[3/7] GLiNER fine-tune: training on IDNer2k train...")
        run_script(
            "train_gliner_idner2k.py",
            "--data_root",
            str(DATA_ROOT),
            "--output_dir",
            str(gliner_ft_dir),
            "--max_steps",
            "2000",
            "--seed",
            "42",
        )
        if not (gliner_ft_dir / "gliner_config.json").exists():
            print("  WARNING: training may have failed (no gliner_config.json)")
    print("  Evaluating GLiNER fine-tuned on test...")
    run_script(
        "run_gliner_idner2k.py",
        "--data_root",
        str(DATA_ROOT),
        "--split",
        "test",
        "--output_dir",
        str(gliner_ft_eval_dir),
        "--checkpoint",
        str(gliner_ft_dir),
        "--threshold",
        "0.3",
    )
    if (gliner_ft_eval_dir / "summary_test.json").exists():
        print("  OK")
    else:
        print("  SKIP or FAIL (install gliner[training], transformers<5 for training)")
    print()

    print("[4/7] Running NusaBERT-NER on IDNer2k test (inference-only)...")
    run_script(
        "run_nusabert_idner2k.py",
        "--data_root",
        str(DATA_ROOT),
        "--split",
        "test",
        "--output_dir",
        str(OUTPUTS / "nusabert"),
    )
    if (OUTPUTS / "nusabert" / "summary_test.json").exists():
        print("  OK")
    else:
        print("  SKIP or FAIL (install transformers, torch for full run)")
    print()

    nusabert_ft_dir = OUTPUTS / "nusabert_finetuned"
    nusabert_ft_eval_dir = OUTPUTS / "nusabert_finetuned_eval"
    skip_nusa_train = os.environ.get("NUSA_SKIP_TRAIN", "").strip() == "1"
    if skip_nusa_train and (nusabert_ft_dir / "config.json").exists():
        print(
            "[5/7] NusaBERT-NER fine-tuned: skip training (NUSA_SKIP_TRAIN=1, checkpoint exists)..."
        )
    else:
        print("[5/7] NusaBERT-NER fine-tune: training on IDNer2k train...")
        run_script(
            "train_nusabert_idner2k.py",
            "--data_root",
            str(DATA_ROOT),
            "--output_dir",
            str(nusabert_ft_dir),
        )
    print("  Evaluating NusaBERT-NER fine-tuned on test...")
    run_script(
        "run_nusabert_idner2k.py",
        "--data_root",
        str(DATA_ROOT),
        "--split",
        "test",
        "--output_dir",
        str(nusabert_ft_eval_dir),
        "--checkpoint",
        str(nusabert_ft_dir),
    )
    if (nusabert_ft_eval_dir / "summary_test.json").exists():
        print("  OK")
    else:
        print("  SKIP or FAIL (run train_nusabert_idner2k.py first)")
    print()

    print("[6/7] Aggregating results...")
    results = {}
    for name, subdir in [
        ("GLiNER", "gliner"),
        ("GLiNER_finetuned", "gliner_finetuned_eval"),
        ("NusaBERT", "nusabert"),
        ("NusaBERT_finetuned", "nusabert_finetuned_eval"),
    ]:
        p = OUTPUTS / subdir / "summary_test.json"
        if p.exists():
            try:
                with open(p, encoding="utf-8") as f:
                    results[name] = json.load(f)
            except Exception:
                results[name] = {"error": "read failed"}
        else:
            results[name] = {"error": "run not executed or failed"}

    summary_path = OUTPUTS / "sota_comparison_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("  Summary written:", summary_path)
    print()

    print("[7/7] Table for article:")
    aggregate_and_print_table()
    print(
        "Done. All SOTA runs (GLiNER zero-shot, GLiNER fine-tuned, NusaBERT-NER, NusaBERT-NER fine-tuned) and aggregation. See docs/ARTICLE_TABLE.md"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
