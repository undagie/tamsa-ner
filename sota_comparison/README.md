# SOTA Comparison (Paper Table 3)

Part of the [TAMSA-NER Reproducibility Package](../README.md). This folder reproduces **Table 3**: quantitative comparison of 2024–2025 NER SOTA methods (GLiNER, NusaBERT-NER) on **the same dataset, splits, and metrics** as the main experiments (IDNer2k, entity-level weighted F1, seqeval IOB2).

## Quick start

From the **reproducibility package root**:

```bash
pip install -r sota_comparison/requirements_sota.txt
python sota_comparison/run_all_sota.py
```

From inside `sota_comparison/`:

```bash
cd sota_comparison
pip install -r requirements_sota.txt
python run_all_sota.py
```

This run: checks dependencies, self-tests the evaluator (gold=pred -> F1 1.0), **GLiNER zero-shot**, **GLiNER fine-tune** (train + eval), **NusaBERT-NER** (inference-only and fine-tuned), aggregates results, and prints the article table. For GLiNER fine-tuning use `transformers<5`.

### Pipeline check without GLiNER

To verify the span -> BIO conversion and evaluator without installing GLiNER:

```bash
python sota_comparison/scripts/test_gliner_conversion.py
```

If F1 ≥ 0.9, the pipeline is correct. Quick test with GLiNER (5 sentences):  
`python sota_comparison/scripts/run_gliner_idner2k.py --data_root data/idner2k --split test --output_dir sota_comparison/outputs/gliner_test --max_sentences 5 --threshold 0.3`

**Skip training** when a checkpoint exists: `GLINER_SKIP_TRAIN=1` or `NUSA_SKIP_TRAIN=1`.

## Structure

```
sota_comparison/
├── configs/              idner2k_protocol.json
├── data/idner2k/         optional (can use ../data/idner2k from package root)
├── docs/
├── scripts/              evaluate_idner2k_standard.py, run_*_idner2k.py, train_*_idner2k.py, ...
├── utils/                bio_io.py, eval_ner.py, convert_gliner.py
├── outputs/              per-model results (created by run_all_sota.py)
├── requirements_sota.txt, requirements_gliner_training.txt
├── run_all_sota.py       single command for full Table 3 run
└── README.md
```

## Data

- **IDNer2k**: same as main package — `data/idner2k/` from repo root (or `../data/idner2k` when run from `sota_comparison/`).
- Files: `train_bio.txt`, `dev_bio.txt`, `test_bio.txt` (format: `token<TAB>tag`, blank line = sentence boundary).
- Splits: train 1464, dev 367, test 509 sentences (same as TAMSA and baselines).

## Metrics

- **Primary:** entity-level **weighted F1** (seqeval, IOB2 scheme).
- Also: Precision and Recall (weighted avg).
- All models are evaluated with the same script: `scripts/evaluate_idner2k_standard.py`.

## How to run

### 1. Install dependencies

From **package root**:

```bash
pip install -r sota_comparison/requirements_sota.txt
```

`seqeval` is required for the standard evaluator. For evaluator-only use: `pip install seqeval`.

### 2. Standard evaluation (gold + prediction BIO)

```bash
cd sota_comparison
python scripts/evaluate_idner2k_standard.py \
  --gold ../data/idner2k/test_bio.txt \
  --pred outputs/gliner/predictions_test_bio.txt \
  --output_dir outputs/gliner
```

### 3. GLiNER (zero-shot on test set)

```bash
cd sota_comparison
python scripts/run_gliner_idner2k.py --split test --output_dir outputs/gliner
```

Environment variable `RANDOM_SEED` is used if set (for reproducibility).

### 4. NusaBERT-NER (inference on test set)

```bash
cd sota_comparison
python scripts/run_nusabert_idner2k.py --split test --output_dir outputs/nusabert
```

### 5. Run all SOTA

```bash
cd sota_comparison
python run_all_sota.py
```

## Seeds and multiple runs

- Protocol: 5 runs with seeds **42, 123, 456, 789, 1011** (see `configs/idner2k_protocol.json`).
- For a single run, set env: `RANDOM_SEED=42` or use `--seed 42`.
- For 5 runs, run the script per seed and aggregate; report mean ± std in the paper.

## Documented mismatches (paper)

- **GLiNER:** span-based; predictions are converted to token-level BIO for evaluation with the same script.
- **NusaBERT:** token-level; subword–word alignment via `word_ids`; evaluated with the same script.

## Including results in the article

- After running SOTA, use `python scripts/aggregate_results_for_article.py` to print the table from summary JSON files.

## Runtime (expected)

- **GLiNER:** One inference per sentence (509 sentences = 509 forwards). On CPU this can take **several to tens of minutes**. Faster on GPU. Script prints progress every 50 sentences.
- **NusaBERT:** Also per sentence; script prints progress every 50 sentences.
- **Quick test:** Run GLiNER on the first 100 sentences only:  
  `python scripts/run_gliner_idner2k.py --split test --max_sentences 100`
- **GPU:** Scripts use the same settings as `train_*`: `CUDA_VISIBLE_DEVICES` (default `0`), `torch.cuda`, cuDNN benchmark. **GLiNER:** model loaded with `map_location="cuda"` and moved to GPU when available (much faster inference). **NusaBERT-NER:** model and input are `.to(device)`. Use another GPU: `--gpu 1` or `export CUDA_VISIBLE_DEVICES=1`.

## Verification

Self-test: gold = pred must yield F1 = 1.0:

```bash
cd sota_comparison
python scripts/evaluate_idner2k_standard.py --gold ../data/idner2k/test_bio.txt --pred ../data/idner2k/test_bio.txt --output_dir outputs/self_test
```

## References

- GLiNER: https://github.com/urchade/GLiNER
- NusaBERT-NER: https://huggingface.co/cahya/NusaBert-ner
