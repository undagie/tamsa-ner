# TAMSA-NER: Reproducibility Package

## Overview

This package contains all code, configurations, and documentation needed to reproduce the experiments from the paper **"Token-Aware Multi-Source Attention for Indonesian Named Entity Recognition"** (TAMSA).

Reproducibility code for the experiments is intended for public release (e.g. https://github.com/undagie/tamsa-ner).

## Package Contents

### 1. Code (`code/`)

- **Training scripts (8 models, paper Table 2):**
  - `train_bilstm.py` — BiLSTM-CRF
  - `train_bilstm_w2v.py` — BiLSTM+Word2Vec
  - `train_bilstm_w2v_cnn.py` — BiLSTM+Word2Vec+Char-CNN
  - `train_indobert.py` — IndoBERT-CRF
  - `train_indobert_bilstm.py` — IndoBERT+BiLSTM
  - `train_mbert_bilstm.py` — mBERT+BiLSTM
  - `train_xlm_roberta_bilstm.py` — XLM-RoBERTa+BiLSTM
  - `train_attention_fusion.py` — TAMSA (proposed)
- **Evaluation scripts** for cross-dataset testing (NER-UI, NER-UGM), including `evaluate_indobert_only_on_nerui.py` for IndoBERT-CRF on NER-UI
- **Analysis scripts:** ablation, efficiency, error analysis, multiple runs (5 seeds), statistical tests, per-entity analysis, learning curves, attention visualization
- **Utility scripts:** `run_all_experiments_and_evaluations.py`, `predict_sentence.py`

### 2. Data (`data/`)

- Sample or full data for IDNer2k, NER-UGM, and NER-UI (see `data/README.md`)
- BIO tagging format; see paper for dataset references

### 3. Configurations (`configs/`)

- Hyperparameter JSON files for all models (Table 1 in paper)
- Vocabulary summaries

### 4. Results (`results/`)

- Training histories, classification reports, cross-dataset and analysis outputs (optional; regenerate by running scripts)

### 5. Scripts (`scripts/`)

- `run_all_experiments.py` — Train all 8 models, run cross-dataset evaluations, then run analyses
- `quick_evaluate.py` — Evaluate a single model

### 6. Documentation (`docs/`)

- Setup guide, result interpretation, troubleshooting

## System Requirements

- **Hardware:** NVIDIA GPU with at least 8GB memory (e.g. A100 20GB as in paper); 16GB+ RAM
- **Software:** Python 3.8+, PyTorch, Transformers; see `requirements.txt`
- **OS:** Linux or Windows

## Quick Start

1. **Setup**

   ```bash
   python -m venv venv
   # Windows: venv\Scripts\activate
   # Linux/Mac: source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Data**  
   Place IDNer2k, NER-UI, and NER-UGM data under `data/idner2k/`, `data/nerui/`, `data/nerugm/` with `train_bio.txt`, `dev_bio.txt`, `test_bio.txt` (see `data/README.md`).

3. **Run one model**

   ```bash
   python code/train_bilstm.py
   # or: python code/train_attention_fusion.py  # TAMSA
   ```

4. **Run all experiments (training + cross-dataset eval + analyses)**
   ```bash
   python scripts/run_all_experiments.py
   ```

## Reproducing Paper Results

- **Table 2 & 3 (main results, per-seed F1):** Train all 8 models, then run `python code/multiple_runs_analysis.py` with seeds 42, 123, 456, 789, 1011 (five runs).
- **Table 4 (statistical significance):** After multiple runs, run `python code/statistical_tests_comprehensive.py`.
- **Table 5 (per-entity):** `python code/per_entity_analysis.py`
- **Table 6 (ablation):** `python code/ablation_study.py`
- **Table 7 (cross-dataset):** Train on IDNer2k, run all `evaluate_*_on_nerui.py` (and NER-UGM if available), then `python code/cross_dataset_evaluation.py`.
- **Table 8 (error analysis):** `python code/linguistic_error_analysis.py`
- **Table 9 (efficiency):** `python code/efficiency_analysis.py`
- **Figure 2 (learning curves):** `python code/epoch_analysis.py` or `python code/generate_paper_figures.py`
- **Figure 3 (attention):** `python code/visualize_attention.py` or `python code/generate_paper_attention_visualization.py`

## Expected Outputs

- Checkpoints: `outputs/experiment_*/` (e.g. `experiment_indobert/`, `experiment_attention_fusion/`)
- Cross-dataset: `outputs/evaluation_nerui_*/`, `outputs/cross_dataset_evaluation/`
- Multiple runs: `outputs/multiple_runs/`
- Figures and reports in the corresponding output directories

## Citation

If you use this code, please cite:

```bibtex
@article{tamsa-ner,
  title={Token-Aware Multi-Source Attention for Indonesian Named Entity Recognition},
  author={Rosadi, Muhammad Edya and Andono, Pulung Nurtantio and Muljono and Fanani, Ahmad Zainul and Marjuni, Aris},
  journal={[International Journal of Intelligent Engineering and Systems]},
  year={[2026]}
}
```

## License

This code is released under the MIT License.
