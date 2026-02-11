# Detailed Setup Guide

## 1. Environment Setup

### Option A: Using Conda (Recommended)
```bash
conda create -n indonesian-ner python=3.8
conda activate indonesian-ner
pip install -r requirements.txt
```

### Option B: Using venv
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. GPU Setup

### Check GPU availability
```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

### Set specific GPU
```bash
export CUDA_VISIBLE_DEVICES=0  # Use GPU 0
```

## 3. Data Preparation

### Directory Structure
```
data/
├── idner2k/
│   ├── train_bio.txt
│   ├── dev_bio.txt
│   └── test_bio.txt
├── nerugm/
│   ├── train_bio.txt
│   ├── dev_bio.txt
│   └── test_bio.txt
└── nerui/
    ├── train_bio.txt
    ├── dev_bio.txt
    └── test_bio.txt
```

### Data Format
Each file should contain:
```
token1<TAB>tag1
token2<TAB>tag2

token1<TAB>tag1
```

## 4. Model Training

Run from the **reproducibility_package** directory (or project root if the package is merged into the main repo). The package includes 8 models (paper Table 2): BiLSTM-CRF, BiLSTM+W2V, BiLSTM+W2V+CNN, IndoBERT-CRF, IndoBERT+BiLSTM, mBERT+BiLSTM, XLM-RoBERTa+BiLSTM, and TAMSA (`train_attention_fusion.py`).

To run all experiments at once: `python run_all_experiments_and_evaluations.py`  
To run step-by-step with progress saving: `python run_experiments_step_by_step.py`

### Basic Training
```bash
python code/train_bilstm.py
# Or TAMSA: python code/train_attention_fusion.py
# Or IndoBERT-CRF: python code/train_indobert.py
```

### Custom Hyperparameters
Edit the hyperparameter dictionary in the training script:
```python
HP = {
    "embedding_dim": 100,
    "lstm_hidden_dim": 256,
    "learning_rate": 0.001,
    "epochs": 50,
    "batch_size": 32
}
```

## 5. Monitoring Training

Training progress is saved to:
- `outputs/experiment_*/training_history.csv`
- Best model: `outputs/experiment_*/*-best.pt`

## 6. Common Issues

### ImportError for transformers
```bash
pip install transformers==4.35.2
```

### Flair embeddings download fails
```bash
# Manual download may be needed
# Check Flair documentation
```

### Memory issues with large models
- Reduce batch_size
- Use gradient accumulation
- Enable mixed precision training
