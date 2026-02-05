import json
import os
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from collections import Counter
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='torch')
warnings.filterwarnings('ignore', category=DeprecationWarning, module='torch')
from torchcrf import CRF
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix
from seqeval.metrics import classification_report as seqeval_classification_report
from seqeval.scheme import IOB2

TEST_FILE = Path('./data/nerui/test_bio.txt')

OUTPUT_DIR = Path('./outputs/evaluation_nerui_bilstm_w2v')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ORIGINAL_EXPERIMENT_DIR = Path('./outputs/experiment_bilstm_w2v')
BEST_MODEL_PATH = ORIGINAL_EXPERIMENT_DIR / "bilstm-w2v-crf-best.pt"
VOCAB_PATH = ORIGINAL_EXPERIMENT_DIR / "vocab.json"
HP_PATH = ORIGINAL_EXPERIMENT_DIR / "hyperparameters.json"

REPORT_PATH = OUTPUT_DIR / "classification_report.txt"
REPORT_JSON_PATH = OUTPUT_DIR / "classification_report.json"
CONF_MATRIX_PATH = OUTPUT_DIR / "confusion_matrix.png"
CONF_MATRIX_CSV_PATH = OUTPUT_DIR / "confusion_matrix.csv"
PREDICTIONS_PATH = OUTPUT_DIR / "test_predictions.txt"
SUMMARY_PATH = OUTPUT_DIR / "summary_report.json"

with open(HP_PATH, 'r') as f:
    HP = json.load(f)

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

print(f"Model: BiLSTM-CRF + W2V | Device: {DEVICE}")
print(f"Evaluating on dataset: {TEST_FILE}")


def load_bio_file(file_path):
    """Load BIO-format NER dataset from file.
    
    Args:
        file_path: Path to BIO-format file
        
    Returns:
        List of tuples (tokens, tags) for each sentence
    """
    sentences = []
    with open(file_path, 'r', encoding='utf-8') as f:
        tokens, tags = [], []
        for line in f:
            line = line.strip()
            if not line:
                if tokens:
                    sentences.append((tokens, tags))
                    tokens, tags = [], []
            else:
                parts = line.split()
                if len(parts) == 2:
                    tokens.append(parts[0])
                    tags.append(parts[1])
        if tokens:
            sentences.append((tokens, tags))
    return sentences

class NERDataset(Dataset):
    """Dataset class for NER data."""
    
    def __init__(self, sentences, word_to_ix, tag_to_ix):
        self.sentences = sentences
        self.word_to_ix = word_to_ix
        self.tag_to_ix = tag_to_ix

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        tokens, tags = self.sentences[idx]
        word_indices = [self.word_to_ix.get(t.lower(), 1) for t in tokens]
        tag_indices = [self.tag_to_ix.get(t, 0) for t in tags]
        return tokens, torch.tensor(word_indices), torch.tensor(tag_indices)

def collate_fn(batch):
    """Collate function for DataLoader to pad sequences."""
    tokens, words, tags = zip(*batch)
    max_len = max(len(w) for w in words)
    
    padded_words = torch.zeros(len(words), max_len, dtype=torch.long)
    padded_tags = torch.zeros(len(tags), max_len, dtype=torch.long)
    mask = torch.zeros(len(words), max_len, dtype=torch.bool)
    
    for i, (w, t) in enumerate(zip(words, tags)):
        l = len(w)
        padded_words[i, :l] = w
        padded_tags[i, :l] = t
        mask[i, :l] = True
        
    return tokens, padded_words, padded_tags, mask

from train_bilstm_w2v import BiLSTM_W2V_CRF


def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    return obj

def evaluate(model, loader, ix_to_tag, device, save_path=None, 
             report_path=None, report_json_path=None, predictions_path=None, 
             conf_matrix_csv_path=None, conf_matrix_path=None):
    """Evaluate model on a dataset.
    
    Args:
        model: Trained model
        loader: DataLoader for evaluation
        ix_to_tag: Index to tag mapping
        device: Device to run evaluation on
        save_path: Optional path to save results
        report_path: Optional path for text report
        report_json_path: Optional path for JSON report
        predictions_path: Optional path for predictions
        conf_matrix_csv_path: Optional path for confusion matrix CSV
        conf_matrix_path: Optional path for confusion matrix image
        
    Returns:
        Tuple of (report_dict, evaluation_time)
    """
    model.eval()
    all_preds, all_true, all_tokens = [], [], []
    eval_start_time = time.time()

    with torch.no_grad():
        for tokens, words, tags, mask in loader:
            words, mask = words.to(device, non_blocking=True), mask.to(device, non_blocking=True)
            
            preds = model.predict(words, mask)
            
            for i in range(len(preds)):
                seq_len = int(mask[i].sum())
                true_tags = [ix_to_tag.get(t.item(), '<UNK>') for t in tags[i][:seq_len]]
                pred_tags = [ix_to_tag.get(p, '<UNK>') for p in preds[i]]
                
                all_true.append(true_tags)
                all_preds.append(pred_tags)
                all_tokens.extend(tokens[i][:seq_len])

    eval_time = time.time() - eval_start_time
    
    report_str = seqeval_classification_report(all_true, all_preds, scheme=IOB2, digits=4, zero_division=0)
    report_dict = seqeval_classification_report(all_true, all_preds, scheme=IOB2, digits=4, output_dict=True, zero_division=0)
    report_dict = convert_numpy_types(report_dict)

    if save_path:
        with open(report_path, 'w', encoding='utf-8') as f: f.write(report_str)
        with open(report_json_path, 'w', encoding='utf-8') as f: json.dump(report_dict, f, indent=4)
        with open(predictions_path, 'w', encoding='utf-8') as f:
            flat_true = [tag for sent in all_true for tag in sent]
            flat_preds = [tag for sent in all_preds for tag in sent]
            for token, true_tag, pred_tag in zip(all_tokens, flat_true, flat_preds):
                f.write(f'{token}\t{true_tag}\t{pred_tag}\n')

        labels = sorted(list(set(tag for sent in all_true for tag in sent) | set(tag for sent in all_preds for tag in sent)))
        if 'O' in labels: labels.remove('O'); labels.append('O')
        cm = confusion_matrix([tag for sent in all_true for tag in sent], [tag for sent in all_preds for tag in sent], labels=labels)
        cm_df = pd.DataFrame(cm, index=labels, columns=labels)
        cm_df.to_csv(conf_matrix_csv_path)
        plt.figure(figsize=(12, 10)); sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues'); plt.title('Confusion Matrix'); plt.ylabel('Actual'); plt.xlabel('Predicted'); plt.savefig(conf_matrix_path); plt.close()

    return report_dict, eval_time

def main():
    """Main evaluation function."""
    start_time = time.time()
    print("Loading data and vocabulary...")
    
    test_data = load_bio_file(TEST_FILE)

    with open(VOCAB_PATH, 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)
    # This ensures the model architecture matches the saved weights
    original_word_to_ix = vocab_data['word_to_ix']
    original_tag_to_ix = vocab_data['tag_to_ix']
    original_ix_to_tag = {int(k): v for k, v in vocab_data['ix_to_tag'].items()}
    
    test_dataset = NERDataset(test_data, original_word_to_ix, original_tag_to_ix)
    loader_args = {"batch_size": HP["batch_size"], "collate_fn": collate_fn}
    if DEVICE.type == 'cuda':
        loader_args.update({
            "num_workers": min(4, os.cpu_count() or 1),
            "pin_memory": True,
            "persistent_workers": True,
            "prefetch_factor": 2
        })
    test_loader = DataLoader(test_dataset, **loader_args)

    print("Initializing BiLSTM-W2V-CRF model...")
    from flair.embeddings import WordEmbeddings
    word_embeddings = None
    try:
        word_embeddings = WordEmbeddings('id')
    except:
        pass
    
    model = BiLSTM_W2V_CRF(original_tag_to_ix, HP, word_to_ix=original_word_to_ix, word_embeddings=word_embeddings)
    model.to(DEVICE)
    

    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params:,}")

    print(f"\nLoading best model from {BEST_MODEL_PATH}...")
    state_dict = torch.load(BEST_MODEL_PATH, map_location=DEVICE, weights_only=True)
    model_state_dict = model.state_dict()
    
    filtered_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('word_embeddings.embedding.'):
            new_key = key.replace('word_embeddings.embedding.', 'embedding.')
            if new_key in model_state_dict and model_state_dict[new_key].shape == value.shape:
                filtered_state_dict[new_key] = value
        elif key in model_state_dict:
            if model_state_dict[key].shape == value.shape:
                filtered_state_dict[key] = value
            else:
                print(f"Warning: Skipping {key} due to shape mismatch: {model_state_dict[key].shape} vs {value.shape}")
        else:
            print(f"Warning: Skipping key {key} not found in model")
    
    missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
    if missing_keys:
        print(f"Warning: Missing keys in state_dict: {missing_keys[:5]}...")  # Show first 5
    if unexpected_keys:
        print(f"Warning: Unexpected keys in state_dict: {unexpected_keys[:5]}...")  # Show first 5
    
    print(f"\n--- Starting Evaluation on NER-UI Test Set ---")
    final_report_dict, final_eval_time = evaluate(
        model, test_loader, original_ix_to_tag, DEVICE, save_path=OUTPUT_DIR,
        report_path=REPORT_PATH, report_json_path=REPORT_JSON_PATH,
        predictions_path=PREDICTIONS_PATH, conf_matrix_csv_path=CONF_MATRIX_CSV_PATH,
        conf_matrix_path=CONF_MATRIX_PATH
    )
    
    print("\n--- Final Classification Report (NER-UI Test Set) ---")
    with open(REPORT_PATH, 'r', encoding='utf-8') as f:
        print(f.read())
    
    summary = {
        "model_name": "BiLSTM-CRF + W2V",
        "evaluated_on": str(TEST_FILE),
        "original_model_path": str(BEST_MODEL_PATH),
        "device": str(DEVICE),
        "evaluation_timestamp": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
        "evaluation_time_seconds": final_eval_time,
        "dataset_info": {
            "test_file": str(TEST_FILE),
            "tagset_size": len(original_tag_to_ix)
        },
        "hyperparameters": HP,
        "final_test_metrics": final_report_dict
    }
    with open(SUMMARY_PATH, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4)
        
    print(f"\nAll evaluation results have been saved to directory: {OUTPUT_DIR}")
    print(f"Final summary report saved to {SUMMARY_PATH}")

if __name__ == '__main__':
    main()
