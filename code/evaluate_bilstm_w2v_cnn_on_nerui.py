import json
import os
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from collections import Counter

_ROOT = Path(__file__).resolve().parent.parent
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

TEST_FILE = _ROOT / "data" / "nerui" / "test_bio.txt"

OUTPUT_DIR = _ROOT / "outputs" / "evaluation_nerui_bilstm_w2v_cnn"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ORIGINAL_EXPERIMENT_DIR = _ROOT / "outputs" / "experiment_bilstm_w2v_cnn"
BEST_MODEL_PATH = ORIGINAL_EXPERIMENT_DIR / "bilstm-w2v-cnn-crf-best.pt"
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

print(f"Model: BiLSTM-CRF + W2V + CharCNN | Device: {DEVICE}")
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
    """Dataset class for NER data with character-level features."""
    
    def __init__(self, sentences, word_to_ix, tag_to_ix, char_to_ix):
        self.sentences = sentences
        self.word_to_ix = word_to_ix
        self.tag_to_ix = tag_to_ix
        self.char_to_ix = char_to_ix

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        tokens, tags = self.sentences[idx]
        word_indices = [self.word_to_ix.get(t.lower(), 1) for t in tokens]
        tag_indices = [self.tag_to_ix.get(t, 0) for t in tags]
        char_indices = [[self.char_to_ix.get(c, 1) for c in token] for token in tokens]
        return tokens, torch.tensor(word_indices), torch.tensor(tag_indices), char_indices

def collate_fn(batch):
    tokens, words, tags, chars = zip(*batch)
    max_word_len = max(len(w) for w in words)

    padded_words = torch.zeros(len(words), max_word_len, dtype=torch.long)
    padded_tags = torch.zeros(len(tags), max_word_len, dtype=torch.long)
    mask = torch.zeros(len(words), max_word_len, dtype=torch.bool)

    for i, (w, t) in enumerate(zip(words, tags)):
        l = len(w)
        padded_words[i, :l] = w
        padded_tags[i, :l] = t
        mask[i, :l] = True

    max_char_len = max(len(c) for cl in chars for c in cl) if any(any(c) for c in chars) else 1
    padded_chars = torch.zeros(len(chars), max_word_len, max_char_len, dtype=torch.long)
    for i, cl in enumerate(chars):
        for j, c in enumerate(cl):
            l = len(c)
            padded_chars[i, j, :l] = torch.tensor(c)

    return tokens, padded_words, padded_tags, mask, padded_chars

class CharCNN(nn.Module):
    """Character-level CNN for word representation."""
    
    def __init__(self, char_vocab_size, embedding_dim, num_filters, kernel_size, dropout):
        super(CharCNN, self).__init__()
        self.embedding = nn.Embedding(char_vocab_size, embedding_dim, padding_idx=0)
        self.conv = nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=kernel_size, padding=kernel_size//2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_len, word_len = x.shape
        x = x.view(-1, word_len)
        x = self.embedding(x)
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = torch.max(x, dim=2)[0]
        x = x.view(batch_size, seq_len, -1)
        return self.dropout(x)

class BiLSTM_W2V_CNN_CRF(nn.Module):
    """BiLSTM-CRF model with Word2Vec and CharCNN embeddings for Named Entity Recognition."""
    
    def __init__(self, w2v_vocab_size, char_vocab_size, tag_to_ix, hp):
        super(BiLSTM_W2V_CNN_CRF, self).__init__()
        self.w2v_embedding = nn.Embedding(w2v_vocab_size, hp["w2v_embedding_dim"], padding_idx=0)
        self.char_cnn = CharCNN(char_vocab_size, hp["char_embedding_dim"], hp["char_cnn_filters"], hp["char_cnn_kernel_size"], hp["dropout"])
        
        lstm_input_size = hp["w2v_embedding_dim"] + hp["char_cnn_filters"]
        self.lstm = nn.LSTM(lstm_input_size, hp["lstm_hidden_dim"] // 2,
                            num_layers=hp["lstm_layers"], bidirectional=True, batch_first=True)
        
        self.dropout = nn.Dropout(hp["dropout"])
        self.hidden2tag = nn.Linear(hp["lstm_hidden_dim"], len(tag_to_ix))
        self.crf = CRF(len(tag_to_ix), batch_first=True)

    def _get_lstm_features(self, words, chars):
        w2v_embeds = self.w2v_embedding(words)
        char_cnn_embeds = self.char_cnn(chars)
        
        combined_embeds = torch.cat([w2v_embeds, char_cnn_embeds], dim=2)
        combined_embeds = self.dropout(combined_embeds)
        
        lstm_out, _ = self.lstm(combined_embeds)
        lstm_out = self.dropout(lstm_out)
        emissions = self.hidden2tag(lstm_out)
        return emissions

    def predict(self, words, mask, chars):
        emissions = self._get_lstm_features(words, chars)
        return self.crf.decode(emissions, mask=mask)

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
        for tokens, words, tags, mask, chars in loader:
            words, mask, chars = words.to(device, non_blocking=True), mask.to(device, non_blocking=True), chars.to(device, non_blocking=True)
            
            preds = model.predict(words, mask, chars)
            
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
    word_to_ix = vocab_data['word_to_ix']
    tag_to_ix = vocab_data['tag_to_ix']
    ix_to_tag = {int(k): v for k, v in vocab_data['ix_to_tag'].items()}
    char_to_ix = vocab_data['char_to_ix']

    test_dataset = NERDataset(test_data, word_to_ix, tag_to_ix, char_to_ix)
    loader_args = {"batch_size": HP["batch_size"], "collate_fn": collate_fn}
    if DEVICE.type == 'cuda':
        loader_args.update({
            "num_workers": min(4, os.cpu_count() or 1),
            "pin_memory": True,
            "persistent_workers": True,
            "prefetch_factor": 2
        })
    test_loader = DataLoader(test_dataset, **loader_args)

    print("Initializing BiLSTM-W2V-CNN-CRF model...")
    model = BiLSTM_W2V_CNN_CRF(len(word_to_ix), len(char_to_ix), tag_to_ix, HP).to(DEVICE)
    

    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params:,}")

    print(f"\nLoading best model from {BEST_MODEL_PATH}...")
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE, weights_only=True))
    
    print(f"\n--- Starting Evaluation on NER-UI Test Set ---")
    final_report_dict, final_eval_time = evaluate(
        model, test_loader, ix_to_tag, DEVICE, save_path=OUTPUT_DIR,
        report_path=REPORT_PATH, report_json_path=REPORT_JSON_PATH,
        predictions_path=PREDICTIONS_PATH, conf_matrix_csv_path=CONF_MATRIX_CSV_PATH,
        conf_matrix_path=CONF_MATRIX_PATH
    )
    
    print("\n--- Final Classification Report (NER-UI Test Set) ---")
    with open(REPORT_PATH, 'r', encoding='utf-8') as f:
        print(f.read())
    
    summary = {
        "model_name": "BiLSTM-CRF + W2V + CharCNN",
        "evaluated_on": str(TEST_FILE),
        "original_model_path": str(BEST_MODEL_PATH),
        "device": str(DEVICE),
        "evaluation_timestamp": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
        "evaluation_time_seconds": final_eval_time,
        "dataset_info": {
            "test_file": str(TEST_FILE),
            "tagset_size": len(tag_to_ix)
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
