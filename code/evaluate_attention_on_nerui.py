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
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix
from seqeval.metrics import classification_report as seqeval_classification_report
from seqeval.scheme import IOB2

TEST_FILE = Path('./data/nerui/test_bio.txt')

OUTPUT_DIR = Path('./outputs/evaluation_nerui_attention_fusion')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ORIGINAL_EXPERIMENT_DIR = Path('./outputs/experiment_attention_fusion')
BEST_MODEL_PATH = ORIGINAL_EXPERIMENT_DIR / "attention-fusion-crf-best.pt"
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

print(f"Model: MultiSource Attention CRF | Device: {DEVICE}")
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

class NERMultiSourceDataset(Dataset):
    """Dataset class for NER data with multiple feature sources."""
    
    def __init__(self, sentences, word_to_ix, tag_to_ix, char_to_ix, tokenizer):
        self.sentences = sentences
        self.word_to_ix = word_to_ix
        self.tag_to_ix = tag_to_ix
        self.char_to_ix = char_to_ix
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        tokens, tags = self.sentences[idx]
        
        word_indices = [self.word_to_ix.get(t.lower(), 1) for t in tokens] # 1 is <UNK>
        tag_indices = [self.tag_to_ix.get(t, self.tag_to_ix['O']) for t in tags]
        char_indices = [[self.char_to_ix.get(c, 1) for c in token] for token in tokens]

        bert_subwords = []
        word_to_subword_map = []
        for token in tokens:
            subws = self.tokenizer.tokenize(token)
            word_to_subword_map.append(len(bert_subwords))
            bert_subwords.extend(subws)
        
        bert_indices = self.tokenizer.convert_tokens_to_ids(bert_subwords)
        return tokens, word_indices, char_indices, tag_indices, bert_indices, word_to_subword_map

def collate_fn_multisource(batch):
    tokens, word_indices, char_indices, tag_indices, bert_indices, word_maps = zip(*batch)

    max_word_len = max(len(w) for w in word_indices)
    padded_words = torch.zeros(len(word_indices), max_word_len, dtype=torch.long)
    padded_tags = torch.zeros(len(tag_indices), max_word_len, dtype=torch.long)
    word_mask = torch.zeros(len(word_indices), max_word_len, dtype=torch.bool)
    for i, (w, t) in enumerate(zip(word_indices, tag_indices)):
        l = len(w)
        padded_words[i, :l] = torch.tensor(w)
        padded_tags[i, :l] = torch.tensor(t)
        word_mask[i, :l] = True

    max_char_len = max(len(c) for cl in char_indices for c in cl) if any(any(c) for c in char_indices) else 1
    padded_chars = torch.zeros(len(char_indices), max_word_len, max_char_len, dtype=torch.long)
    for i, cl in enumerate(char_indices):
        for j, c in enumerate(cl):
            l = len(c)
            padded_chars[i, j, :l] = torch.tensor(c)

    max_sub_len = max(len(s) for s in bert_indices)
    padded_bert_indices = torch.zeros(len(bert_indices), max_sub_len, dtype=torch.long)
    bert_attention_mask = torch.zeros(len(bert_indices), max_sub_len, dtype=torch.long)
    for i, s in enumerate(bert_indices):
        l = len(s)
        padded_bert_indices[i, :l] = torch.tensor(s)
        bert_attention_mask[i, :l] = 1

    return tokens, padded_words, padded_chars, padded_tags, word_mask, padded_bert_indices, bert_attention_mask, word_maps


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

class AttentionFusion(nn.Module):
    """Attention mechanism for fusing multiple feature sources."""
    
    def __init__(self, feature_dim, attention_dim):
        super(AttentionFusion, self).__init__()
        self.attention_projection = nn.Linear(feature_dim, attention_dim)
        self.attention_scorer = nn.Linear(attention_dim, 1)

    def forward(self, features_list):
        stacked_features = torch.stack(features_list, dim=2)
        batch_size, seq_len, num_sources, feature_dim = stacked_features.shape
        flat_features = stacked_features.view(-1, feature_dim)
        projected_features = torch.tanh(self.attention_projection(flat_features))
        scores = self.attention_scorer(projected_features)
        scores = scores.view(batch_size, seq_len, num_sources)
        weights = torch.softmax(scores, dim=2)
        weighted_features = (weights.unsqueeze(-1) * stacked_features).sum(dim=2)
        return weighted_features, weights

class MultiSource_Attention_CRF(nn.Module):
    """Multi-source attention fusion model with CRF for Named Entity Recognition."""
    
    def __init__(self, word_vocab_size, char_vocab_size, tag_to_ix, hp):
        super(MultiSource_Attention_CRF, self).__init__()
        self.hp = hp
        self.tagset_size = len(tag_to_ix)
        self.w2v_embedding = nn.Embedding(word_vocab_size, self.hp['w2v_embedding_dim'], padding_idx=0)
        self.char_cnn = CharCNN(char_vocab_size, self.hp['char_embedding_dim'], self.hp['char_cnn_filters'], self.hp['char_cnn_kernel_size'], self.hp['dropout'])
        self.transformer = AutoModel.from_pretrained(self.hp["transformer_model"])
        self.projection_dim = self.hp['lstm_hidden_dim']
        self.w2v_proj = nn.Linear(self.hp['w2v_embedding_dim'], self.projection_dim)
        self.char_proj = nn.Linear(self.hp['char_cnn_filters'], self.projection_dim)
        self.trans_proj = nn.Linear(self.hp['transformer_dim'], self.projection_dim)
        self.fusion = AttentionFusion(self.projection_dim, self.projection_dim // 2)
        self.lstm = nn.LSTM(self.projection_dim, self.hp['lstm_hidden_dim'] // 2, 
                            num_layers=self.hp['lstm_layers'], bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(self.hp['dropout'])
        self.hidden2tag = nn.Linear(self.hp['lstm_hidden_dim'], self.tagset_size)
        self.crf = CRF(self.tagset_size, batch_first=True)

    def _get_features(self, words, chars, bert_indices, bert_mask, word_maps):
        w2v_features = self.w2v_embedding(words)
        char_cnn_features = self.char_cnn(chars)
        transformer_outputs = self.transformer(bert_indices, attention_mask=bert_mask)
        transformer_hidden_states = transformer_outputs.last_hidden_state
        batch_size, seq_len, _ = w2v_features.shape
        transformer_features = torch.zeros(batch_size, seq_len, self.hp["transformer_dim"], device=DEVICE)
        for i, mapping in enumerate(word_maps):
            if not mapping: continue
            for j, subword_idx in enumerate(mapping):
                 if j < seq_len:
                    transformer_features[i, j, :] = transformer_hidden_states[i, subword_idx, :]
        w2v_proj = torch.relu(self.w2v_proj(w2v_features))
        char_proj = torch.relu(self.char_proj(char_cnn_features))
        trans_proj = torch.relu(self.trans_proj(transformer_features))
        fused_features, _ = self.fusion([w2v_proj, char_proj, trans_proj])
        fused_features = self.dropout(fused_features)
        lstm_out, _ = self.lstm(fused_features)
        lstm_out = self.dropout(lstm_out)
        emissions = self.hidden2tag(lstm_out)
        return emissions

    def predict(self, words, chars, word_mask, bert_indices, bert_mask, word_maps):
        emissions = self._get_features(words, chars, bert_indices, bert_mask, word_maps)
        return self.crf.decode(emissions, mask=word_mask)


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
        for tokens, words, chars, tags, word_mask, bert_indices, bert_mask, word_maps in loader:
            words, chars, word_mask = words.to(device, non_blocking=True), chars.to(device, non_blocking=True), word_mask.to(device, non_blocking=True)
            bert_indices, bert_mask = bert_indices.to(device, non_blocking=True), bert_mask.to(device, non_blocking=True)
            
            preds = model.predict(words, chars, word_mask, bert_indices, bert_mask, word_maps)
            
            for i in range(len(preds)):
                seq_len = int(word_mask[i].sum())
                true_tags = [ix_to_tag.get(t.item(), '<UNK>') for t in tags[i][:seq_len]]
                pred_tags = [ix_to_tag.get(p, '<UNK>') for p in preds[i][:seq_len]]
                
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
    print("Loading tokenizer and data...")
    tokenizer = AutoTokenizer.from_pretrained(HP["transformer_model"])
    
    test_data = load_bio_file(TEST_FILE)

    with open(VOCAB_PATH, 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)
    word_to_ix = vocab_data['word_to_ix']
    tag_to_ix = vocab_data['tag_to_ix']
    ix_to_tag = {int(k): v for k, v in vocab_data['ix_to_tag'].items()}
    char_to_ix = vocab_data['char_to_ix']

    test_dataset = NERMultiSourceDataset(test_data, word_to_ix, tag_to_ix, char_to_ix, tokenizer)
    loader_args = {"batch_size": HP["batch_size"], "collate_fn": collate_fn_multisource}
    if DEVICE.type == 'cuda':
        loader_args.update({
            "num_workers": min(4, os.cpu_count() or 1),
            "pin_memory": True,
            "persistent_workers": True,
            "prefetch_factor": 2
        })
    test_loader = DataLoader(test_dataset, **loader_args)

    print("Initializing MultiSource Attention CRF model...")
    model = MultiSource_Attention_CRF(len(word_to_ix), len(char_to_ix), tag_to_ix, HP).to(DEVICE)
    

    
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
        "model_name": "MultiSource Attention CRF",
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
