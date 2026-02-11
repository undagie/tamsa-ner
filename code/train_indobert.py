import json
import torch
import torch.nn as nn
import torch.optim as optim
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
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix
from seqeval.metrics import classification_report as seqeval_classification_report
from seqeval.scheme import IOB2
from torch.amp import autocast, GradScaler

TRAIN_FILE = _ROOT / "data" / "idner2k" / "train_bio.txt"
DEV_FILE = _ROOT / "data" / "idner2k" / "dev_bio.txt"
TEST_FILE = _ROOT / "data" / "idner2k" / "test_bio.txt"

OUTPUT_DIR = _ROOT / "outputs" / "experiment_indobert"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = OUTPUT_DIR / "indobert-crf-best.pt"
VOCAB_PATH = OUTPUT_DIR / "vocab.json"
HP_PATH = OUTPUT_DIR / "hyperparameters.json"
HISTORY_PATH = OUTPUT_DIR / "training_history.csv"
REPORT_PATH = OUTPUT_DIR / "classification_report.txt"
REPORT_JSON_PATH = OUTPUT_DIR / "classification_report.json"
CONF_MATRIX_PATH = OUTPUT_DIR / "confusion_matrix.png"
CONF_MATRIX_CSV_PATH = OUTPUT_DIR / "confusion_matrix.csv"
PREDICTIONS_PATH = OUTPUT_DIR / "test_predictions.txt"
SUMMARY_PATH = OUTPUT_DIR / "summary_report.json"

HP = {
    "transformer_model": "indobenchmark/indobert-base-p1",
    "transformer_dim": 768,
    "dropout": 0.3,
    "learning_rate": 5e-5,
    "epochs": 100,
    "batch_size": 8,
    "early_stopping_patience": 15
}

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
print(f"Model: IndoBERT-CRF | Device: {DEVICE}")


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

def build_tag_vocab(sentences):
    """Build tag vocabulary mappings from sentences.
    
    Args:
        sentences: List of tuples (tokens, tags)
        
    Returns:
        Tuple of (tag_to_ix, ix_to_tag) dictionaries
    """
    tag_counts = Counter(tag for _, tags in sentences for tag in tags)
    tag_to_ix = {tag: i for i, (tag, _) in enumerate(tag_counts.items())}
    ix_to_tag = {i: tag for tag, i in tag_to_ix.items()}
    return tag_to_ix, ix_to_tag

class NERIndoBERTDataset(Dataset):
    """Dataset class for NER data with IndoBERT tokenization."""
    
    def __init__(self, sentences, tag_to_ix, tokenizer):
        self.sentences = sentences
        self.tag_to_ix = tag_to_ix
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        tokens, tags = self.sentences[idx]
        tag_indices = [self.tag_to_ix[t] for t in tags]

        bert_subwords = []
        word_to_subword_map = []
        for token in tokens:
            subws = self.tokenizer.tokenize(token)
            word_to_subword_map.append(len(bert_subwords))
            bert_subwords.extend(subws)
        
        bert_indices = self.tokenizer.convert_tokens_to_ids(bert_subwords)
        return tokens, bert_indices, tag_indices, word_to_subword_map

def collate_fn_indobert(batch):
    """Collate function for DataLoader to pad BERT sequences."""
    tokens, bert_indices, tag_indices, word_maps = zip(*batch)

    max_sub_len = max(len(s) for s in bert_indices)
    padded_bert_indices = torch.zeros(len(bert_indices), max_sub_len, dtype=torch.long)
    bert_attention_mask = torch.zeros(len(bert_indices), max_sub_len, dtype=torch.long)
    for i, s in enumerate(bert_indices):
        l = len(s)
        padded_bert_indices[i, :l] = torch.tensor(s)
        bert_attention_mask[i, :l] = 1

    max_word_len = max(len(t) for t in tag_indices)
    padded_tags = torch.zeros(len(tag_indices), max_word_len, dtype=torch.long)
    word_mask = torch.zeros(len(tag_indices), max_word_len, dtype=torch.bool)
    for i, t in enumerate(tag_indices):
        l = len(t)
        padded_tags[i, :l] = torch.tensor(t)
        word_mask[i, :l] = True

    return tokens, padded_bert_indices, bert_attention_mask, padded_tags, word_mask, word_maps


class IndoBERT_CRF(nn.Module):
    """IndoBERT-CRF model for Named Entity Recognition."""
    
    def __init__(self, tag_to_ix, hp):
        super(IndoBERT_CRF, self).__init__()
        self.hp = hp
        self.tagset_size = len(tag_to_ix)
        self.transformer = AutoModel.from_pretrained(self.hp["transformer_model"])
        
        self.dropout = nn.Dropout(self.hp["dropout"])
        self.hidden2tag = nn.Linear(self.hp["transformer_dim"], self.tagset_size)
        self.crf = CRF(self.tagset_size, batch_first=True)

    def _get_features(self, bert_indices, bert_mask, word_maps):
        transformer_outputs = self.transformer(bert_indices, attention_mask=bert_mask)
        transformer_hidden_states = transformer_outputs.last_hidden_state

        batch_size = bert_indices.size(0)
        seq_len = max(len(m) for m in word_maps)
        word_level_features = torch.zeros(batch_size, seq_len, self.hp["transformer_dim"], device=DEVICE)

        for i, mapping in enumerate(word_maps):
            for j, subword_idx in enumerate(mapping):
                if j < seq_len:
                    word_level_features[i, j, :] = transformer_hidden_states[i, subword_idx, :]

        features = self.dropout(word_level_features)
        emissions = self.hidden2tag(features)
        return emissions

    def forward(self, bert_indices, bert_mask, tags, word_mask, word_maps):
        emissions = self._get_features(bert_indices, bert_mask, word_maps)
        loss = -self.crf(emissions, tags, mask=word_mask, reduction='mean')
        return loss

    def predict(self, bert_indices, bert_mask, word_mask, word_maps):
        emissions = self._get_features(bert_indices, bert_mask, word_maps)
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

def evaluate(model, loader, ix_to_tag, device, save_path=None):
    """Evaluate model on a dataset.
    
    Args:
        model: Trained model
        loader: DataLoader for evaluation
        ix_to_tag: Index to tag mapping
        device: Device to run evaluation on
        save_path: Optional path to save results
        
    Returns:
        Tuple of (report_dict, evaluation_time)
    """
    model.eval()
    all_preds, all_true, all_tokens = [], [], []
    eval_start_time = time.time()

    with torch.no_grad():
        for tokens, bert_indices, bert_mask, tags, word_mask, word_maps in loader:
            bert_indices, bert_mask, word_mask = bert_indices.to(device, non_blocking=True), bert_mask.to(device, non_blocking=True), word_mask.to(device, non_blocking=True)
            
            preds = model.predict(bert_indices, bert_mask, word_mask, word_maps)
            
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
        with open(REPORT_PATH, 'w', encoding='utf-8') as f: f.write(report_str)
        with open(REPORT_JSON_PATH, 'w', encoding='utf-8') as f: json.dump(report_dict, f, indent=4)
        with open(PREDICTIONS_PATH, 'w', encoding='utf-8') as f:
            flat_true = [tag for sent in all_true for tag in sent]
            flat_preds = [tag for sent in all_preds for tag in sent]
            for token, true_tag, pred_tag in zip(all_tokens, flat_true, flat_preds):
                f.write(f'{token}\t{true_tag}\t{pred_tag}\n')

        labels = sorted(list(set(tag for sent in all_true for tag in sent) | set(tag for sent in all_preds for tag in sent)))
        if 'O' in labels: labels.remove('O'); labels.append('O')
        cm = confusion_matrix([tag for sent in all_true for tag in sent], [tag for sent in all_preds for tag in sent], labels=labels)
        cm_df = pd.DataFrame(cm, index=labels, columns=labels)
        cm_df.to_csv(CONF_MATRIX_CSV_PATH)
        plt.figure(figsize=(12, 10)); sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues'); plt.title('Confusion Matrix'); plt.ylabel('Actual'); plt.xlabel('Predicted'); plt.savefig(CONF_MATRIX_PATH); plt.close()

    return report_dict, eval_time

def main():
    """Main training function."""
    start_time = time.time()
    print("Loading tokenizer and data...")
    tokenizer = AutoTokenizer.from_pretrained(HP["transformer_model"])
    train_data = load_bio_file(TRAIN_FILE)
    dev_data = load_bio_file(DEV_FILE)
    test_data = load_bio_file(TEST_FILE)

    tag_to_ix, ix_to_tag = build_tag_vocab(train_data + dev_data)
    
    vocab_data = {'tag_to_ix': tag_to_ix, 'ix_to_tag': ix_to_tag}
    with open(VOCAB_PATH, 'w', encoding='utf-8') as f: json.dump(vocab_data, f, indent=4)
    with open(HP_PATH, 'w', encoding='utf-8') as f: json.dump(HP, f, indent=4)

    train_dataset = NERIndoBERTDataset(train_data, tag_to_ix, tokenizer)
    dev_dataset = NERIndoBERTDataset(dev_data, tag_to_ix, tokenizer)
    test_dataset = NERIndoBERTDataset(test_data, tag_to_ix, tokenizer)

    loader_args = {"batch_size": HP["batch_size"], "collate_fn": collate_fn_indobert}
    if DEVICE.type == 'cuda':
        loader_args.update({
            "num_workers": min(4, os.cpu_count() or 1),
            "pin_memory": True,
            "persistent_workers": True,
            "prefetch_factor": 2
        })

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)
    dev_loader = DataLoader(dev_dataset, **loader_args)
    test_loader = DataLoader(test_dataset, **loader_args)

    print("Initializing IndoBERT-CRF model...")
    model = IndoBERT_CRF(tag_to_ix, HP).to(DEVICE)



    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=HP["learning_rate"])

    num_training_steps = len(train_loader) * HP["epochs"]
    num_warmup_steps = int(0.1 * num_training_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
    )

    print("\n--- Starting Training ---")
    best_dev_f1 = -1
    patience_counter = 0
    total_training_time = 0
    training_history = []

    for epoch in range(HP["epochs"]):
        epoch_start_time = time.time()
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        
        for batch_idx, (_, bert_indices, bert_mask, tags, word_mask, word_maps) in enumerate(train_loader):
            bert_indices, bert_mask, tags, word_mask = bert_indices.to(DEVICE, non_blocking=True), bert_mask.to(DEVICE, non_blocking=True), tags.to(DEVICE, non_blocking=True), word_mask.to(DEVICE, non_blocking=True)
            
            loss = model(bert_indices, bert_mask, tags, word_mask, word_maps)
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: Invalid loss detected at batch {batch_idx}: {loss.item()}. Skipping batch.")
                continue
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        dev_report_dict, _ = evaluate(model, dev_loader, ix_to_tag, DEVICE)
        dev_f1 = dev_report_dict['micro avg']['f1-score']
        epoch_time = time.time() - epoch_start_time
        total_training_time += epoch_time
        
        print(f"Epoch {epoch+1}/{HP['epochs']} | Train Loss: {avg_loss:.4f} | Dev F1: {dev_f1:.4f} | Time: {epoch_time:.2f}s")
        
        history_epoch = {'epoch': epoch + 1, 'train_loss': avg_loss, 'dev_f1': dev_f1, 'dev_precision': dev_report_dict['micro avg']['precision'], 'dev_recall': dev_report_dict['micro avg']['recall']}
        training_history.append(history_epoch)

        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            patience_counter = 0
            print(f"  -> Performance improved, saving model to {BEST_MODEL_PATH}")
        else:
            patience_counter += 1
            print(f"  -> No improvement. Patience: {patience_counter}/{HP['early_stopping_patience']}")

        if patience_counter >= HP['early_stopping_patience']:
            print("\nEarly stopping reached.")
            break

    print("\n--- Training Complete ---")
    pd.DataFrame(training_history).to_csv(HISTORY_PATH, index=False)
    print(f"Training history saved to {HISTORY_PATH}")

    print("\nLoading best model for final evaluation on test set...")
    model.load_state_dict(torch.load(BEST_MODEL_PATH, weights_only=True))
    final_report_dict, final_eval_time = evaluate(model, test_loader, ix_to_tag, DEVICE, save_path=OUTPUT_DIR)
    
    print("\n--- Final Classification Report (Test Set) ---")
    with open(REPORT_PATH, 'r', encoding='utf-8') as f: print(f.read())
    
    summary = {
        "model_name": "IndoBERT-CRF",
        "best_model_path": str(BEST_MODEL_PATH),
        "device": str(DEVICE),
        "timestamps": {
            "training_start": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)),
            "training_end": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        },
        "training_duration_seconds": total_training_time,
        "evaluation_time_seconds": final_eval_time,
        "dataset_info": {
            "training_file": str(TRAIN_FILE),
            "development_file": str(DEV_FILE),
            "test_file": str(TEST_FILE),
            "vocab_size": tokenizer.vocab_size,
            "tagset_size": len(tag_to_ix)
        },
        "model_info": {
            "num_trainable_params": num_params
        },
        "hyperparameters": HP,
        "final_test_metrics": final_report_dict
    }
    with open(SUMMARY_PATH, 'w', encoding='utf-8') as f: json.dump(summary, f, indent=4)
        
    print(f"\nAll experiment results have been saved to directory: {OUTPUT_DIR}")
    print(f"Final summary report saved to {SUMMARY_PATH}")

if __name__ == '__main__':
    main()
