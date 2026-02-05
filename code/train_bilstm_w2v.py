import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from collections import Counter
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='torch')
warnings.filterwarnings('ignore', category=DeprecationWarning, module='torch')
from torchcrf import CRF
import time
from transformers import get_linear_schedule_with_warmup
from flair.embeddings import WordEmbeddings
from flair.data import Sentence
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix
from seqeval.metrics import classification_report as seqeval_classification_report
from seqeval.scheme import IOB2
from torch.amp import autocast, GradScaler

TRAIN_FILE = Path('./data/idner2k/train_bio.txt')
DEV_FILE = Path('./data/idner2k/dev_bio.txt')
TEST_FILE = Path('./data/idner2k/test_bio.txt')

OUTPUT_DIR = Path('./outputs/experiment_bilstm_w2v')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = OUTPUT_DIR / "bilstm-w2v-crf-best.pt"
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
    "embedding_dim": 300,
    "lstm_hidden_dim": 256,
    "lstm_layers": 1,
    "dropout": 0.3,
    "learning_rate": 1e-3,
    "epochs": 100,
    "batch_size": 16,
    "early_stopping_patience": 15,
    "gradient_accumulation_steps": 2,
    "use_amp": True
}

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
print(f"Model: BiLSTM-CRF + Word2Vec | Device: {DEVICE}")


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

def build_vocab(sentences):
    """Build vocabulary mappings from sentences.
    
    Args:
        sentences: List of tuples (tokens, tags)
        
    Returns:
        Tuple of (word_to_ix, tag_to_ix, ix_to_tag) dictionaries
    """
    word_counts = Counter(token.lower() for sentence, _ in sentences for token in sentence)
    tag_counts = Counter(tag for _, tags in sentences for tag in tags)
    
    word_to_ix = {'<PAD>': 0, '<UNK>': 1}
    for word, _ in word_counts.items():
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
            
    tag_to_ix = {tag: i for i, (tag, _) in enumerate(tag_counts.items())}
    ix_to_tag = {i: tag for tag, i in tag_to_ix.items()}
    
    return word_to_ix, tag_to_ix, ix_to_tag

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
        tag_indices = [self.tag_to_ix[t] for t in tags]
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

class BiLSTM_W2V_CRF(nn.Module):
    """BiLSTM-CRF model with Word2Vec embeddings for Named Entity Recognition."""
    
    def __init__(self, tag_to_ix, hp, word_to_ix=None, word_embeddings=None):
        super(BiLSTM_W2V_CRF, self).__init__()
        if word_to_ix is not None:
            vocab_size = len(word_to_ix)
            embedding_dim = hp.get("embedding_dim", 300)
            hidden_dim = hp.get("lstm_hidden_dim", 256)
            lstm_layers = hp.get("lstm_layers", 1)
            dropout = hp.get("dropout", 0.3)
        else:
            raise ValueError("BiLSTM_W2V_CRF requires tag_to_ix and hp parameters")
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=lstm_layers, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.hidden2tag = nn.Linear(hidden_dim, len(tag_to_ix))
        self.crf = CRF(len(tag_to_ix))
        
        self.word_embeddings = word_embeddings

    def _get_lstm_features(self, words, w2v_feats=None):
        embeds = self.embedding(words)
        embeds = self.dropout(embeds)
        lstm_out, _ = self.lstm(embeds)
        lstm_out = self.dropout(lstm_out)
        emissions = self.hidden2tag(lstm_out)
        return emissions

    def forward(self, words, tags, mask, w2v_feats=None):
        emissions = self._get_lstm_features(words, w2v_feats)
        
        emissions = emissions.transpose(0, 1)
        tags = tags.transpose(0, 1)
        mask = mask.transpose(0, 1)
        
        loss = -self.crf(emissions, tags, mask=mask, reduction='mean')
        return loss

    def predict(self, words, mask, w2v_feats=None):
        emissions = self._get_lstm_features(words, w2v_feats)
        
        emissions = emissions.transpose(0, 1)
        mask = mask.transpose(0, 1)
        
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
        for tokens, words, tags, mask in loader:
            words, mask = words.to(device, non_blocking=True), mask.to(device, non_blocking=True)
            
            preds = model.predict(words, mask)
            
            for i in range(len(preds)):
                seq_len = int(mask[i].sum())
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
    print("Loading and processing data...")
    train_data = load_bio_file(TRAIN_FILE)
    dev_data = load_bio_file(DEV_FILE)
    test_data = load_bio_file(TEST_FILE)

    word_to_ix, tag_to_ix, ix_to_tag = build_vocab(train_data + dev_data)
    
    vocab_data = {'word_to_ix': word_to_ix, 'tag_to_ix': tag_to_ix, 'ix_to_tag': ix_to_tag}
    with open(VOCAB_PATH, 'w', encoding='utf-8') as f: json.dump(vocab_data, f, indent=4)
    with open(HP_PATH, 'w', encoding='utf-8') as f: json.dump(HP, f, indent=4)

    train_dataset = NERDataset(train_data, word_to_ix, tag_to_ix)
    dev_dataset = NERDataset(dev_data, word_to_ix, tag_to_ix)
    test_dataset = NERDataset(test_data, word_to_ix, tag_to_ix)

    loader_args = {"batch_size": HP["batch_size"], "collate_fn": collate_fn}
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

    print("Initializing BiLSTM-CRF + Word2Vec model...")
    print("Loading pre-trained Word2Vec (FastText) via Flair...")
    fasttext = None
    try:
        fasttext = WordEmbeddings('id')
        print("-> Word2Vec embeddings loaded successfully.")
    except Exception as e:
        print(f"-> Warning: Failed to load Word2Vec embeddings: {e}. Training will continue with random embeddings.")
    
    model = BiLSTM_W2V_CRF(tag_to_ix, HP, word_to_ix=word_to_ix, word_embeddings=fasttext)
    model.to(DEVICE)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params:,}")

    if fasttext is not None:
        try:
            pretrained_weights = model.embedding.weight.data.clone()
            found_words = 0
            for word, i in word_to_ix.items():
                dummy_sentence = Sentence(word)
                fasttext.embed(dummy_sentence)
                if dummy_sentence[0].embedding.nelement() > 0:
                    pretrained_weights[i] = dummy_sentence[0].embedding
                    found_words += 1
            model.embedding.weight.data.copy_(pretrained_weights)
            print(f"-> Successfully loaded weights for {found_words} out of {len(word_to_ix)} words.")
        except Exception as e:
            print(f"-> Failed to load pre-trained embeddings: {e}. Training will continue with random embeddings.")

    optimizer = optim.Adam(model.parameters(), lr=HP["learning_rate"])
    scaler = GradScaler() if HP.get("use_amp", True) and torch.cuda.is_available() else None

    effective_batches_per_epoch = len(train_loader) // HP.get("gradient_accumulation_steps", 1)
    num_training_steps = effective_batches_per_epoch * HP["epochs"]
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
        
        for batch_idx, (_, words, tags, mask) in enumerate(train_loader):
            words, tags, mask = words.to(DEVICE, non_blocking=True), tags.to(DEVICE, non_blocking=True), mask.to(DEVICE, non_blocking=True)
            
            if scaler:
                with autocast(device_type='cuda'):
                    loss = model(words, tags, mask, w2v_feats=None)
                    loss = loss / HP.get("gradient_accumulation_steps", 1)
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Warning: Invalid loss detected at batch {batch_idx}: {loss.item()}. Skipping batch.")
                    continue
                scaler.scale(loss).backward()
            else:
                loss = model(words, tags, mask, w2v_feats=None)
                loss = loss / HP.get("gradient_accumulation_steps", 1)
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Warning: Invalid loss detected at batch {batch_idx}: {loss.item()}. Skipping batch.")
                    continue
                loss.backward()
            
            if (batch_idx + 1) % HP.get("gradient_accumulation_steps", 1) == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * HP.get("gradient_accumulation_steps", 1)

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
        "model_name": "BiLSTM-CRF + Word2Vec (FastText)",
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
            "vocab_size": len(word_to_ix),
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