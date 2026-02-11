import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from collections import Counter

_ROOT = Path(__file__).resolve().parent.parent
from torchcrf import CRF
import time
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from flair.embeddings import WordEmbeddings
from flair.data import Sentence
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix
from seqeval.metrics import classification_report as seqeval_classification_report
from seqeval.scheme import IOB2

TRAIN_FILE = _ROOT / "data" / "idner2k" / "train_bio.txt"
DEV_FILE = _ROOT / "data" / "idner2k" / "dev_bio.txt"
TEST_FILE = _ROOT / "data" / "idner2k" / "test_bio.txt"

OUTPUT_DIR = _ROOT / "outputs" / "experiment_attention_fusion"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = OUTPUT_DIR / "attention-fusion-crf-best.pt"
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
    "w2v_embedding_dim": 300,
    "char_embedding_dim": 50,
    "char_cnn_filters": 50,
    "char_cnn_kernel_size": 3,
    "lstm_hidden_dim": 256,
    "lstm_layers": 1,
    "dropout": 0.3,
    "learning_rate": 1e-5,
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
print(f"Model: MultiSource Attention CRF | Device: {DEVICE}")


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
    """Build vocabulary mappings from sentences including character vocabulary.
    
    Args:
        sentences: List of tuples (tokens, tags)
        
    Returns:
        Tuple of (word_to_ix, tag_to_ix, ix_to_tag, char_to_ix) dictionaries
    """
    word_counts = Counter(token.lower() for sentence, _ in sentences for token in sentence)
    tag_counts = Counter(tag for _, tags in sentences for tag in tags)
    chars = set("".join([word for sentence, _ in sentences for word in sentence]))

    word_to_ix = {'<PAD>': 0, '<UNK>': 1}
    for word, _ in word_counts.items():
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)

    tag_to_ix = {tag: i for i, (tag, _) in enumerate(tag_counts.items())}
    ix_to_tag = {i: tag for tag, i in tag_to_ix.items()}

    char_to_ix = {'<PAD>': 0, '<UNK>': 1}
    for char in chars:
        if char not in char_to_ix:
            char_to_ix[char] = len(char_to_ix)

    return word_to_ix, tag_to_ix, ix_to_tag, char_to_ix

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
        
        word_indices = [self.word_to_ix.get(t.lower(), 1) for t in tokens]
        tag_indices = [self.tag_to_ix[t] for t in tags]
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
    """Character-level CNN for morphological word representation."""
    
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
    """Attention-based fusion module for multi-source features."""
    
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
    """Multi-Source Attention CRF model for Named Entity Recognition."""
    
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
        device = w2v_features.device
        transformer_features = torch.zeros(batch_size, seq_len, self.hp["transformer_dim"], device=device)
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

    def forward(self, words, chars, tags, word_mask, bert_indices, bert_mask, word_maps):
        emissions = self._get_features(words, chars, bert_indices, bert_mask, word_maps)
        loss = -self.crf(emissions, tags, mask=word_mask, reduction='mean')
        return loss

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
        with open(REPORT_PATH, 'w', encoding='utf-8') as f:
            f.write(report_str)
        with open(REPORT_JSON_PATH, 'w', encoding='utf-8') as f:
            json.dump(report_dict, f, indent=4)

        with open(PREDICTIONS_PATH, 'w', encoding='utf-8') as f:
            flat_true = [tag for sent in all_true for tag in sent]
            flat_preds = [tag for sent in all_preds for tag in sent]
            for token, true_tag, pred_tag in zip(all_tokens, flat_true, flat_preds):
                f.write(f'{token}\t{true_tag}\t{pred_tag}\n')

        labels = sorted(list(set(tag for sent in all_true for tag in sent) | set(tag for sent in all_preds for tag in sent)))
        if 'O' in labels:
            labels.remove('O')
            labels.append('O')

        flat_true = [tag for sent in all_true for tag in sent]
        flat_preds = [tag for sent in all_preds for tag in sent]
        
        cm = confusion_matrix(flat_true, flat_preds, labels=labels)
        cm_df = pd.DataFrame(cm, index=labels, columns=labels)
        cm_df.to_csv(CONF_MATRIX_CSV_PATH)

        plt.figure(figsize=(12, 10))
        sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix - MultiSource Attention CRF')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig(CONF_MATRIX_PATH, dpi=150)
        plt.close()

    return report_dict, eval_time

def main():
    """Main training function for MultiSource Attention CRF model."""
    start_time = time.time()

    print("=" * 60)
    print("MultiSource Attention CRF for Indonesian NER")
    print("=" * 60)

    print("\n[1/5] Loading tokenizer and data...")
    tokenizer = AutoTokenizer.from_pretrained(HP["transformer_model"])
    train_data = load_bio_file(TRAIN_FILE)
    dev_data = load_bio_file(DEV_FILE)
    test_data = load_bio_file(TEST_FILE)

    print(f"  - Training samples: {len(train_data)}")
    print(f"  - Development samples: {len(dev_data)}")
    print(f"  - Test samples: {len(test_data)}")

    word_to_ix, tag_to_ix, ix_to_tag, char_to_ix = build_vocab(train_data + dev_data)

    print(f"  - Vocabulary size: {len(word_to_ix)}")
    print(f"  - Character vocabulary size: {len(char_to_ix)}")
    print(f"  - Tag vocabulary size: {len(tag_to_ix)}")
    print(f"  - Tags: {list(tag_to_ix.keys())}")
    
    vocab_data = {'word_to_ix': word_to_ix, 'tag_to_ix': tag_to_ix, 'ix_to_tag': ix_to_tag, 'char_to_ix': char_to_ix}
    with open(VOCAB_PATH, 'w', encoding='utf-8') as f:
        json.dump(vocab_data, f, indent=4)
    with open(HP_PATH, 'w', encoding='utf-8') as f:
        json.dump(HP, f, indent=4)

    print("\n[2/5] Creating data loaders...")
    train_dataset = NERMultiSourceDataset(train_data, word_to_ix, tag_to_ix, char_to_ix, tokenizer)
    dev_dataset = NERMultiSourceDataset(dev_data, word_to_ix, tag_to_ix, char_to_ix, tokenizer)
    test_dataset = NERMultiSourceDataset(test_data, word_to_ix, tag_to_ix, char_to_ix, tokenizer)

    loader_args = {"batch_size": HP["batch_size"], "collate_fn": collate_fn_multisource}
    if DEVICE.type == 'cuda':
        loader_args.update({"num_workers": min(4, os.cpu_count() or 1), "pin_memory": True, "persistent_workers": True, "prefetch_factor": 2})

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)
    dev_loader = DataLoader(dev_dataset, **loader_args)
    test_loader = DataLoader(test_dataset, **loader_args)

    print("\n[3/5] Initializing MultiSource Attention CRF model...")
    model = MultiSource_Attention_CRF(len(word_to_ix), len(char_to_ix), tag_to_ix, HP).to(DEVICE)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  - Trainable parameters: {num_params:,}")

    print("\n[4/5] Loading pre-trained Word2Vec (FastText) embeddings...")
    try:
        fasttext = WordEmbeddings('id')
        pretrained_weights = model.w2v_embedding.weight.data
        found_words = 0
        for word, i in word_to_ix.items():
            if word in ['<PAD>', '<UNK>']:
                continue
            dummy_sentence = Sentence(word)
            fasttext.embed(dummy_sentence)
            if dummy_sentence[0].embedding.nelement() > 0:
                pretrained_weights[i] = dummy_sentence[0].embedding
                found_words += 1
        model.w2v_embedding.weight.data.copy_(pretrained_weights)
        print(f"  - Loaded embeddings for {found_words}/{len(word_to_ix)} words")
    except Exception as e:
        print(f"  - Warning: Failed to load embeddings: {e}")
        print("  - Continuing with random initialization...")

    print("\n[5/5] Setting up optimizer...")
    optimizer = optim.AdamW(model.parameters(), lr=HP["learning_rate"])

    num_training_steps = len(train_loader) * HP["epochs"]
    num_warmup_steps = int(0.1 * num_training_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
    )

    print(f"  - Learning rate: {HP['learning_rate']}")
    print(f"  - Total training steps: {num_training_steps}")
    print(f"  - Warmup steps: {num_warmup_steps}")

    print("\n[6/6] Starting training...")
    print("-" * 60)
    best_dev_f1 = -1
    patience_counter = 0
    total_training_time = 0
    training_history = []

    for epoch in range(HP["epochs"]):
        epoch_start_time = time.time()
        model.train()
        total_loss = 0
        for _, words, chars, tags, word_mask, bert_indices, bert_mask, word_maps in train_loader:
            words, chars, tags, word_mask = words.to(DEVICE), chars.to(DEVICE), tags.to(DEVICE), word_mask.to(DEVICE)
            bert_indices, bert_mask = bert_indices.to(DEVICE), bert_mask.to(DEVICE)
            
            optimizer.zero_grad()
            loss = model(words, chars, tags, word_mask, bert_indices, bert_mask, word_maps)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        
        dev_report_dict, _ = evaluate(model, dev_loader, ix_to_tag, DEVICE)
        dev_f1 = dev_report_dict['micro avg']['f1-score']

        dev_precision = dev_report_dict['micro avg']['precision']
        dev_recall = dev_report_dict['micro avg']['recall']
        
        epoch_time = time.time() - epoch_start_time
        total_training_time += epoch_time
        
        print(f"Epoch {epoch+1:3d}/{HP['epochs']} | "
              f"Loss: {avg_loss:.4f} | "
              f"Dev P/R/F1: {dev_precision:.4f}/{dev_recall:.4f}/{dev_f1:.4f} | "
              f"Time: {epoch_time:.1f}s")
        
        history_epoch = {
            'epoch': epoch + 1,
            'train_loss': avg_loss,
            'dev_f1': dev_f1,
            'dev_precision': dev_precision,
            'dev_recall': dev_recall
        }
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

    print("-" * 60)
    print("Training complete!")
    
    pd.DataFrame(training_history).to_csv(HISTORY_PATH, index=False)
    print(f"Training history saved to {HISTORY_PATH}")

    print("\nLoading best model for final evaluation...")
    model.load_state_dict(torch.load(BEST_MODEL_PATH, weights_only=True))
    
    final_report_dict, final_eval_time = evaluate(model, test_loader, ix_to_tag, DEVICE, save_path=OUTPUT_DIR)
    
    print("\n" + "=" * 60)
    print("FINAL TEST SET RESULTS")
    print("=" * 60)
    with open(REPORT_PATH, 'r', encoding='utf-8') as f:
        print(f.read())
    
    summary = {
        "model_name": "MultiSource Attention CRF",
        "best_model_path": str(BEST_MODEL_PATH),
        "device": str(DEVICE),
        "timestamps": {
            "training_start": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)),
            "training_end": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        },
        "training_duration_seconds": total_training_time,
        "evaluation_time_seconds": final_eval_time,
        "best_dev_f1": best_dev_f1,
        "dataset_info": {
            "training_file": str(TRAIN_FILE),
            "development_file": str(DEV_FILE),
            "test_file": str(TEST_FILE),
            "train_samples": len(train_data),
            "dev_samples": len(dev_data),
            "test_samples": len(test_data),
            "vocab_size": len(word_to_ix),
            "char_vocab_size": len(char_to_ix),
            "tagset_size": len(tag_to_ix)
        },
        "model_info": {
            "num_trainable_params": num_params
        },
        "hyperparameters": HP,
        "final_test_metrics": final_report_dict
    }
    with open(SUMMARY_PATH, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4)
        
    print(f"\nAll experiment results have been saved to directory: {OUTPUT_DIR}")
    print(f"Final summary report saved to {SUMMARY_PATH}")
    
    test_f1 = final_report_dict['micro avg']['f1-score']
    print(f"\n{'=' * 60}")
    print(f"MultiSource Attention CRF Test F1-Score: {test_f1:.4f} ({test_f1*100:.2f}%)")
    print(f"{'=' * 60}")

if __name__ == '__main__':
    main()
