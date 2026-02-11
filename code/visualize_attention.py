import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from transformers import AutoTokenizer, AutoModel

_ROOT = Path(__file__).resolve().parent.parent
from flair.embeddings import WordEmbeddings
import json
import sys
import pandas as pd
import warnings

# Suppress FutureWarnings from external libraries
warnings.filterwarnings('ignore', category=FutureWarning, module='transformers')
warnings.filterwarnings('ignore', category=FutureWarning, module='huggingface_hub')
warnings.filterwarnings('ignore', category=FutureWarning, message='.*_register_pytree_node.*')
warnings.filterwarnings('ignore', category=FutureWarning, message='.*resume_download.*')
warnings.filterwarnings('ignore', category=FutureWarning, message='.*weights_only.*')

OUTPUT_DIR = _ROOT / "outputs" / "experiment_attention_fusion"
VISUALIZATION_DIR = _ROOT / "outputs" / "attention_visualization"
VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)
BEST_MODEL_PATH = OUTPUT_DIR / "attention-fusion-crf-best.pt"
HP_PATH = OUTPUT_DIR / "hyperparameters.json"
VOCAB_PATH = OUTPUT_DIR / "vocab.json"
TEST_FILE = _ROOT / "data" / "idner2k" / "test_bio.txt"

# Must match train_attention_fusion.py.


class CharCNN(nn.Module):
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
    def __init__(self, word_vocab_size, char_vocab_size, tag_to_ix, hp):
        super(MultiSource_Attention_CRF, self).__init__()
        self.hp = hp
        self.w2v_embedding = nn.Embedding(word_vocab_size, self.hp['w2v_embedding_dim'], padding_idx=0)
        self.char_cnn = CharCNN(char_vocab_size, self.hp['char_embedding_dim'], self.hp['char_cnn_filters'], self.hp['char_cnn_kernel_size'], self.hp['dropout'])
        self.transformer = AutoModel.from_pretrained(self.hp["transformer_model"])
        
        self.projection_dim = self.hp['lstm_hidden_dim']
        self.w2v_proj = nn.Linear(self.hp['w2v_embedding_dim'], self.projection_dim)
        self.char_proj = nn.Linear(self.hp['char_cnn_filters'], self.projection_dim)
        self.trans_proj = nn.Linear(self.hp['transformer_dim'], self.projection_dim)

        self.fusion = AttentionFusion(self.projection_dim, self.projection_dim // 2)

    def get_attention_weights(self, words, chars, bert_indices, bert_mask, word_maps):
        # 1. Get raw embeddings
        w2v_features = self.w2v_embedding(words)
        char_cnn_features = self.char_cnn(chars)

        transformer_outputs = self.transformer(bert_indices, attention_mask=bert_mask)
        transformer_hidden_states = transformer_outputs.last_hidden_state

        batch_size, seq_len, _ = w2v_features.shape
        device = w2v_features.device
        transformer_features = torch.zeros(batch_size, seq_len, self.hp["transformer_dim"], device=device)
        for i, mapping in enumerate(word_maps): # i is batch index, mapping is the word_map for that item
            if not mapping: continue
            for j, subword_indices in enumerate(mapping):
                 if j < seq_len:
                    if subword_indices: # Check if the list is not empty
                        # Use the embedding of the first subword to represent the whole word
                        first_subword_idx = subword_indices[0]
                        transformer_features[i, j, :] = transformer_hidden_states[i, first_subword_idx, :]

        # 2. Project
        w2v_proj = torch.relu(self.w2v_proj(w2v_features))
        char_proj = torch.relu(self.char_proj(char_cnn_features))
        trans_proj = torch.relu(self.trans_proj(transformer_features))

        # 3. Fuse & Get Weights
        _, weights = self.fusion([w2v_proj, char_proj, trans_proj])
        return weights

def plot_attention_heatmap(tokens, weights, filename="attention_heatmap.png"):
    """
    tokens: list of strings
    weights: numpy array of shape (seq_len, 3) -> [Word, Char, Trans]
    """
    sources = ['Word2Vec', 'CharCNN', 'Transformer']
    weights_t = weights.T 
    
    plt.figure(figsize=(max(10, len(tokens) * 0.5), 4))
    sns.set_style("whitegrid")
    sns.set(font_scale=1.2)
    
    ax = sns.heatmap(weights_t, annot=True, fmt=".2f", cmap="YlOrRd", 
                     xticklabels=tokens, yticklabels=sources,
                     cbar_kws={'label': 'Attention Weight'},
                     linewidths=0.5, linecolor='gray')
    
    plt.title("Visualization of Token-Aware Attention Weights on Sample Sentence", 
              fontsize=14, pad=20, weight='bold')
    plt.xlabel("Tokens", fontsize=12, weight='bold')
    plt.ylabel("Input Sources", fontsize=12, weight='bold')
    plt.xticks(rotation=0, ha='center')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    save_path = VISUALIZATION_DIR / filename
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved heatmap to {save_path}")

def prepare_input(text, tokenizer, word_to_ix, char_to_ix, device):
    tokens = text.split()
    
    # Word indices
    word_indices = [word_to_ix.get(t.lower(), word_to_ix['<UNK>']) for t in tokens]
    word_tensor = torch.tensor([word_indices]).to(device, non_blocking=True)
    
    # Char indices
    max_word_len = max(len(t) for t in tokens)
    char_indices = []
    for t in tokens:
        chars = [char_to_ix.get(c, char_to_ix['<UNK>']) for c in t]
        chars += [0] * (max_word_len - len(chars))
        char_indices.append(chars)
    char_tensor = torch.tensor([char_indices]).to(device, non_blocking=True)
    
    # BERT indices
    bert_subwords = []
    word_maps = []
    for t in tokens:
        subws = tokenizer.tokenize(t)
        word_maps.append(list(range(len(bert_subwords), len(bert_subwords) + len(subws))))
        bert_subwords.extend(subws)
    
    bert_indices = tokenizer.convert_tokens_to_ids(bert_subwords)
    bert_tensor = torch.tensor([bert_indices]).to(device, non_blocking=True)
    bert_mask = torch.ones_like(bert_tensor).to(device, non_blocking=True)
    
    return tokens, word_tensor, char_tensor, bert_tensor, bert_mask, [word_maps]

def main():
    print("Starting Attention Visualization (REAL DATA ONLY)...")
    
    if not BEST_MODEL_PATH.exists():
        print(f"CRITICAL ERROR: Model file not found at {BEST_MODEL_PATH}")
        print("Cannot proceed without trained model.")
        sys.exit(1)
        
    try:
        print("Loading configuration...")
        with open(HP_PATH, 'r') as f: hp = json.load(f)
        with open(VOCAB_PATH, 'r') as f: vocab = json.load(f)
        
        word_to_ix = vocab['word_to_ix']
        char_to_ix = vocab['char_to_ix']
        tag_to_ix = vocab['tag_to_ix']

        # GPU setup - ensure we use device 0
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # GPU optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False  # For performance
        print(f"Using device: {device}")
        
        print("Initializing model architecture...")
        model = MultiSource_Attention_CRF(len(word_to_ix), len(char_to_ix), tag_to_ix, hp).to(device)
        
        print(f"Loading weights from {BEST_MODEL_PATH}...")
        # Load the saved state dictionary
        state_dict = torch.load(BEST_MODEL_PATH, map_location=device, weights_only=True)
        
        # Get the current model's state dictionary
        model_state_dict = model.state_dict()
        
        # Filter the loaded state_dict to include only keys that are in the current model
        filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict and model_state_dict[k].shape == v.shape}
        
        # Load the filtered state dictionary
        model.load_state_dict(filtered_state_dict, strict=False)
        
        model.eval()
        print("Model loaded successfully with filtered weights.")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(hp["transformer_model"])
        
    except Exception as e:
        print(f"ERROR loading model: {e}")
        sys.exit(1)

    # 1. Qualitative Analysis (Heatmaps)
    print("\nGenerating Heatmaps for Test Sentences from Dataset...")
    
    # Load sentences from test dataset
    def load_sentences_from_dataset(file_path, limit=500):
        """Load sentences from BIO format file"""
        sentences = []
        with open(file_path, 'r', encoding='utf-8') as f:
            current_tokens = []
            for line in f:
                line = line.strip()
                if not line:
                    if current_tokens:
                        sentences.append(" ".join(current_tokens))
                        current_tokens = []
                        if len(sentences) >= limit:
                            break
                else:
                    parts = line.split('\t')
                    if len(parts) >= 1:
                        current_tokens.append(parts[0])
            if current_tokens:
                sentences.append(" ".join(current_tokens))
        return sentences
    
    # Find suitable sentences from dataset (EXCLUDE presiden)
    test_sentences = load_sentences_from_dataset(TEST_FILE, limit=500)
    
    # Load sentences with tags to check for entities
    def load_sentences_with_tags(file_path, limit=500):
        """Load sentences with tags from BIO format file"""
        sentences = []
        with open(file_path, 'r', encoding='utf-8') as f:
            current_tokens = []
            current_tags = []
            for line in f:
                line = line.strip()
                if not line:
                    if current_tokens:
                        sentences.append((current_tokens, current_tags))
                        current_tokens, current_tags = [], []
                        if len(sentences) >= limit:
                            break
                else:
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        current_tokens.append(parts[0])
                        current_tags.append(parts[1])
            if current_tokens:
                sentences.append((current_tokens, current_tags))
        return sentences
    
    sentences_with_tags = load_sentences_with_tags(TEST_FILE, limit=500)
    
    # Select sentences WITHOUT presiden but with multiple entity types
    selected_sentences = []
    for tokens, tags in sentences_with_tags:
        text = " ".join(tokens)
        sent_lower = text.lower()
        
        # EXCLUDE sentences with presiden
        if 'presiden' in sent_lower:
            continue
        
        # Check for entities
        has_person = any('PER' in tag for tag in tags)
        has_location = any('LOC' in tag for tag in tags)
        has_org = any('ORG' in tag for tag in tags)
        
        # Count entities
        num_entities = sum(1 for tag in tags if tag != 'O')
        entity_types = sum([has_person, has_location, has_org])
        
        tokens_count = len(tokens)
        
        # Prefer sentences with multiple entity types (2-3 types) and reasonable length (8-18 tokens)
        if entity_types >= 2 and 8 <= tokens_count <= 18 and 2 <= num_entities <= 6:
            selected_sentences.append(text)
            if len(selected_sentences) >= 3:
                break
    
    # If no perfect match, use sentences with at least one entity type
    if not selected_sentences:
        for tokens, tags in sentences_with_tags:
            text = " ".join(tokens)
            sent_lower = text.lower()
            
            # EXCLUDE sentences with presiden
            if 'presiden' in sent_lower:
                continue
            
            has_person = any('PER' in tag for tag in tags)
            has_location = any('LOC' in tag for tag in tags)
            has_org = any('ORG' in tag for tag in tags)
            entity_types = sum([has_person, has_location, has_org])
            num_entities = sum(1 for tag in tags if tag != 'O')
            tokens_count = len(tokens)
            
            if entity_types >= 1 and 6 <= tokens_count <= 20 and num_entities >= 1:
                selected_sentences.append(text)
                if len(selected_sentences) >= 3:
                    break
    
    # Fallback to first sentence from dataset without presiden
    if not selected_sentences:
        for sent in test_sentences:
            if 'presiden' not in sent.lower():
                if 5 <= len(sent.split()) <= 25:
                    selected_sentences.append(sent)
                    if len(selected_sentences) >= 1:
                        break
    
    print(f"Selected {len(selected_sentences)} sentence(s) from dataset:")
    for i, sent in enumerate(selected_sentences):
        print(f"  {i+1}. {sent[:100]}...")
    
    for i, text in enumerate(selected_sentences):
        try:
            tokens, words, chars, bert_ids, bert_mask, word_maps = prepare_input(text, tokenizer, word_to_ix, char_to_ix, device)
            
            with torch.no_grad():
                weights = model.get_attention_weights(words, chars, bert_ids, bert_mask, word_maps)
                # weights shape: (1, seq_len, 3)
                weights_np = weights[0].cpu().numpy()
                
            if i == 0:
                # Save first sentence (from dataset) with specific name for paper
                plot_attention_heatmap(tokens, weights_np, "attention_weights_example.png")
                print(f"\nSaved paper example visualization: {text[:80]}...")
            plot_attention_heatmap(tokens, weights_np, f"heatmap_real_{i+1}.png")
        except Exception as e:
            print(f"Error processing sentence '{text[:50]}...': {e}")

    # 2. Quantitative Analysis (Aggregate)
    print("\nGenerating Quantitative Analysis from Test Set...")
    # Note: For full quantitative analysis, we should iterate over the test set.
    # Here we will process a subset of the test file to generate the distribution plot.
    
    if not TEST_FILE.exists():
        print("Test file not found, skipping quantitative analysis.")
        return

    # Simple loader for test file
    def load_sentences(path, limit=200):
        sents = []
        with open(path, 'r', encoding='utf-8') as f:
            curr_tokens = []
            curr_tags = []
            for line in f:
                line = line.strip()
                if not line:
                    if curr_tokens:
                        sents.append((curr_tokens, curr_tags))
                        curr_tokens, curr_tags = [], []
                    if len(sents) >= limit: break
                else:
                    parts = line.split()
                    if len(parts) == 2:
                        curr_tokens.append(parts[0])
                        curr_tags.append(parts[1])
        return sents

    test_data = load_sentences(TEST_FILE)
    print(f"Loaded {len(test_data)} sentences for analysis.")
    
    stats_data = []
    
    for tokens, tags in test_data:
        try:
            text = " ".join(tokens)
            _, words, chars, bert_ids, bert_mask, word_maps = prepare_input(text, tokenizer, word_to_ix, char_to_ix, device)
            
            with torch.no_grad():
                weights = model.get_attention_weights(words, chars, bert_ids, bert_mask, word_maps)
                weights_np = weights[0].cpu().numpy() # (seq_len, 3)
            
            # Aggregate by entity type
            for j, (token, tag) in enumerate(zip(tokens, tags)):
                if j >= len(weights_np): break
                
                # Simplify tag (B-PER -> PER)
                entity_type = tag.split('-')[1] if '-' in tag else tag
                if entity_type == 'O': entity_type = 'O (Non-Entity)'
                
                w_w2v = weights_np[j, 0]
                w_char = weights_np[j, 1]
                w_trans = weights_np[j, 2]
                
                stats_data.append({'Entity': entity_type, 'Source': 'Word2Vec', 'Weight': w_w2v})
                stats_data.append({'Entity': entity_type, 'Source': 'CharCNN', 'Weight': w_char})
                stats_data.append({'Entity': entity_type, 'Source': 'Transformer', 'Weight': w_trans})
                
        except Exception as e:
            continue
            
    if stats_data:
        df_stats = pd.DataFrame(stats_data)
        
        plt.figure(figsize=(10, 6))
        sns.set_style("whitegrid")
        ax = sns.barplot(data=df_stats, x='Entity', y='Weight', hue='Source', palette='viridis', errorbar='sd')
        
        plt.title("Average Attention Weight Distribution by Entity Type (REAL DATA)", fontsize=14, fontweight='bold')
        plt.ylabel("Average Attention Weight", fontsize=12)
        plt.xlabel("Entity Type", fontsize=12)
        plt.legend(title='Input Source')
        plt.ylim(0, 1.0)
        
        save_path = VISUALIZATION_DIR / "attention_distribution_real.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved real distribution plot to {save_path}")
    
    print("Real Data Visualization Complete.")

if __name__ == "__main__":
    main()
