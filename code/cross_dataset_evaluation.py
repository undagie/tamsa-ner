import json
import torch
import numpy as np
from pathlib import Path
from collections import Counter
import time
import pandas as pd
from itertools import product
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from seqeval.metrics import classification_report as seqeval_classification_report
from seqeval.scheme import IOB2

from train_bilstm import BiLSTM_CRF
from train_bilstm_w2v import BiLSTM_W2V_CRF
from train_bilstm_w2v_cnn import BiLSTM_W2V_CNN_CRF
from train_indobert import IndoBERT_CRF
from train_indobert_bilstm import IndoBERT_BiLSTM_CRF
from train_mbert_bilstm import Transformer_BiLSTM_CRF as mBERT_BiLSTM_CRF
from train_xlm_roberta_bilstm import Transformer_BiLSTM_CRF as XLMRoBERTa_BiLSTM_CRF
from train_attention_fusion import MultiSource_Attention_CRF as MultiSourceAttention_CRF

from transformers import AutoTokenizer
from flair.embeddings import WordEmbeddings
from flair.data import Sentence as FlairSentence
from torch.utils.data import Dataset, DataLoader

DATASETS = ['idner2k', 'nerugm', 'nerui']
MODELS = ['bilstm', 'bilstm_w2v', 'bilstm_w2v_cnn', 'indobert', 'indobert_bilstm',
          'mbert_bilstm', 'xlm_roberta_bilstm', 'attention_fusion']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
BATCH_SIZE = 16

# Output directory
OUTPUT_DIR = Path('./outputs/cross_dataset_evaluation')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_bio_file(file_path):
    """Load BIO-tagged file."""
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

def analyze_dataset_statistics(dataset_name, sentences):
    """Analyze dataset statistics."""
    total_tokens = sum(len(tokens) for tokens, _ in sentences)
    total_sentences = len(sentences)
    entity_counts = Counter()
    for _, tags in sentences:
        for tag in tags:
            if tag != 'O':
                entity_type = tag.split('-')[1] if '-' in tag else tag
                entity_counts[entity_type] += 1
    sentence_lengths = [len(tokens) for tokens, _ in sentences]
    
    return {
        'dataset': dataset_name,
        'total_sentences': total_sentences,
        'total_tokens': total_tokens,
        'avg_sentence_length': np.mean(sentence_lengths),
        'std_sentence_length': np.std(sentence_lengths),
        'min_sentence_length': min(sentence_lengths),
        'max_sentence_length': max(sentence_lengths),
        'entity_distribution': dict(entity_counts),
        'total_entities': sum(entity_counts.values()),
        'entity_density': sum(entity_counts.values()) / total_tokens
    }

class UniversalNERDataset(Dataset):
    """Universal dataset that can handle different model types."""
    def __init__(self, sentences, word_to_ix, tag_to_ix, char_to_ix=None, tokenizer=None, model_type='bilstm'):
        self.sentences = sentences
        self.word_to_ix = word_to_ix
        self.tag_to_ix = tag_to_ix
        self.char_to_ix = char_to_ix
        self.tokenizer = tokenizer
        self.model_type = model_type
        
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        tokens, tags = self.sentences[idx]
        
        # Basic word indices (for all models)
        word_indices = [self.word_to_ix.get(t.lower(), 1) for t in tokens]
        tag_indices = [self.tag_to_ix.get(t, self.tag_to_ix.get('O', 0)) for t in tags]
        
        result = {
            'tokens': tokens,
            'word_indices': word_indices,
            'tag_indices': tag_indices
        }
        
        # Add character indices if needed
        if self.char_to_ix is not None:
            char_indices = [[self.char_to_ix.get(c, 1) for c in token] for token in tokens]
            result['char_indices'] = char_indices
        
        # Add BERT indices if needed
        if self.tokenizer is not None:
            bert_subwords = []
            word_to_subword_map = []
            for token in tokens:
                subws = self.tokenizer.tokenize(token)
                word_to_subword_map.append(len(bert_subwords))
                bert_subwords.extend(subws)
            bert_indices = self.tokenizer.convert_tokens_to_ids(bert_subwords)
            result['bert_indices'] = bert_indices
            result['word_to_subword_map'] = word_to_subword_map
            
        return result

def create_universal_collate_fn(model_type):
    """Create appropriate collate function based on model type."""
    def collate_fn(batch):
        # Extract common fields
        tokens = [item['tokens'] for item in batch]
        word_indices = [item['word_indices'] for item in batch]
        tag_indices = [item['tag_indices'] for item in batch]
        
        # Pad word indices
        max_word_len = max(len(w) for w in word_indices)
        padded_words = torch.zeros(len(word_indices), max_word_len, dtype=torch.long)
        padded_tags = torch.zeros(len(tag_indices), max_word_len, dtype=torch.long)
        word_mask = torch.zeros(len(word_indices), max_word_len, dtype=torch.bool)
        
        for i, (w, t) in enumerate(zip(word_indices, tag_indices)):
            l = len(w)
            padded_words[i, :l] = torch.tensor(w)
            padded_tags[i, :l] = torch.tensor(t)
            word_mask[i, :l] = True
        
        result = {
            'tokens': tokens,
            'words': padded_words,
            'tags': padded_tags,
            'word_mask': word_mask
        }
        
        # Handle character indices
        if 'char_indices' in batch[0]:
            char_indices = [item['char_indices'] for item in batch]
            max_char_len = max(max(len(c) for c in chars) for chars in char_indices)
            padded_chars = torch.zeros(len(char_indices), max_word_len, max_char_len, dtype=torch.long)
            
            for i, chars in enumerate(char_indices):
                for j, char_seq in enumerate(chars):
                    if j < max_word_len:
                        char_len = len(char_seq)
                        padded_chars[i, j, :char_len] = torch.tensor(char_seq)
            
            result['chars'] = padded_chars
        
        # Handle BERT indices
        if 'bert_indices' in batch[0]:
            bert_indices = [item['bert_indices'] for item in batch]
            word_maps = [item['word_to_subword_map'] for item in batch]
            
            max_sub_len = max(len(s) for s in bert_indices)
            padded_bert_indices = torch.zeros(len(bert_indices), max_sub_len, dtype=torch.long)
            bert_mask = torch.zeros(len(bert_indices), max_sub_len, dtype=torch.long)
            
            for i, s in enumerate(bert_indices):
                l = len(s)
                padded_bert_indices[i, :l] = torch.tensor(s)
                bert_mask[i, :l] = 1
            
            result['bert_indices'] = padded_bert_indices
            result['bert_mask'] = bert_mask
            result['word_maps'] = word_maps
        
        return result
    
    return collate_fn

def load_model_and_vocab(model_name, tag_to_ix):
    """Load pre-trained model and its vocabulary.
    
    Note: Uses the tagset from the trained model (vocab_data['tag_to_ix']), 
    not the unified tagset, to avoid size mismatches.
    """
    model_dir = Path(f'./outputs/experiment_{model_name}')
    
    with open(model_dir / 'hyperparameters.json', 'r') as f:
        hp = json.load(f)
    
    with open(model_dir / 'vocab.json', 'r') as f:
        vocab_data = json.load(f)
    
    model_tag_to_ix = vocab_data.get('tag_to_ix', tag_to_ix)
    
    tokenizer = None
    if model_name in ['indobert', 'indobert_bilstm', 'mbert_bilstm', 'xlm_roberta_bilstm', 'attention_fusion']:
        tokenizer = AutoTokenizer.from_pretrained(hp['transformer_model'])
    
    if model_name == 'bilstm':
        # BiLSTM_CRF signature: (vocab_size, tag_to_ix, embedding_dim, hidden_dim, lstm_layers, dropout)
        vocab_size = len(vocab_data.get('word_to_ix', {'<PAD>': 0, '<UNK>': 1}))
        model = BiLSTM_CRF(
            vocab_size, 
            model_tag_to_ix, 
            hp.get("embedding_dim", 300),
            hp.get("lstm_hidden_dim", 256),
            hp.get("lstm_layers", 1),
            hp.get("dropout", 0.3)
        )
    elif model_name == 'bilstm_w2v':
        # Load Word2Vec embeddings for the model
        from flair.embeddings import WordEmbeddings
        word_embeddings = None
        try:
            word_embeddings = WordEmbeddings('id')
        except:
            pass
        model = BiLSTM_W2V_CRF(model_tag_to_ix, hp, word_to_ix=vocab_data.get('word_to_ix'), word_embeddings=word_embeddings)
    elif model_name == 'bilstm_w2v_cnn':
        # BiLSTM_W2V_CNN_CRF signature: (w2v_vocab_size, char_vocab_size, tag_to_ix, hp)
        w2v_vocab_size = len(vocab_data.get('word_to_ix', {'<PAD>': 0, '<UNK>': 1}))
        char_vocab_size = len(vocab_data.get('char_to_ix', {'<PAD>': 0, '<UNK>': 1}))
        model = BiLSTM_W2V_CNN_CRF(w2v_vocab_size, char_vocab_size, model_tag_to_ix, hp)
    elif model_name == 'indobert':
        model = IndoBERT_CRF(model_tag_to_ix, hp)
    elif model_name == 'indobert_bilstm':
        model = IndoBERT_BiLSTM_CRF(model_tag_to_ix, hp)
    elif model_name == 'mbert_bilstm':
        model = mBERT_BiLSTM_CRF(model_tag_to_ix, hp)
    elif model_name == 'xlm_roberta_bilstm':
        model = XLMRoBERTa_BiLSTM_CRF(model_tag_to_ix, hp)
    elif model_name == 'attention_fusion':
        # MultiSourceAttention_CRF signature: (word_vocab_size, char_vocab_size, tag_to_ix, hp)
        word_vocab_size = len(vocab_data.get('word_to_ix', {'<PAD>': 0, '<UNK>': 1}))
        char_vocab_size = len(vocab_data.get('char_to_ix', {'<PAD>': 0, '<UNK>': 1}))
        model = MultiSourceAttention_CRF(
            word_vocab_size, 
            char_vocab_size, 
            model_tag_to_ix, 
            hp
        )
    
    # Load model weights with strict=False to handle missing keys (e.g., word_embeddings)
    model_path = model_dir / f'{model_name.replace("_", "-")}-crf-best.pt'
    if model_name == 'attention_fusion':
        model_path = model_dir / 'attention-fusion-crf-best.pt'
    
    # Move model to device first (before loading weights) to ensure device consistency
    model = model.to(DEVICE)
    
    try:
        state_dict = torch.load(model_path, map_location=DEVICE, weights_only=True)
        # Remove keys that don't exist in the model (e.g., word_embeddings.embedding.weight)
        model_state_dict = model.state_dict()
        filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict and model_state_dict[k].shape == v.shape}
        model.load_state_dict(filtered_state_dict, strict=False)
    except Exception as e:
        print(f"Warning: Error loading state_dict: {e}")
        print("Attempting to load with strict=False...")
        try:
            state_dict = torch.load(model_path, map_location=DEVICE, weights_only=True)
            model.load_state_dict(state_dict, strict=False)
        except Exception as e2:
            print(f"Error loading state_dict even with strict=False: {e2}")
            raise
    
    model.eval()
    
    # Return both the model tagset and unified tagset for mapping
    return model, vocab_data, tokenizer, hp, model_tag_to_ix

def evaluate_model_on_dataset(model, loader, model_ix_to_tag, unified_ix_to_tag, model_type):
    """Evaluate model on a dataset.
    
    Args:
        model: The trained model
        loader: DataLoader for the test set
        model_ix_to_tag: Tag mapping from the model's tagset (used during training)
        unified_ix_to_tag: Tag mapping from unified tagset (for evaluation)
        model_type: Type of model
    """
    model.eval()
    all_preds, all_true, all_tokens = [], [], []
    
    with torch.no_grad():
        for batch in loader:
            # Move tensors to device
            for key in ['words', 'tags', 'word_mask', 'chars', 'bert_indices', 'bert_mask']:
                if key in batch and isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(DEVICE, non_blocking=True)
            
            # Get predictions based on model type
            if model_type == 'bilstm':
                preds = model.predict(batch['words'], batch['word_mask'])
            elif model_type == 'bilstm_w2v':
                # Create Word2Vec features if word_embeddings is available
                if hasattr(model, 'word_embeddings') and model.word_embeddings is not None:
                    w2v_feats = []
                    for tokens in batch['tokens']:
                        sent = FlairSentence(' '.join(tokens))
                        model.word_embeddings.embed(sent)
                        feats = [token.embedding.cpu().numpy() for token in sent]
                        w2v_feats.append(feats)
                    preds = model.predict(batch['words'], batch['word_mask'], w2v_feats)
                else:
                    # Fallback: use embedding layer only (model was trained with Word2Vec weights in embedding)
                    preds = model.predict(batch['words'], batch['word_mask'], w2v_feats=None)
            elif model_type == 'bilstm_w2v_cnn':
                # BiLSTM_W2V_CNN_CRF uses embedding layer, not Flair WordEmbeddings
                # The model.predict() signature is (words, mask, chars), not (words, chars, mask, w2v_feats)
                preds = model.predict(batch['words'], batch['word_mask'], batch['chars'])
            elif model_type in ['indobert', 'indobert_bilstm', 'mbert_bilstm', 'xlm_roberta_bilstm']:
                preds = model.predict(batch['bert_indices'], batch['bert_mask'], 
                                    batch['word_mask'], batch['word_maps'])
            elif model_type == 'attention_fusion':
                preds = model.predict(batch['words'], batch['chars'], batch['word_mask'],
                                    batch['bert_indices'], batch['bert_mask'], batch['word_maps'])
            
            # Convert predictions to tags using model's tagset
            for i in range(len(preds)):
                seq_len = int(batch['word_mask'][i].sum())
                # True tags use unified tagset (from test data)
                true_tags = [unified_ix_to_tag.get(batch['tags'][i][j].item(), 'O') for j in range(seq_len)]
                # Predicted tags use model's tagset, then map to unified if needed
                pred_tags = [model_ix_to_tag.get(p, 'O') for p in preds[i][:seq_len]]
                
                all_true.append(true_tags)
                all_preds.append(pred_tags)
                all_tokens.extend(batch['tokens'][i][:seq_len])
    
    # Calculate metrics
    report_dict = seqeval_classification_report(all_true, all_preds, scheme=IOB2, 
                                               digits=4, output_dict=True, zero_division=0)
    
    return report_dict, all_true, all_preds, all_tokens

def main():
    """Run cross-dataset evaluation."""
    print("=== Cross-Dataset Evaluation for Indonesian NER ===\n")
    
    # First, analyze dataset statistics
    print("1. Analyzing dataset statistics...")
    dataset_stats = []
    
    for dataset in DATASETS:
        train_path = Path(f'./data/{dataset}/train_bio.txt')
        dev_path = Path(f'./data/{dataset}/dev_bio.txt')
        test_path = Path(f'./data/{dataset}/test_bio.txt')
        
        all_sentences = []
        for path in [train_path, dev_path, test_path]:
            all_sentences.extend(load_bio_file(path))
        
        stats = analyze_dataset_statistics(dataset, all_sentences)
        dataset_stats.append(stats)
        
        print(f"\n{dataset.upper()} Statistics:")
        print(f"  Total sentences: {stats['total_sentences']:,}")
        print(f"  Total tokens: {stats['total_tokens']:,}")
        print(f"  Avg sentence length: {stats['avg_sentence_length']:.2f} ± {stats['std_sentence_length']:.2f}")
        print(f"  Entity density: {stats['entity_density']:.4f}")
        print(f"  Entity types: {stats['entity_distribution']}")
    
    # Save dataset statistics
    stats_df = pd.DataFrame(dataset_stats)
    stats_df.to_csv(OUTPUT_DIR / 'dataset_statistics.csv', index=False)
    
    # Create unified tag vocabulary from all datasets
    print("\n2. Creating unified tag vocabulary...")
    all_tags = set(['O'])
    for dataset in DATASETS:
        for split in ['train', 'dev', 'test']:
            path = Path(f'./data/{dataset}/{split}_bio.txt')
            sentences = load_bio_file(path)
            for _, tags in sentences:
                all_tags.update(tags)
    
    unified_tag_to_ix = {tag: i for i, tag in enumerate(sorted(all_tags))}
    unified_ix_to_tag = {i: tag for tag, i in unified_tag_to_ix.items()}
    
    print(f"Unified tag set: {sorted(all_tags)}")
    
    # Cross-dataset evaluation
    print("\n3. Starting cross-dataset evaluation...")
    results = []
    
    for train_dataset, test_dataset in product(DATASETS, DATASETS):
        print(f"\n--- Evaluating models trained on {train_dataset.upper()} on {test_dataset.upper()} test set ---")
        
        # Load test data
        test_path = Path(f'./data/{test_dataset}/test_bio.txt')
        test_sentences = load_bio_file(test_path)
        
        for model_name in MODELS:
            print(f"\nEvaluating {model_name}...")
            
            try:
                # Check if model exists for this training dataset
                model_dir = Path(f'./outputs/experiment_{model_name}')
                if not model_dir.exists():
                    print(f"Model {model_name} not found, skipping...")
                    continue
                
                # Load model and vocabularies
                # Note: model uses its own tagset from training, not unified tagset
                model, vocab_data, tokenizer, hp, model_tag_to_ix = load_model_and_vocab(model_name, unified_tag_to_ix)
                
                # Create mapping from model tagset to unified tagset
                model_ix_to_tag = vocab_data.get('ix_to_tag', {i: tag for tag, i in model_tag_to_ix.items()})
                # Ensure it's a dict with int keys
                if isinstance(model_ix_to_tag, dict) and all(isinstance(k, str) for k in model_ix_to_tag.keys()):
                    # Convert string keys to int if needed
                    model_ix_to_tag = {int(k): v for k, v in model_ix_to_tag.items()}
                
                # Prepare dataset using unified tagset for test data
                word_to_ix = vocab_data.get('word_to_ix', {'<PAD>': 0, '<UNK>': 1})
                char_to_ix = vocab_data.get('char_to_ix', None)
                
                test_dataset_obj = UniversalNERDataset(
                    test_sentences, word_to_ix, unified_tag_to_ix, 
                    char_to_ix, tokenizer, model_name
                )
                
                loader_args = {
                    "batch_size": BATCH_SIZE, 
                    "collate_fn": create_universal_collate_fn(model_name)
                }
                if DEVICE.type == 'cuda':
                    import os
                    loader_args.update({
                        "num_workers": min(4, os.cpu_count() or 1),
                        "pin_memory": True,
                        "persistent_workers": True,
                        "prefetch_factor": 2
                    })
                test_loader = DataLoader(test_dataset_obj, **loader_args)
                
                # Evaluate
                start_time = time.time()
                report_dict, _, _, _ = evaluate_model_on_dataset(
                    model, test_loader, model_ix_to_tag, unified_ix_to_tag, model_name
                )
                eval_time = time.time() - start_time
                
                # Store results
                result = {
                    'train_dataset': train_dataset,
                    'test_dataset': test_dataset,
                    'model': model_name,
                    'weighted_f1': report_dict.get('weighted avg', {}).get('f1-score', 0),
                    'macro_f1': report_dict.get('macro avg', {}).get('f1-score', 0),
                    'micro_f1': report_dict.get('micro avg', {}).get('f1-score', 0),
                    'eval_time': eval_time
                }
                
                # Add per-entity scores
                for entity in ['PER', 'LOC', 'ORG', 'MISC']:
                    if entity in report_dict:
                        result[f'{entity}_f1'] = report_dict[entity]['f1-score']
                        result[f'{entity}_precision'] = report_dict[entity]['precision']
                        result[f'{entity}_recall'] = report_dict[entity]['recall']
                
                results.append(result)
                
                print(f"  Weighted F1: {result['weighted_f1']:.4f}")
                print(f"  Macro F1: {result['macro_f1']:.4f}")
                print(f"  Evaluation time: {eval_time:.2f}s")
                
            except Exception as e:
                print(f"Error evaluating {model_name}: {str(e)}")
                continue
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / 'cross_dataset_results.csv', index=False)
    
    # Create visualization matrices
    print("\n4. Creating visualization matrices...")
    
    for metric in ['weighted_f1', 'macro_f1']:
        plt.figure(figsize=(12, 10))
        
        for model_name in MODELS:
            model_results = results_df[results_df['model'] == model_name]
            if len(model_results) == 0:
                continue
            
            # Create matrix
            matrix = np.zeros((len(DATASETS), len(DATASETS)))
            for i, train_ds in enumerate(DATASETS):
                for j, test_ds in enumerate(DATASETS):
                    value = model_results[
                        (model_results['train_dataset'] == train_ds) & 
                        (model_results['test_dataset'] == test_ds)
                    ][metric].values
                    
                    if len(value) > 0:
                        matrix[i, j] = value[0]
            
            # Plot heatmap
            plt.subplot(3, 3, MODELS.index(model_name) + 1)
            sns.heatmap(matrix, annot=True, fmt='.3f', cmap='YlOrRd',
                       xticklabels=DATASETS, yticklabels=DATASETS,
                       vmin=0, vmax=1)
            plt.title(f'{model_name}')
            plt.xlabel('Test Dataset')
            plt.ylabel('Train Dataset')
        
        plt.suptitle(f'Cross-Dataset {metric.replace("_", " ").title()} Scores', fontsize=16)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f'cross_dataset_{metric}_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Calculate generalization scores
    print("\n5. Calculating generalization scores...")
    generalization_scores = []
    
    for model_name in MODELS:
        model_results = results_df[results_df['model'] == model_name]
        if len(model_results) == 0:
            continue
        
        # In-domain performance (average of diagonal)
        in_domain_scores = []
        for dataset in DATASETS:
            score = model_results[
                (model_results['train_dataset'] == dataset) & 
                (model_results['test_dataset'] == dataset)
            ]['weighted_f1'].values
            if len(score) > 0:
                in_domain_scores.append(score[0])
        
        # Cross-domain performance (average of off-diagonal)
        cross_domain_scores = []
        for train_ds, test_ds in product(DATASETS, DATASETS):
            if train_ds != test_ds:
                score = model_results[
                    (model_results['train_dataset'] == train_ds) & 
                    (model_results['test_dataset'] == test_ds)
                ]['weighted_f1'].values
                if len(score) > 0:
                    cross_domain_scores.append(score[0])
        
        if in_domain_scores and cross_domain_scores:
            generalization_scores.append({
                'model': model_name,
                'avg_in_domain_f1': np.mean(in_domain_scores),
                'avg_cross_domain_f1': np.mean(cross_domain_scores),
                'generalization_gap': np.mean(in_domain_scores) - np.mean(cross_domain_scores),
                'generalization_ratio': np.mean(cross_domain_scores) / np.mean(in_domain_scores)
            })
    
    gen_df = pd.DataFrame(generalization_scores)
    gen_df.to_csv(OUTPUT_DIR / 'generalization_scores.csv', index=False)
    
    # Plot generalization comparison
    plt.figure(figsize=(10, 6))
    x = np.arange(len(gen_df))
    width = 0.35
    
    plt.bar(x - width/2, gen_df['avg_in_domain_f1'], width, label='In-domain', alpha=0.8)
    plt.bar(x + width/2, gen_df['avg_cross_domain_f1'], width, label='Cross-domain', alpha=0.8)
    
    plt.xlabel('Model')
    plt.ylabel('Average F1 Score')
    plt.title('In-domain vs Cross-domain Performance')
    plt.xticks(x, gen_df['model'], rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'generalization_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nCross-dataset evaluation completed!")
    print(f"Results saved to: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
