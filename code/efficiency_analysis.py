import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import time
import psutil
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
import gc
import warnings

# Suppress FutureWarnings from external libraries
warnings.filterwarnings('ignore', category=FutureWarning, module='transformers')
warnings.filterwarnings('ignore', category=FutureWarning, module='huggingface_hub')
warnings.filterwarnings('ignore', category=FutureWarning, message='.*_register_pytree_node.*')
warnings.filterwarnings('ignore', category=FutureWarning, message='.*resume_download.*')

# Optional imports
try:
    import GPUtil
    HAS_GPUTIL = True
except ImportError:
    HAS_GPUTIL = False
    print("GPUtil not available - GPU monitoring disabled")

from train_bilstm import BiLSTM_CRF
from train_bilstm_w2v import BiLSTM_W2V_CRF
from train_bilstm_w2v_cnn import BiLSTM_W2V_CNN_CRF
from train_indobert_bilstm import IndoBERT_BiLSTM_CRF
from train_indobert import IndoBERT_CRF
from train_mbert_bilstm import Transformer_BiLSTM_CRF as mBERT_BiLSTM_CRF
from train_xlm_roberta_bilstm import Transformer_BiLSTM_CRF as XLMRoBERTa_BiLSTM_CRF
from train_attention_fusion import MultiSource_Attention_CRF as MultiSourceAttention_CRF

from transformers import AutoTokenizer
from flair.embeddings import WordEmbeddings
from flair.data import Sentence as FlairSentence

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    # GPU optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False  # For performance
OUTPUT_DIR = Path('./outputs/efficiency_analysis')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Test configurations
BATCH_SIZES = [1, 8, 16, 32]
SEQUENCE_LENGTHS = [10, 50, 100, 200]
REPETITIONS = 3  # Number of repetitions for timing

class EfficiencyAnalyzer:
    def __init__(self):
        self.results = []
        
    def measure_model_size(self, model_path: Path) -> Dict:
        """Measure model file size and parameter count."""
        # File size
        file_size_mb = model_path.stat().st_size / (1024 * 1024)
        
        # Load model to count parameters
        state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
        
        # Count parameters
        total_params = 0
        trainable_params = 0
        
        for name, param in state_dict.items():
            if isinstance(param, torch.Tensor):
                total_params += param.numel()
                # Assume all params are trainable (simplified)
                trainable_params += param.numel()
        
        return {
            'file_size_mb': file_size_mb,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'params_millions': total_params / 1e6
        }
    
    def measure_inference_speed(self, model, test_loader, model_type: str, 
                               batch_size: int, seq_length: int) -> Dict:
        """Measure inference speed and throughput."""
        model.eval()
        
        # Warm up
        with torch.no_grad():
            for _ in range(2):
                batch = next(iter(test_loader))
                self._run_inference(model, batch, model_type)
        
        # Actual measurement
        times = []
        num_samples = 0
        
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                if i >= 10:  # Test on 10 batches
                    break
                
                start_time = time.perf_counter()
                outputs = self._run_inference(model, batch, model_type)
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                end_time = time.perf_counter()
                
                times.append(end_time - start_time)
                num_samples += batch_size
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        return {
            'batch_size': batch_size,
            'sequence_length': seq_length,
            'avg_inference_time': avg_time,
            'std_inference_time': std_time,
            'samples_per_second': batch_size / avg_time if avg_time > 0 else 0,
            'tokens_per_second': batch_size * seq_length / avg_time if avg_time > 0 else 0
        }
    
    def measure_memory_usage(self, model, test_loader, model_type: str,
                            batch_size: int, seq_length: int) -> Dict:
        """Measure memory usage during inference."""
        model.eval()
        
        # Clear cache
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Get initial memory
        if torch.cuda.is_available():
            initial_gpu_memory = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
            max_gpu_memory = 0
        
        initial_cpu_memory = psutil.Process().memory_info().rss / (1024 ** 2)  # MB
        max_cpu_memory = initial_cpu_memory
        
        # Run inference and track memory
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                if i >= 5:  # Test on 5 batches
                    break
                
                _ = self._run_inference(model, batch, model_type)
                
                # Track memory
                current_cpu_memory = psutil.Process().memory_info().rss / (1024 ** 2)
                max_cpu_memory = max(max_cpu_memory, current_cpu_memory)
                
                if torch.cuda.is_available():
                    current_gpu_memory = torch.cuda.memory_allocated() / (1024 ** 2)
                    max_gpu_memory = max(max_gpu_memory, current_gpu_memory)
        
        result = {
            'batch_size': batch_size,
            'sequence_length': seq_length,
            'cpu_memory_mb': max_cpu_memory - initial_cpu_memory,
            'peak_cpu_memory_mb': max_cpu_memory
        }
        
        if torch.cuda.is_available():
            result['gpu_memory_mb'] = max_gpu_memory - initial_gpu_memory
            result['peak_gpu_memory_mb'] = max_gpu_memory
        
        return result
    
    def _run_inference(self, model, batch, model_type: str):
        """Run inference based on model type."""
        if isinstance(batch, dict):
            # Move tensors to device
            for key in ['words', 'chars', 'word_mask', 'bert_indices', 'bert_mask']:
                if key in batch and isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(DEVICE, non_blocking=True)
        else:
            # Legacy format (tuple)
            batch = [b.to(DEVICE, non_blocking=True) if isinstance(b, torch.Tensor) else b for b in batch]
        
        # Model-specific inference
        if model_type == 'bilstm':
            if isinstance(batch, dict):
                return model.predict(batch['words'], batch['word_mask'])
            else:
                return model.predict(batch[1], batch[3])  # words, mask
        
        elif model_type == 'bilstm_w2v':
            # Create dummy Word2Vec features
            if isinstance(batch, dict):
                batch_size, seq_len = batch['words'].shape
            else:
                batch_size, seq_len = batch[1].shape
            dummy_w2v = [np.random.randn(seq_len, 300) for _ in range(batch_size)]
            
            if isinstance(batch, dict):
                return model.predict(batch['words'], batch['word_mask'], dummy_w2v)
            else:
                return model.predict(batch[1], batch[3], dummy_w2v)
        
        elif model_type == 'bilstm_w2v_cnn':
            # BiLSTM_W2V_CNN_CRF uses embedding layer, not Flair WordEmbeddings
            # Signature: predict(words, mask, chars)
            if isinstance(batch, dict):
                return model.predict(batch['words'], batch['word_mask'], batch['chars'])
            else:
                return model.predict(batch[1], batch[3], batch[2])  # words, mask, chars
        
        elif model_type in ['indobert_bilstm', 'mbert_bilstm', 'xlm_roberta_bilstm']:
            if isinstance(batch, dict):
                return model.predict(batch['bert_indices'], batch['bert_mask'], 
                                   batch['word_mask'], batch['word_maps'])
            else:
                return model.predict(batch[5], batch[6], batch[4], batch[7])  # bert_indices, bert_mask, word_mask, word_maps
        
        elif model_type == 'attention_fusion':
            if isinstance(batch, dict):
                return model.predict(batch['words'], batch['chars'], batch['word_mask'],
                                   batch['bert_indices'], batch['bert_mask'], batch['word_maps'])
            else:
                return model.predict(batch[1], batch[2], batch[4], batch[5], batch[6], batch[7])
    
    def measure_training_efficiency(self, model_name: str) -> Dict:
        """Extract training efficiency metrics from saved logs."""
        result = {'model': model_name}
        
        # Load training history if available
        history_path = Path(f'./outputs/experiment_{model_name}/training_history.csv')
        if history_path.exists():
            history_df = pd.read_csv(history_path)
            
            # Training metrics
            result['total_epochs'] = len(history_df)
            result['best_epoch'] = history_df['dev_f1'].idxmax() + 1
            result['best_dev_f1'] = history_df['dev_f1'].max()
            result['final_train_loss'] = history_df['train_loss'].iloc[-1]
            
            # Convergence speed
            target_f1 = result['best_dev_f1'] * 0.95  # 95% of best
            convergence_epoch = history_df[history_df['dev_f1'] >= target_f1].index[0] + 1
            result['epochs_to_95_percent'] = convergence_epoch
        
        # Load summary report for timing
        summary_path = Path(f'./outputs/experiment_{model_name}/summary_report.json')
        if summary_path.exists():
            with open(summary_path, 'r') as f:
                summary = json.load(f)
                result['training_time_seconds'] = summary.get('training_duration_seconds', 0)
                result['training_time_hours'] = result['training_time_seconds'] / 3600
        
        return result

def create_synthetic_data(vocab_size: int, tag_size: int, seq_length: int, 
                         batch_size: int, char_vocab_size: int = 100):
    """Create synthetic data for testing."""
    # Word indices
    words = torch.randint(0, vocab_size, (batch_size, seq_length))
    tags = torch.randint(0, tag_size, (batch_size, seq_length))
    word_mask = torch.ones(batch_size, seq_length, dtype=torch.bool)
    
    # Character indices
    max_char_len = 15
    chars = torch.randint(0, char_vocab_size, (batch_size, seq_length, max_char_len))
    
    # BERT indices (simulate subword tokenization)
    bert_length = seq_length * 2  # Assume average 2 subwords per word
    bert_indices = torch.randint(0, 30000, (batch_size, bert_length))
    bert_mask = torch.ones(batch_size, bert_length, dtype=torch.long)
    
    # Word to subword mapping
    word_maps = []
    for _ in range(batch_size):
        mapping = []
        for i in range(seq_length):
            mapping.append(i * 2)  # Simple mapping
        word_maps.append(mapping)
    
    # Dummy tokens
    tokens = [['token'] * seq_length for _ in range(batch_size)]
    
    return {
        'tokens': tokens,
        'words': words,
        'chars': chars,
        'tags': tags,
        'word_mask': word_mask,
        'bert_indices': bert_indices,
        'bert_mask': bert_mask,
        'word_maps': word_maps
    }

def analyze_model_efficiency(model_name: str, model_configs: Dict):
    """Analyze efficiency of a single model."""
    analyzer = EfficiencyAnalyzer()
    results = []
    
    print(f"\nAnalyzing {model_name}...")
    
    # Load model
    model_path = Path(f'./outputs/experiment_{model_name}')
    if not model_path.exists():
        print(f"Model {model_name} not found, skipping...")
        return None
    
    # Model file analysis
    model_file = model_path / f'{model_name.replace("_", "-")}-crf-best.pt'
    if model_name == 'attention_fusion':
        model_file = model_path / 'attention-fusion-crf-best.pt'
    
    if model_file.exists():
        size_metrics = analyzer.measure_model_size(model_file)
    else:
        print(f"Model file not found: {model_file}")
        return None
    
    # Load model configuration
    with open(model_path / 'hyperparameters.json', 'r') as f:
        hp = json.load(f)
    
    with open(model_path / 'vocab.json', 'r') as f:
        vocab_data = json.load(f)
    
    # Initialize model
    tag_to_ix = vocab_data['tag_to_ix']
    
    if model_name == 'bilstm':
        vocab_size = len(vocab_data['word_to_ix'])
        embedding_dim = hp.get('embedding_dim', 300)
        hidden_dim = hp.get('lstm_hidden_dim', 256)
        lstm_layers = hp.get('lstm_layers', 1)
        dropout = hp.get('dropout', 0.3)
        model = BiLSTM_CRF(vocab_size, tag_to_ix, embedding_dim, hidden_dim, lstm_layers, dropout)
    elif model_name == 'bilstm_w2v':
        # Load Word2Vec embeddings for the model
        from flair.embeddings import WordEmbeddings
        word_embeddings = None
        try:
            word_embeddings = WordEmbeddings('id')
        except:
            pass
        model = BiLSTM_W2V_CRF(tag_to_ix, hp, word_to_ix=vocab_data.get('word_to_ix'), word_embeddings=word_embeddings)
    elif model_name == 'bilstm_w2v_cnn':
        w2v_vocab_size = len(vocab_data.get('word_to_ix', {}))
        char_vocab_size = len(vocab_data.get('char_to_ix', {}))
        model = BiLSTM_W2V_CNN_CRF(w2v_vocab_size, char_vocab_size, tag_to_ix, hp)
    elif model_name == 'indobert':
        model = IndoBERT_CRF(tag_to_ix, hp)
    elif model_name == 'indobert_bilstm':
        model = IndoBERT_BiLSTM_CRF(tag_to_ix, hp)
    elif model_name == 'mbert_bilstm':
        model = mBERT_BiLSTM_CRF(tag_to_ix, hp)
    elif model_name == 'xlm_roberta_bilstm':
        model = XLMRoBERTa_BiLSTM_CRF(tag_to_ix, hp)
    elif model_name == 'attention_fusion':
        word_vocab_size = len(vocab_data.get('word_to_ix', {}))
        char_vocab_size = len(vocab_data.get('char_to_ix', {}))
        model = MultiSourceAttention_CRF(word_vocab_size, 
                                       char_vocab_size, 
                                       tag_to_ix, hp)
    
    # Load model weights
    model.load_state_dict(torch.load(model_file, map_location='cpu', weights_only=True))
    model = model.to(DEVICE)
    
    model.eval()
    
    # Test different configurations
    for batch_size in BATCH_SIZES:
        for seq_length in SEQUENCE_LENGTHS:
            print(f"  Testing batch_size={batch_size}, seq_length={seq_length}")
            
            # Create synthetic data
            synthetic_batch = create_synthetic_data(
                len(vocab_data.get('word_to_ix', {'<PAD>': 0, '<UNK>': 1})),
                len(tag_to_ix),
                seq_length,
                batch_size,
                len(vocab_data.get('char_to_ix', {'<PAD>': 0, '<UNK>': 1}))
            )
            
            # Create simple dataset
            dataset = [synthetic_batch] * 20  # 20 batches
            
            # Inference speed
            speed_metrics = analyzer.measure_inference_speed(
                model, dataset, model_name, batch_size, seq_length
            )
            
            # Memory usage
            memory_metrics = analyzer.measure_memory_usage(
                model, dataset, model_name, batch_size, seq_length
            )
            
            # Combine results
            result = {
                'model': model_name,
                **size_metrics,
                **speed_metrics,
                **memory_metrics
            }
            results.append(result)
    
    # Training efficiency
    training_metrics = analyzer.measure_training_efficiency(model_name)
    
    return results, training_metrics

def create_efficiency_visualizations(all_results: pd.DataFrame, training_results: pd.DataFrame):
    """Create comprehensive efficiency visualizations."""
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 1. Model size comparison
    fig = plt.figure(figsize=(20, 15))
    
    # Model size and parameters
    plt.subplot(3, 3, 1)
    model_sizes = all_results.groupby('model').first()[['file_size_mb', 'params_millions']]
    
    x = np.arange(len(model_sizes))
    width = 0.35
    
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    bars1 = ax1.bar(x - width/2, model_sizes['file_size_mb'], width, label='File Size (MB)', color='skyblue')
    bars2 = ax2.bar(x + width/2, model_sizes['params_millions'], width, label='Parameters (M)', color='orange')
    
    ax1.set_xlabel('Model')
    ax1.set_ylabel('File Size (MB)', color='skyblue')
    ax2.set_ylabel('Parameters (Millions)', color='orange')
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_sizes.index, rotation=45, ha='right')
    ax1.tick_params(axis='y', labelcolor='skyblue')
    ax2.tick_params(axis='y', labelcolor='orange')
    plt.title('Model Size Comparison')
    
    # 2. Inference speed vs batch size
    plt.subplot(3, 3, 2)
    for model in all_results['model'].unique():
        model_data = all_results[(all_results['model'] == model) & 
                                (all_results['sequence_length'] == 50)]
        plt.plot(model_data['batch_size'], model_data['samples_per_second'], 
                marker='o', label=model)
    
    plt.xlabel('Batch Size')
    plt.ylabel('Samples per Second')
    plt.title('Inference Speed vs Batch Size (Seq Length = 50)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    
    # 3. Inference speed vs sequence length
    plt.subplot(3, 3, 3)
    for model in all_results['model'].unique():
        model_data = all_results[(all_results['model'] == model) & 
                                (all_results['batch_size'] == 8)]
        plt.plot(model_data['sequence_length'], model_data['tokens_per_second'], 
                marker='o', label=model)
    
    plt.xlabel('Sequence Length')
    plt.ylabel('Tokens per Second')
    plt.title('Throughput vs Sequence Length (Batch Size = 8)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    
    # 4. Memory usage comparison
    plt.subplot(3, 3, 4)
    memory_data = all_results[(all_results['batch_size'] == 16) & 
                             (all_results['sequence_length'] == 100)]
    
    models = memory_data['model'].tolist()
    cpu_memory = memory_data['cpu_memory_mb'].tolist()
    
    if 'gpu_memory_mb' in memory_data.columns:
        gpu_memory = memory_data['gpu_memory_mb'].tolist()
        
        x = np.arange(len(models))
        width = 0.35
        
        plt.bar(x - width/2, cpu_memory, width, label='CPU Memory', color='lightcoral')
        plt.bar(x + width/2, gpu_memory, width, label='GPU Memory', color='lightgreen')
        plt.xticks(x, models, rotation=45, ha='right')
    else:
        plt.bar(models, cpu_memory, color='lightcoral')
        plt.xticks(rotation=45, ha='right')
    
    plt.ylabel('Memory Usage (MB)')
    plt.title('Memory Usage Comparison (Batch=16, Seq=100)')
    plt.legend()
    plt.grid(True, axis='y')
    
    # 5. Training efficiency
    if not training_results.empty:
        plt.subplot(3, 3, 5)
        
        x = np.arange(len(training_results))
        width = 0.35
        
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        
        bars1 = ax1.bar(x - width/2, training_results['training_time_hours'], 
                        width, label='Training Time (h)', color='purple')
        bars2 = ax2.bar(x + width/2, training_results['epochs_to_95_percent'], 
                        width, label='Epochs to 95%', color='brown')
        
        ax1.set_xlabel('Model')
        ax1.set_ylabel('Training Time (hours)', color='purple')
        ax2.set_ylabel('Epochs to 95% Performance', color='brown')
        ax1.set_xticks(x)
        ax1.set_xticklabels(training_results['model'], rotation=45, ha='right')
        ax1.tick_params(axis='y', labelcolor='purple')
        ax2.tick_params(axis='y', labelcolor='brown')
        plt.title('Training Efficiency')
    
    # 6. Performance vs Efficiency Trade-off
    plt.subplot(3, 3, 6)
    if not training_results.empty:
        # Merge with performance data
        perf_eff = training_results[['model', 'best_dev_f1', 'training_time_hours']].copy()
        model_params = all_results.groupby('model')['params_millions'].first()
        perf_eff = perf_eff.merge(model_params, left_on='model', right_index=True)
        
        # Create bubble chart
        plt.scatter(perf_eff['params_millions'], perf_eff['best_dev_f1'], 
                   s=perf_eff['training_time_hours']*100, alpha=0.6)
        
        for _, row in perf_eff.iterrows():
            plt.annotate(row['model'], (row['params_millions'], row['best_dev_f1']),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.xlabel('Model Size (Million Parameters)')
        plt.ylabel('Best Dev F1 Score')
        plt.title('Performance vs Model Complexity\n(Bubble size = Training time)')
        plt.grid(True)
    
    # 7. Inference latency heatmap
    plt.subplot(3, 3, 7)
    latency_pivot = all_results.pivot_table(
        index='sequence_length',
        columns='batch_size',
        values='avg_inference_time',
        aggfunc='mean'
    )
    
    sns.heatmap(latency_pivot, annot=True, fmt='.3f', cmap='YlOrRd')
    plt.title('Average Inference Latency (seconds)')
    plt.xlabel('Batch Size')
    plt.ylabel('Sequence Length')
    
    # 8. Scalability analysis
    plt.subplot(3, 3, 8)
    for model in all_results['model'].unique()[:3]:  # Top 3 models
        model_data = all_results[all_results['model'] == model]
        
        # Calculate scalability score (tokens/sec normalized by model size)
        scalability = model_data['tokens_per_second'] / model_data['params_millions'].iloc[0]
        
        plt.plot(model_data['sequence_length'], scalability, 
                marker='o', label=model)
    
    plt.xlabel('Sequence Length')
    plt.ylabel('Scalability Score\n(Tokens/sec / Million params)')
    plt.title('Model Scalability Analysis')
    plt.legend()
    plt.grid(True)
    
    # 9. Cost-effectiveness matrix
    plt.subplot(3, 3, 9)
    if not training_results.empty:
        # Calculate cost-effectiveness score
        cost_eff = training_results[['model', 'best_dev_f1', 'training_time_hours']].copy()
        inference_speed = all_results.groupby('model')['samples_per_second'].mean()
        
        cost_eff = cost_eff.merge(inference_speed, left_on='model', right_index=True)
        cost_eff['efficiency_score'] = (cost_eff['best_dev_f1'] * cost_eff['samples_per_second']) / cost_eff['training_time_hours']
        
        plt.bar(cost_eff['model'], cost_eff['efficiency_score'])
        plt.xlabel('Model')
        plt.ylabel('Efficiency Score')
        plt.title('Cost-Effectiveness Score\n(F1 × Speed / Training Time)')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'efficiency_analysis_comprehensive.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_efficiency_report(all_results: pd.DataFrame, training_results: pd.DataFrame):
    """Generate detailed efficiency report."""
    with open(OUTPUT_DIR / 'efficiency_report.md', 'w') as f:
        f.write("# Comprehensive Efficiency Analysis for Indonesian NER Models\n\n")
        
        # Executive Summary
        f.write("## Executive Summary\n\n")
        
        # Find most efficient models
        avg_speed = all_results.groupby('model')['samples_per_second'].mean().sort_values(ascending=False)
        avg_memory = all_results.groupby('model')['cpu_memory_mb'].mean().sort_values()
        
        f.write(f"- **Fastest Model**: {avg_speed.index[0]} ({avg_speed.iloc[0]:.2f} samples/sec)\n")
        f.write(f"- **Most Memory Efficient**: {avg_memory.index[0]} ({avg_memory.iloc[0]:.2f} MB)\n")
        
        if not training_results.empty:
            fastest_training = training_results.loc[training_results['training_time_hours'].idxmin()]
            f.write(f"- **Fastest Training**: {fastest_training['model']} ({fastest_training['training_time_hours']:.2f} hours)\n")
        
        # Model Comparison Table
        f.write("\n## Model Comparison\n\n")
        
        comparison_data = []
        for model in all_results['model'].unique():
            model_data = all_results[all_results['model'] == model]
            
            row = {
                'Model': model,
                'Parameters (M)': model_data['params_millions'].iloc[0],
                'File Size (MB)': model_data['file_size_mb'].iloc[0],
                'Avg Speed (samples/s)': model_data['samples_per_second'].mean(),
                'Avg Memory (MB)': model_data['cpu_memory_mb'].mean()
            }
            
            if model in training_results['model'].values:
                train_data = training_results[training_results['model'] == model].iloc[0]
                row['Training Time (h)'] = train_data['training_time_hours']
                row['Best F1'] = train_data['best_dev_f1']
            
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        f.write(comparison_df.to_markdown(index=False, floatfmt='.2f'))
        
        # Detailed Analysis
        f.write("\n\n## Detailed Analysis\n\n")
        
        # Scalability Analysis
        f.write("### Scalability with Sequence Length\n\n")
        for model in all_results['model'].unique():
            model_data = all_results[(all_results['model'] == model) & 
                                    (all_results['batch_size'] == 8)]
            
            if len(model_data) > 1:
                # Calculate degradation rate
                speeds = model_data.sort_values('sequence_length')['tokens_per_second'].values
                if len(speeds) > 1:
                    degradation = (speeds[0] - speeds[-1]) / speeds[0] * 100
                    f.write(f"- **{model}**: {degradation:.1f}% throughput degradation from seq_len 10 to 200\n")
        
        # Memory Scaling
        f.write("\n### Memory Scaling\n\n")
        for model in all_results['model'].unique():
            model_data = all_results[all_results['model'] == model]
            
            # Linear regression to find memory scaling
            if len(model_data) > 2:
                try:
                    from scipy import stats
                    slope, intercept, r_value, _, _ = stats.linregress(
                        model_data['sequence_length'] * model_data['batch_size'],
                        model_data['cpu_memory_mb']
                    )
                    
                    f.write(f"- **{model}**: {slope:.4f} MB per token (R² = {r_value**2:.3f})\n")
                except ImportError:
                    # Fallback if scipy is not available
                    f.write(f"- **{model}**: Memory scaling analysis requires scipy\n")
        
        # Recommendations
        f.write("\n## Recommendations\n\n")
        
        f.write("### For Production Deployment\n\n")
        
        # High throughput scenario
        high_throughput = avg_speed.index[0]
        f.write(f"1. **High Throughput Applications**: Use {high_throughput}\n")
        f.write(f"   - Achieves {avg_speed.iloc[0]:.2f} samples/second on average\n")
        
        # Low latency scenario
        low_latency_data = all_results[all_results['batch_size'] == 1].groupby('model')['avg_inference_time'].mean()
        low_latency = low_latency_data.idxmin()
        f.write(f"\n2. **Low Latency Applications**: Use {low_latency}\n")
        f.write(f"   - Single sample latency: {low_latency_data[low_latency]*1000:.2f} ms\n")
        
        # Resource constrained
        f.write(f"\n3. **Resource-Constrained Environments**: Use {avg_memory.index[0]}\n")
        f.write(f"   - Average memory usage: {avg_memory.iloc[0]:.2f} MB\n")
        
        # Best trade-off
        if not training_results.empty:
            # Calculate composite score
            scores = []
            for model in all_results['model'].unique():
                if model in training_results['model'].values:
                    perf = training_results[training_results['model'] == model]['best_dev_f1'].iloc[0]
                    speed = avg_speed.get(model, 0)
                    memory = avg_memory.get(model, float('inf'))
                    
                    # Normalize and combine (higher is better)
                    score = (perf * speed) / (memory + 1)
                    scores.append((model, score))
            
            if scores:
                best_tradeoff = max(scores, key=lambda x: x[1])
                f.write(f"\n4. **Best Overall Trade-off**: {best_tradeoff[0]}\n")
                f.write(f"   - Balances performance, speed, and memory efficiency\n")
        
        # Batch size recommendations
        f.write("\n### Optimal Batch Sizes\n\n")
        for model in all_results['model'].unique():
            model_data = all_results[all_results['model'] == model].copy()
            
            # Find batch size with best throughput/memory ratio
            model_data['efficiency'] = model_data['samples_per_second'] / model_data['cpu_memory_mb']
            optimal_batch = model_data.loc[model_data['efficiency'].idxmax()]
            
            f.write(f"- **{model}**: Batch size {int(optimal_batch['batch_size'])} "
                   f"(efficiency score: {optimal_batch['efficiency']:.2f})\n")

def main():
    """Run comprehensive efficiency analysis."""
    print("=== Comprehensive Efficiency Analysis for Indonesian NER ===\n")
    
    models = ['bilstm', 'bilstm_w2v', 'bilstm_w2v_cnn', 'indobert', 'indobert_bilstm',
              'mbert_bilstm', 'xlm_roberta_bilstm', 'attention_fusion']    
    # Model configurations for initialization
    model_configs = {
        'bilstm': {},
        'bilstm_w2v': {},
        'bilstm_w2v_cnn': {},
        'indobert_bilstm': {'transformer_model': 'indobenchmark/indobert-base-p1'},
        'mbert_bilstm': {'transformer_model': 'bert-base-multilingual-cased'},
        'xlm_roberta_bilstm': {'transformer_model': 'xlm-roberta-base'},
        'attention_fusion': {'transformer_model': 'indobenchmark/indobert-base-p1'}
    }
    
    all_results = []
    training_results = []
    
    for model_name in models:
        result = analyze_model_efficiency(model_name, model_configs.get(model_name, {}))
        if result:
            model_results, training_metrics = result
            all_results.extend(model_results)
            if training_metrics:
                training_results.append(training_metrics)
    
    if not all_results:
        print("No results collected. Please ensure models are trained first.")
        return
    
    # Convert to DataFrames
    all_results_df = pd.DataFrame(all_results)
    training_results_df = pd.DataFrame(training_results)
    
    # Save raw results
    all_results_df.to_csv(OUTPUT_DIR / 'efficiency_raw_results.csv', index=False)
    training_results_df.to_csv(OUTPUT_DIR / 'training_efficiency.csv', index=False)
    
    # Create visualizations
    print("\nCreating visualizations...")
    create_efficiency_visualizations(all_results_df, training_results_df)
    
    # Generate report
    print("Generating efficiency report...")
    generate_efficiency_report(all_results_df, training_results_df)
    
    # Summary statistics
    print("\n=== Summary Statistics ===")
    print("\nModel Rankings:")
    
    # Speed ranking
    speed_ranking = all_results_df.groupby('model')['samples_per_second'].mean().sort_values(ascending=False)
    print("\nInference Speed (samples/sec):")
    for i, (model, speed) in enumerate(speed_ranking.items(), 1):
        print(f"{i}. {model}: {speed:.2f}")
    
    # Memory ranking
    memory_ranking = all_results_df.groupby('model')['cpu_memory_mb'].mean().sort_values()
    print("\nMemory Efficiency (MB):")
    for i, (model, memory) in enumerate(memory_ranking.items(), 1):
        print(f"{i}. {model}: {memory:.2f}")
    
    # Size ranking
    size_ranking = all_results_df.groupby('model')['params_millions'].first().sort_values()
    print("\nModel Size (Million params):")
    for i, (model, size) in enumerate(size_ranking.items(), 1):
        print(f"{i}. {model}: {size:.2f}")
    
    print(f"\nEfficiency analysis completed!")
    print(f"Results saved to: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
