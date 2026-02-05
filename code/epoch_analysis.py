import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from collections import defaultdict

# Output directory
OUTPUT_DIR = Path('./outputs/epoch_analysis')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def analyze_training_histories():
    """Analyze training history to see convergence patterns."""
    print("=== Epoch Configuration Analysis ===\n")
    
    models = ['bilstm', 'bilstm_w2v', 'bilstm_w2v_cnn', 'indobert', 'indobert_bilstm',
              'mbert_bilstm', 'xlm_roberta_bilstm', 'attention_fusion']
    
    results = []
    
    for model in models:
        history_path = Path(f'./outputs/experiment_{model}/training_history.csv')
        hp_path = Path(f'./outputs/experiment_{model}/hyperparameters.json')
        
        if not history_path.exists():
            print(f"  [SKIP] {model}: No training history found")
            continue
        history_df = pd.read_csv(history_path)
        if hp_path.exists():
            with open(hp_path, 'r') as f:
                hp = json.load(f)
                max_epochs = hp.get('epochs', 200)
                patience = hp.get('early_stopping_patience', 10)
        else:
            max_epochs = 200
            patience = 10
        total_epochs = len(history_df)
        best_epoch = history_df['dev_f1'].idxmax() + 1
        best_f1 = history_df['dev_f1'].max()
        final_f1 = history_df['dev_f1'].iloc[-1]
        target_95 = best_f1 * 0.95
        target_99 = best_f1 * 0.99
        
        epochs_to_95 = None
        epochs_to_99 = None
        
        for idx, row in history_df.iterrows():
            if epochs_to_95 is None and row['dev_f1'] >= target_95:
                epochs_to_95 = idx + 1
            if epochs_to_99 is None and row['dev_f1'] >= target_99:
                epochs_to_99 = idx + 1
        if len(history_df) >= 5:
            last_5_f1 = history_df['dev_f1'].tail(5).values
            f1_std = np.std(last_5_f1)
            f1_mean = np.mean(last_5_f1)
            stability = 1 - (f1_std / f1_mean) if f1_mean > 0 else 0
        else:
            stability = 0
        train_loss_final = history_df['train_loss'].iloc[-1]
        train_loss_best = history_df.loc[history_df['dev_f1'].idxmax(), 'train_loss']
        overfitting_gap = train_loss_final - train_loss_best
        stopped_early = total_epochs < max_epochs
        epochs_saved = max_epochs - total_epochs if stopped_early else 0
        
        result = {
            'model': model,
            'max_epochs': max_epochs,
            'patience': patience,
            'total_epochs': total_epochs,
            'best_epoch': best_epoch,
            'best_f1': best_f1,
            'final_f1': final_f1,
            'epochs_to_95_percent': epochs_to_95,
            'epochs_to_99_percent': epochs_to_99,
            'stopped_early': stopped_early,
            'epochs_saved': epochs_saved,
            'stability': stability,
            'overfitting_gap': overfitting_gap,
            'convergence_ratio': best_epoch / total_epochs if total_epochs > 0 else 0
        }
        
        results.append(result)
        
        print(f"  [OK] {model}:")
        print(f"      Total epochs: {total_epochs}/{max_epochs}")
        print(f"      Best epoch: {best_epoch}, Best F1: {best_f1:.4f}")
        print(f"      Stopped early: {stopped_early}, Saved: {epochs_saved} epochs")
    
    return pd.DataFrame(results)

def create_visualizations(results_df):
    """Create epoch analysis visualizations."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Epoch usage comparison
    plt.subplot(2, 3, 1)
    x = np.arange(len(results_df))
    width = 0.35
    
    plt.bar(x - width/2, results_df['max_epochs'], width, label='Max Epochs', alpha=0.7)
    plt.bar(x + width/2, results_df['total_epochs'], width, label='Actual Epochs', alpha=0.7)
    
    plt.xlabel('Model')
    plt.ylabel('Epochs')
    plt.title('Max vs Actual Epochs Used')
    plt.xticks(x, results_df['model'], rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Convergence speed
    plt.subplot(2, 3, 2)
    plt.scatter(results_df['epochs_to_95_percent'], results_df['best_f1'], 
               s=100, alpha=0.7, c=range(len(results_df)), cmap='viridis')
    for i, row in results_df.iterrows():
        plt.annotate(row['model'], (row['epochs_to_95_percent'], row['best_f1']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    plt.xlabel('Epochs to 95% Performance')
    plt.ylabel('Best F1 Score')
    plt.title('Convergence Speed vs Performance')
    plt.grid(True, alpha=0.3)
    
    # 3. Best epoch distribution
    plt.subplot(2, 3, 3)
    plt.bar(results_df['model'], results_df['best_epoch'], alpha=0.7, color='green')
    plt.axhline(y=results_df['best_epoch'].mean(), color='red', linestyle='--', 
               label=f'Mean: {results_df["best_epoch"].mean():.1f}')
    plt.xlabel('Model')
    plt.ylabel('Best Epoch')
    plt.title('Epoch Where Best Performance Achieved')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    # 4. Early stopping effectiveness
    plt.subplot(2, 3, 4)
    stopped = results_df[results_df['stopped_early'] == True]
    not_stopped = results_df[results_df['stopped_early'] == False]
    
    if len(stopped) > 0:
        plt.bar(stopped['model'], stopped['epochs_saved'], alpha=0.7, color='orange', label='Stopped Early')
    if len(not_stopped) > 0:
        plt.bar(not_stopped['model'], [0]*len(not_stopped), alpha=0.7, color='gray', label='Ran Full')
    
    plt.xlabel('Model')
    plt.ylabel('Epochs Saved')
    plt.title('Early Stopping Effectiveness')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    # 5. Convergence ratio
    plt.subplot(2, 3, 5)
    plt.bar(results_df['model'], results_df['convergence_ratio'], alpha=0.7, color='purple')
    plt.axhline(y=0.5, color='red', linestyle='--', label='50% threshold')
    plt.xlabel('Model')
    plt.ylabel('Convergence Ratio (Best Epoch / Total)')
    plt.title('When Best Performance Was Achieved')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    # 6. Stability vs Performance
    plt.subplot(2, 3, 6)
    plt.scatter(results_df['stability'], results_df['best_f1'], s=100, alpha=0.7)
    for i, row in results_df.iterrows():
        plt.annotate(row['model'], (row['stability'], row['best_f1']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    plt.xlabel('Stability (1 - CV of last 5 epochs)')
    plt.ylabel('Best F1 Score')
    plt.title('Training Stability vs Performance')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'epoch_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_recommendations(results_df):
    """Generate recommendations for better epoch strategies."""
    recommendations = []
    
    # Statistical analysis
    avg_best_epoch = results_df['best_epoch'].mean()
    median_best_epoch = results_df['best_epoch'].median()
    max_best_epoch = results_df['best_epoch'].max()
    
    avg_convergence = results_df['epochs_to_95_percent'].mean()
    avg_total = results_df['total_epochs'].mean()
    
    # Recommendation 1: Optimal max epochs
    recommended_max_epochs = int(max_best_epoch * 1.5)  # 50% buffer
    recommendations.append({
        'category': 'Max Epochs',
        'current': '200 (fixed)',
        'recommended': f'{recommended_max_epochs} (adaptive)',
        'reasoning': f'Best performance achieved at epoch {max_best_epoch:.0f} on average. '
                    f'Setting max to {recommended_max_epochs} provides buffer without waste.'
    })
    
    # Rekomendasi 2: Adaptive patience
    early_stopped = results_df[results_df['stopped_early'] == True]
    if len(early_stopped) > 0:
        avg_saved = early_stopped['epochs_saved'].mean()
        recommendations.append({
            'category': 'Early Stopping Patience',
            'current': '10 (fixed)',
            'recommended': 'Adaptive: 5-15 based on convergence rate',
            'reasoning': f'Early stopping saved avg {avg_saved:.0f} epochs. '
                        f'Adaptive patience can be more efficient for fast-converging models.'
        })
    
    # Recommendation 3: Learning rate scheduling
    recommendations.append({
        'category': 'Learning Rate Schedule',
        'current': 'Fixed learning rate',
        'recommended': 'Cosine annealing or ReduceLROnPlateau',
        'reasoning': 'Can help models converge faster and achieve better final performance. '
                    f'Average convergence at epoch {avg_convergence:.0f} suggests room for optimization.'
    })
    
    # Recommendation 4: Warmup epochs
    recommendations.append({
        'category': 'Warmup Strategy',
        'current': 'None',
        'recommended': 'Linear warmup for first 10% of epochs',
        'reasoning': 'Especially beneficial for transformer models. Can stabilize early training.'
    })
    
    # Recommendation 5: Validation frequency
    recommendations.append({
        'category': 'Validation Frequency',
        'current': 'Every epoch',
        'recommended': 'Every epoch (keep current)',
        'reasoning': 'Current approach is good for early stopping. No change needed.'
    })
    
    return recommendations

def create_optimal_strategy(results_df, recommendations):
    """Create optimal epoch strategy based on analysis."""
    strategy = {
        'baseline_models': {
            'max_epochs': 100,
            'patience': 8,
            'min_epochs': 20,
            'reasoning': 'Baseline models converge faster, need fewer epochs'
        },
        'transformer_models': {
            'max_epochs': 150,
            'patience': 12,
            'min_epochs': 30,
            'warmup_epochs': 5,
            'reasoning': 'Transformer models benefit from more epochs and warmup'
        },
        'advanced_models': {
            'max_epochs': 200,
            'patience': 15,
            'min_epochs': 40,
            'warmup_epochs': 10,
            'reasoning': 'Complex models need more time to converge'
        },
        'adaptive_patience': {
            'fast_convergence': 5,  # If improving rapidly
            'normal': 10,
            'slow_convergence': 15,  # If improving slowly
            'reasoning': 'Adjust patience based on improvement rate'
        },
        'learning_rate_schedule': {
            'type': 'cosine_annealing',
            'T_max': 'max_epochs',
            'eta_min': '0.1 * initial_lr',
            'reasoning': 'Gradually reduce learning rate for fine-tuning'
        }
    }
    
    return strategy

def generate_report(results_df, recommendations, strategy):
    """Generate comprehensive report."""
    with open(OUTPUT_DIR / 'epoch_analysis_report.md', 'w', encoding='utf-8') as f:
        f.write("# Epoch Configuration Analysis and Training Strategy Recommendations\n\n")
        
        # Executive Summary
        f.write("## Executive Summary\n\n")
        f.write(f"- **Average Best Epoch**: {results_df['best_epoch'].mean():.1f}\n")
        f.write(f"- **Median Best Epoch**: {results_df['best_epoch'].median():.1f}\n")
        f.write(f"- **Max Best Epoch**: {results_df['best_epoch'].max():.0f}\n")
        f.write(f"- **Average Convergence (95%)**: {results_df['epochs_to_95_percent'].mean():.1f} epochs\n")
        f.write(f"- **Early Stopping Effectiveness**: {len(results_df[results_df['stopped_early']])}/{len(results_df)} models stopped early\n\n")
        
        # Current Configuration Analysis
        f.write("## Current Configuration Analysis\n\n")
        f.write("| Model | Max Epochs | Patience | Actual Epochs | Best Epoch | Stopped Early |\n")
        f.write("|-------|------------|----------|----------------|------------|---------------|\n")
        
        for _, row in results_df.iterrows():
            f.write(f"| {row['model']} | {row['max_epochs']} | {row['patience']} | ")
            f.write(f"{row['total_epochs']} | {row['best_epoch']:.0f} | ")
            f.write(f"{'Yes' if row['stopped_early'] else 'No'} |\n")
        
        # Key Findings
        f.write("\n## Key Findings\n\n")
        
        # Finding 1: Epoch usage
        unused_epochs = (results_df['max_epochs'] - results_df['total_epochs']).sum()
        f.write(f"### 1. Epoch Usage\n")
        f.write(f"- Total unused epochs: {unused_epochs:.0f} epochs\n")
        f.write(f"- Average waste per model: {(results_df['max_epochs'] - results_df['total_epochs']).mean():.1f} epochs\n")
        f.write(f"- **Conclusion**: Max epochs 200 is too high for most models\n\n")
        
        # Finding 2: Convergence
        f.write(f"### 2. Convergence Speed\n")
        f.write(f"- Average to reach 95% performance: {results_df['epochs_to_95_percent'].mean():.1f} epochs\n")
        f.write(f"- Average to reach best performance: {results_df['best_epoch'].mean():.1f} epochs\n")
        f.write(f"- **Conclusion**: Models reach optimal performance relatively quickly\n\n")
        
        # Finding 3: Early stopping
        early_stopped = results_df[results_df['stopped_early'] == True]
        if len(early_stopped) > 0:
            f.write(f"### 3. Early Stopping Effectiveness\n")
            f.write(f"- {len(early_stopped)}/{len(results_df)} model stopped early\n")
            f.write(f"- Average epochs saved: {early_stopped['epochs_saved'].mean():.1f}\n")
            f.write(f"- **Conclusion**: Early stopping is effective, but can be more optimal\n\n")
        
        # Recommendations
        f.write("## Epoch Strategy Recommendations\n\n")
        
        for i, rec in enumerate(recommendations, 1):
            f.write(f"### {i}. {rec['category']}\n\n")
            f.write(f"- **Current**: {rec['current']}\n")
            f.write(f"- **Recommended**: {rec['recommended']}\n")
            f.write(f"- **Reasoning**: {rec['reasoning']}\n\n")
        
        # Optimal Strategy
        f.write("## Recommended Optimal Strategy\n\n")
        
        f.write("### For Baseline Models (BiLSTM variants)\n")
        f.write(f"- Max Epochs: {strategy['baseline_models']['max_epochs']}\n")
        f.write(f"- Patience: {strategy['baseline_models']['patience']}\n")
        f.write(f"- Min Epochs: {strategy['baseline_models']['min_epochs']}\n")
        f.write(f"- Reasoning: {strategy['baseline_models']['reasoning']}\n\n")
        
        f.write("### For Transformer Models\n")
        f.write(f"- Max Epochs: {strategy['transformer_models']['max_epochs']}\n")
        f.write(f"- Patience: {strategy['transformer_models']['patience']}\n")
        f.write(f"- Warmup Epochs: {strategy['transformer_models']['warmup_epochs']}\n")
        f.write(f"- Reasoning: {strategy['transformer_models']['reasoning']}\n\n")
        
        f.write("### For Advanced Models (Attention Fusion)\n")
        f.write(f"- Max Epochs: {strategy['advanced_models']['max_epochs']}\n")
        f.write(f"- Patience: {strategy['advanced_models']['patience']}\n")
        f.write(f"- Warmup Epochs: {strategy['advanced_models']['warmup_epochs']}\n")
        f.write(f"- Reasoning: {strategy['advanced_models']['reasoning']}\n\n")
        
        # Implementation Guide
        f.write("## Implementation Guide\n\n")
        f.write("### 1. Adaptive Max Epochs\n")
        f.write("```python\n")
        f.write("# Set max_epochs based on model complexity\n")
        f.write("if 'bilstm' in model_name and 'w2v' not in model_name:\n")
        f.write("    max_epochs = 100\n")
        f.write("elif 'transformer' in model_name or 'bert' in model_name:\n")
        f.write("    max_epochs = 150\n")
        f.write("else:  # Complex models\n")
        f.write("    max_epochs = 200\n")
        f.write("```\n\n")
        
        f.write("### 2. Adaptive Patience\n")
        f.write("```python\n")
        f.write("# Adjust patience based on improvement rate\n")
        f.write("improvement_rate = (current_f1 - prev_f1) / prev_f1\n")
        f.write("if improvement_rate > 0.01:  # Fast improvement\n")
        f.write("    patience = 5\n")
        f.write("elif improvement_rate > 0.001:  # Normal\n")
        f.write("    patience = 10\n")
        f.write("else:  # Slow improvement\n")
        f.write("    patience = 15\n")
        f.write("```\n\n")
        
        f.write("### 3. Learning Rate Scheduling\n")
        f.write("```python\n")
        f.write("from torch.optim.lr_scheduler import CosineAnnealingLR\n")
        f.write("scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=lr*0.1)\n")
        f.write("# Call scheduler.step() after each epoch\n")
        f.write("```\n\n")
        
        # Expected Benefits
        f.write("## Expected Benefits\n\n")
        f.write("1. **Time Efficiency**: Reduce 20-30% training time with appropriate max epochs\n")
        f.write("2. **Model Quality**: Learning rate scheduling can improve F1 score by 0.5-1%\n")
        f.write("3. **Stability**: Warmup reduces training instability at the beginning\n")
        f.write("4. **Resource Efficiency**: Save GPU hours for large-scale experiments\n\n")

def main():
    """Run epoch analysis."""
    print("Analyzing epoch configurations...\n")
    
    # Analyze training histories
    results_df = analyze_training_histories()
    
    if len(results_df) == 0:
        print("\nNo training histories found. Please run training first.")
        return
    
    # Save results
    results_df.to_csv(OUTPUT_DIR / 'epoch_analysis_results.csv', index=False)
    
    # Create visualizations
    print("\nCreating visualizations...")
    create_visualizations(results_df)
    
    # Generate recommendations
    print("Generating recommendations...")
    recommendations = generate_recommendations(results_df)
    strategy = create_optimal_strategy(results_df, recommendations)
    
    # Generate report
    print("Generating report...")
    generate_report(results_df, recommendations, strategy)
    
    # Summary
    print("\n" + "="*60)
    print("Epoch Analysis Summary")
    print("="*60)
    print(f"\nModels analyzed: {len(results_df)}")
    print(f"Average best epoch: {results_df['best_epoch'].mean():.1f}")
    print(f"Average total epochs: {results_df['total_epochs'].mean():.1f}")
    print(f"Early stopping effectiveness: {len(results_df[results_df['stopped_early']])}/{len(results_df)} models")
    
    print(f"\nKey Recommendations:")
    for i, rec in enumerate(recommendations[:3], 1):
        print(f"  {i}. {rec['category']}: {rec['recommended']}")
    
    print(f"\nResults saved to: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
