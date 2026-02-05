import json
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from collections import Counter
from math import sqrt

# Suppress matplotlib/seaborn UserWarnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 14,
    'figure.titlesize': 20,
    'figure.dpi': 300
})

def parse_classification_report(report_path):
    """
    Parse classification_report.txt and summary_report.json files for complete data.
    """
    model_name = report_path.parent.name.replace('experiment_', '').replace('_', ' ').title()
    try:
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        return None, None

    training_time, inference_time, num_params = 0.0, 0.0, 0
    summary_path = report_path.parent / 'summary_report.json'
    if summary_path.exists():
        with open(summary_path, 'r', encoding='utf-8') as f:
            summary_data = json.load(f)
            training_time = summary_data.get('training_duration_seconds', 0.0)
            inference_time = summary_data.get('evaluation_time_seconds', 0.0)
            model_info = summary_data.get('model_info', {})
            num_params = model_info.get('num_trainable_params', 0)

    lines = content.split('\n')
    data = []
    table_started = False
    for line in lines:
        if 'precision' in line and 'recall' in line and 'f1-score' in line: table_started = True; continue
        if not table_started or line.strip() == "" or "accuracy" in line: continue
        if "macro avg" in line or "weighted avg" in line:
            parts = line.split(); class_name = parts[0] + " " + parts[1]; precision, recall, f1_score, support = parts[2], parts[3], parts[4], parts[5]
        else:
            match = re.match(r'^\s*([a-zA-Z0-9\-_ ]+?)\s{2,}(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+)', line)
            if match: class_name, precision, recall, f1_score, support = match.groups()
            else: continue
        data.append({
            'model': model_name, 'class': class_name.strip(), 'precision': float(precision),
            'recall': float(recall), 'f1-score': float(f1_score), 'support': int(support)
        })
        
    if not data: return None, None

    df = pd.DataFrame(data)
    df['training_time'] = training_time
    df['inference_time'] = inference_time
    df['num_params'] = num_params
    
    return df, model_name

def generate_summary_tables_str(main_df):
    """Generate Markdown string for performance summary tables."""
    if main_df.empty: return "", None

    report_lines = []

    report_lines.append("### Table 1: Model Performance Comparison (Overall)")
    overall_df = main_df[main_df['class'] == 'weighted avg'].drop_duplicates(subset='model').copy()
    pivot_df = main_df[main_df['class'].isin(['micro avg', 'macro avg', 'weighted avg'])]
    # Use pivot_table instead of pivot to handle duplicate entries
    pivot_df = pivot_df.pivot_table(index='model', columns='class', values='f1-score', aggfunc='first').reset_index()
    summary_df = pd.merge(pivot_df, overall_df[['model', 'training_time', 'inference_time', 'num_params']], on='model')

    summary_df.rename(columns={
        'micro avg': 'F1 Micro', 'macro avg': 'F1 Macro', 'weighted avg': 'F1 Weighted',
        'training_time': 'Training Time (s)', 'inference_time': 'Inference Time (s)', 'num_params': 'Trainable Params'
    }, inplace=True)
    
    summary_df.sort_values(by="F1 Weighted", ascending=False, inplace=True)
    
    display_cols = ["model", "F1 Weighted", "F1 Macro", "Training Time (s)", "Trainable Params"]
    if 'Trainable Params' not in summary_df.columns or summary_df['Trainable Params'].isnull().all():
        display_cols.remove('Trainable Params')

    report_lines.append(summary_df[display_cols].to_markdown(index=False, floatfmt=".4f"))
    report_lines.append("\n" + "-"*80 + "\n")

    report_lines.append("### Table 2: Entity Performance Analysis (F1-Score / Precision / Recall)")
    entity_df = main_df[~main_df['class'].str.contains('avg')].copy()
    entity_df['F1/P/R'] = entity_df.apply(lambda r: f"{r['f1-score']:.4f} / {r['precision']:.4f} / {r['recall']:.4f}", axis=1)
    entity_pivot = entity_df.pivot_table(index='model', columns='class', values='F1/P/R', aggfunc='first')
    entity_pivot = entity_pivot.loc[summary_df['model']]
    report_lines.append(entity_pivot.to_markdown())
    
    return "\n".join(report_lines), summary_df

def generate_hyperparameter_table_str(output_dir):
    """Generate Markdown string for hyperparameter comparison table."""
    hyper_files = list(output_dir.glob('**/hyperparameters.json'))
    if not hyper_files: return "Warning: No 'hyperparameters.json' files found."

    all_hypers = []
    for hyper_file in hyper_files:
        model_name = hyper_file.parent.name.replace('experiment_', '').replace('_', ' ').title()
        with open(hyper_file, 'r') as f: hypers = json.load(f); hypers['model'] = model_name; all_hypers.append(hypers)
            
    hyper_df = pd.DataFrame(all_hypers).set_index('model')
    cols = hyper_df.columns.tolist()
    common_cols = ['embedding_dim', 'lstm_hidden_dim', 'lstm_layers', 'dropout', 'learning_rate', 'batch_size', 'epochs']
    ordered_cols = [col for col in common_cols if col in cols] + [col for col in cols if col not in common_cols]
    hyper_df = hyper_df[ordered_cols]
    hyper_df.rename(columns={
        'embedding_dim': 'Embedding Dim', 'lstm_hidden_dim': 'LSTM Hidden Dim', 'lstm_layers': 'LSTM Layers',
        'dropout': 'Dropout', 'learning_rate': 'Learning Rate', 'batch_size': 'Batch Size', 'epochs': 'Epochs',
        'early_stopping_patience': 'Patience'
    }, inplace=True)

    report_lines = ["\n### Table 3: Model Hyperparameter Comparison", hyper_df.to_markdown(floatfmt=".0e"), "\n" + "-"*80 + "\n"]
    return "\n".join(report_lines)

def perform_error_analysis(output_dir, results_dir, best_model_name):
    """Create confusion matrix plots and error analysis, then return summary string."""
    report_lines = ["\n### Error Analysis: Confusion Matrix & Common Errors"]
    
    model_slug = "experiment_" + best_model_name.lower().replace(' ', '_')
    best_model_path = output_dir / model_slug / 'test_predictions.txt'

    if not best_model_path.exists(): 
        return f"Warning: File 'test_predictions.txt' for best model '{best_model_name}' not found."

    def get_tags_from_file(file_path):
        true_tags, pred_tags = [], []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip();
                if not line: continue
                parts = line.split()
                if len(parts) == 3: true_tags.append(parts[1]); pred_tags.append(parts[2])
        return true_tags, pred_tags

    true_tags, best_model_preds = get_tags_from_file(best_model_path)
    if not true_tags: return "No valid tag data found for analysis."

    labels = sorted(list(set(true_tags) | set(best_model_preds)))
    cm = pd.crosstab(pd.Series(true_tags, name='True'), pd.Series(best_model_preds, name='Predicted'), dropna=False)
    
    plt.figure(figsize=(12, 10)); sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix for Best Model (Absolute Numbers)'); plt.tight_layout()
    cm_path = results_dir / 'confusion_matrix_best_model.png'; plt.savefig(cm_path); plt.close()
    report_lines.append(f"Confusion matrix (absolute) saved to '{cm_path}'")

    cm_normalized = cm.astype('float') / cm.sum(axis=1).to_numpy()[:, np.newaxis]
    plt.figure(figsize=(12, 10)); sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='cividis', xticklabels=labels, yticklabels=labels)
    plt.title('Normalized Confusion Matrix for Best Model (Rates)'); plt.tight_layout()
    cm_norm_path = results_dir / 'confusion_matrix_normalized.png'; plt.savefig(cm_norm_path); plt.close()
    report_lines.append(f"Confusion matrix (normalized) saved to '{cm_norm_path}'")

    errors = [(true, pred) for true, pred in zip(true_tags, best_model_preds) if true != pred]
    error_counts = Counter(errors)
    top_n = 15
    error_df = pd.DataFrame(error_counts.most_common(top_n), columns=['error_pair', 'count'])
    error_df['error_str'] = error_df['error_pair'].apply(lambda x: f"{x[0]} -> {x[1]}")

    plt.figure(figsize=(10, 8));     sns.barplot(x='count', y='error_str', data=error_df, hue='error_str', palette='plasma', legend=False)
    plt.title(f'Top {top_n} Classification Errors'); plt.xlabel('Frequency Count'); plt.ylabel('Error Type (True -> Predicted)'); plt.tight_layout()
    top_errors_path = results_dir / 'top_errors_barchart.png'; plt.savefig(top_errors_path); plt.close()
    report_lines.append(f"Top {top_n} errors plot saved to '{top_errors_path}'")

    report_lines.append(f"\nTop {top_n} Most Common Errors (True -> Predicted):")
    for (true, pred), count in error_counts.most_common(top_n):
        report_lines.append(f"{count:>5}: {true} -> {pred}")
    report_lines.append("\n" + "-"*80 + "\n")
    return "\n".join(report_lines)

def generate_statistical_tests_str(output_dir, best_model_name, other_models):
    """Generate Markdown string for McNemar's test results."""
    report_lines = ["\n### Table 4: Statistical Significance Test (McNemar's Test)"]
    best_model_slug = "experiment_" + best_model_name.lower().replace(' ', '_')
    best_model_preds_path = output_dir / best_model_slug / 'test_predictions.txt'

    if not best_model_preds_path.exists(): return f"Warning: Prediction file for best model '{best_model_name}' not found."

    def get_predictions(file_path):
        true_tags, pred_tags = [], []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip();
                if not line: continue
                parts = line.split()
                if len(parts) == 3: true_tags.append(parts[1]); pred_tags.append(parts[2])
        return true_tags, pred_tags

    true_labels, best_model_preds = get_predictions(best_model_preds_path)
    results = []

    for model_name in other_models:
        if model_name == best_model_name: continue
        model_slug = "experiment_" + model_name.lower().replace(' ', '_')
        model_preds_path = output_dir / model_slug / 'test_predictions.txt'
        if not model_preds_path.exists(): continue
        _, other_model_preds = get_predictions(model_preds_path)

        yes_no, no_yes = 0, 0
        for i in range(len(true_labels)):
            best_correct = best_model_preds[i] == true_labels[i]
            other_correct = other_model_preds[i] == true_labels[i]
            if best_correct and not other_correct: yes_no += 1
            elif not best_correct and other_correct: no_yes += 1
        
        if (yes_no + no_yes) == 0: continue
        chi2_stat = ((abs(yes_no - no_yes) - 1)**2) / (yes_no + no_yes)
        p_value = "< 0.01" if chi2_stat > 6.63 else "< 0.05" if chi2_stat > 3.84 else "> 0.05"
        results.append({
            'Comparison': f"{best_model_name} vs. {model_name}", 'chi2-statistic': chi2_stat,
            'p-value': p_value, 'Result': 'Significant' if chi2_stat > 3.84 else 'Not Significant'
        })

    if not results: return "No comparisons can be made."
    results_df = pd.DataFrame(results)
    report_lines.append(results_df.to_markdown(index=False, floatfmt=".3f"))
    report_lines.append("\n" + "-"*80 + "\n")
    return "\n".join(report_lines)

def plot_learning_curves(results_dir):
    """Create learning curve comparison plots with publication style."""
    print("\n--- Creating Learning Curve Plots ---")
    history_files = list(Path('outputs').glob('**/training_history.csv'))
    if not history_files: print("Warning: No 'training_history.csv' files found."); return
    all_histories = []
    for history_file in history_files:
        model_name = history_file.parent.name.replace('experiment_', '').replace('_', ' ').title()
        try: history_df = pd.read_csv(history_file); history_df['model'] = model_name; all_histories.append(history_df)
        except pd.errors.EmptyDataError: print(f"Warning: History file for '{model_name}' is empty.")
    if not all_histories: print("No valid history data to plot."); return
    full_history_df = pd.concat(all_histories, ignore_index=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 15), sharex=True); fig.suptitle('Model Learning Curves', y=1.02)
    sns.lineplot(data=full_history_df, x='epoch', y='dev_f1', hue='model', ax=ax1, marker='o', markersize=6, palette='viridis')
    ax1.set_title('Validation F1-Score vs. Epoch'); ax1.set_ylabel('Macro F1-Score'); ax1.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left'); ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    sns.lineplot(data=full_history_df, x='epoch', y='train_loss', hue='model', ax=ax2, marker='X', markersize=6, palette='viridis')
    ax2.set_title('Training Loss vs. Epoch'); ax2.set_xlabel('Epoch'); ax2.set_ylabel('Average Loss'); ax2.get_legend().remove(); ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout(rect=[0, 0, 0.85, 1]); plot_path = results_dir / 'learning_curves_comparison.png'; plt.savefig(plot_path); plt.close()
    print(f"Learning curve plot saved to '{plot_path}'")

def create_plots_and_csv(main_df, results_dir):
    """
    Create all plots and save CSV files with publication style.
    """
    print("\n--- Creating Detailed Report (Plots and CSV) ---")
    summary_df = main_df[main_df['class'] == 'weighted avg'].drop_duplicates(subset='model').copy()
    summary_df.sort_values(by='f1-score', ascending=False, inplace=True)
    summary_csv_path = results_dir / 'overall_summary.csv'; summary_df.to_csv(summary_csv_path, index=False)
    print(f"Summary table saved to '{summary_csv_path}'")
    entity_df = main_df[~main_df['class'].str.contains('avg')]
    plt.figure(figsize=(14, 8)); sns.barplot(data=entity_df, x='class', y='f1-score', hue='model', palette='plasma')
    plt.title('F1-Score Comparison by Entity Type'); plt.xlabel('Entity Type'); plt.ylabel('F1-Score'); plt.xticks(rotation=45, ha='right'); plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left'); plt.grid(axis='y', linestyle='--', linewidth=0.7); plt.tight_layout(rect=[0, 0, 0.85, 1])
    entity_plot_path = results_dir / 'f1_score_per_entity.png'; plt.savefig(entity_plot_path); plt.close()
    print(f"F1 per entity plot saved to '{entity_plot_path}'")
    f1_pivot = entity_df.pivot_table(index='class', columns='model', values='f1-score'); f1_pivot_path = results_dir / 'f1_per_entity.csv'; f1_pivot.to_csv(f1_pivot_path)
    print(f"F1 per entity table saved to '{f1_pivot_path}'")
    plt.figure(figsize=(12, 8)); plot_df = summary_df.copy()
    scatter = sns.scatterplot(data=plot_df, x='training_time', y='f1-score', hue='model', s=250, alpha=0.9, palette='cividis')
    plt.title('Performance vs. Training Time Trade-off'); plt.xlabel('Training Time (seconds)'); plt.ylabel('Weighted F1-Score'); plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    for i in range(plot_df.shape[0]): plt.text(x=plot_df['training_time'].iloc[i], y=plot_df['f1-score'].iloc[i] + 0.002, s=plot_df['model'].iloc[i], fontdict=dict(color='black', size=12, ha='center'))
    plt.grid(True, which='both', linestyle='--', linewidth=0.5); plt.tight_layout(rect=[0, 0, 0.85, 1])
    tradeoff_plot_path = results_dir / 'performance_vs_time_tradeoff.png'; plt.savefig(tradeoff_plot_path); plt.close()
    print(f"Performance vs time trade-off plot saved to '{tradeoff_plot_path}'")

    if 'num_params' in summary_df.columns and summary_df['num_params'].notna().any():
        plt.figure(figsize=(12, 8))
        plot_df_params = summary_df.dropna(subset=['num_params']).copy()
        plot_df_params['num_params_millions'] = plot_df_params['num_params'] / 1e6
        
        scatter = sns.scatterplot(data=plot_df_params, x='num_params_millions', y='f1-score', hue='model', s=250, alpha=0.9, palette='viridis')
        plt.title('Performance vs. Model Complexity Trade-off')
        plt.xlabel('Trainable Parameters (Millions)')
        plt.ylabel('Weighted F1-Score')
        plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        for i in range(plot_df_params.shape[0]):
            plt.text(x=plot_df_params['num_params_millions'].iloc[i], y=plot_df_params['f1-score'].iloc[i] + 0.002, s=plot_df_params['model'].iloc[i], fontdict=dict(color='black', size=12, ha='center'))
        
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        complexity_plot_path = results_dir / 'performance_vs_complexity_tradeoff.png'
        plt.savefig(complexity_plot_path); plt.close()
        print(f"Performance vs complexity trade-off plot saved to '{complexity_plot_path}'")

def main():
    """
    Main script to run all analyses, create plots, and print summary.
    """
    output_dir = Path('outputs')
    results_dir = Path('comparison_results')
    results_dir.mkdir(exist_ok=True)

    report_files = list(output_dir.glob('**/classification_report.txt'))
    if not report_files: print("No 'classification_report.txt' files found."); return

    all_dfs = []
    for report_file in report_files: 
        df, _ = parse_classification_report(report_file)
        if df is not None: 
            all_dfs.append(df)

    if not all_dfs: print("Failed to parse data from all reports."); return

    expected_cols = ['model', 'class', 'precision', 'recall', 'f1-score', 'support', 'training_time', 'inference_time', 'num_params']
    for i in range(len(all_dfs)):
        all_dfs[i] = all_dfs[i].reindex(columns=expected_cols)
        all_dfs[i]['num_params'] = all_dfs[i]['num_params'].astype(float)

    main_df = pd.concat(all_dfs, ignore_index=True)
    
    summary_str, summary_df = generate_summary_tables_str(main_df)
    hyper_str = generate_hyperparameter_table_str(output_dir)
    
    error_str = ""
    stats_str = ""
    if summary_df is not None and not summary_df.empty:
        best_model_name = summary_df.iloc[0]['model']
        error_str = perform_error_analysis(output_dir, results_dir, best_model_name)
        other_model_names = summary_df.iloc[1:]['model'].tolist()
        stats_str = generate_statistical_tests_str(output_dir, best_model_name, other_model_names)

    full_report = "\n".join([summary_str, hyper_str, error_str, stats_str])
    print(full_report)

    report_path = results_dir / "full_analysis_report.md"
    with open(report_path, 'w', encoding='utf-8') as f: f.write(full_report)
    print(f"\nComplete analysis report saved to: {report_path}")

    create_plots_and_csv(main_df, results_dir)
    plot_learning_curves(results_dir)
    
    print("\nDeep analysis complete.")

if __name__ == '__main__':
    main()
