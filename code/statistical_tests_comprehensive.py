import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

OUTPUT_BASE_DIR = Path('./outputs')
MULTIPLE_RUNS_DIR = OUTPUT_BASE_DIR / 'multiple_runs'
STATS_OUTPUT_DIR = Path('./outputs/statistical_tests')
STATS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Plot style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 12,
    'figure.dpi': 300
})


def load_multiple_runs_data(output_dir):
    """Load data from multiple runs."""
    all_data = {}
    
    # Try to load from aggregated summary first
    summary_path = MULTIPLE_RUNS_DIR / 'all_runs_summary.json'
    if summary_path.exists():
        with open(summary_path, 'r', encoding='utf-8') as f:
            summary_data = json.load(f)
        
        for model_data in summary_data:
            model_name = model_data['model']
            metrics = model_data.get('metrics', {})
            
            # Extract F1 scores from all runs
            if 'weighted_f1' in metrics:
                f1_values = metrics['weighted_f1'].get('values', [])
                if f1_values:
                    all_data[model_name] = np.array(f1_values)
        
        return all_data
    
    # Fallback: load from individual model summaries
    for exp_dir in output_dir.glob('experiment_*'):
        summary_path = exp_dir / 'multiple_runs_summary.json'
        if summary_path.exists():
            with open(summary_path, 'r', encoding='utf-8') as f:
                model_data = json.load(f)
            
            model_name = model_data['model']
            metrics = model_data.get('metrics', {})
            
            if 'weighted_f1' in metrics:
                f1_values = metrics['weighted_f1'].get('values', [])
                if f1_values:
                    all_data[model_name] = np.array(f1_values)
    
    return all_data


def load_per_entity_data(output_dir):
    """Load per-entity data from multiple runs."""
    per_entity_data = {}  # {entity_type: {model_name: [scores]}}
    
    # Try to load from aggregated summary first
    summary_path = MULTIPLE_RUNS_DIR / 'all_runs_summary.json'
    if summary_path.exists():
        with open(summary_path, 'r', encoding='utf-8') as f:
            summary_data = json.load(f)
        
        # Extract all entity types
        entity_types = set()
        for model_data in summary_data:
            metrics = model_data.get('metrics', {})
            for key in metrics.keys():
                if key.endswith('_f1') and key != 'weighted_f1' and key != 'macro_f1' and key != 'micro_f1':
                    entity_type = key.replace('_f1', '')
                    entity_types.add(entity_type)
        
        # Load data for each entity type
        for entity_type in entity_types:
            per_entity_data[entity_type] = {}
            for model_data in summary_data:
                model_name = model_data['model']
                metrics = model_data.get('metrics', {})
                metric_key = f'{entity_type}_f1'
                if metric_key in metrics:
                    f1_values = metrics[metric_key].get('values', [])
                    if f1_values:
                        per_entity_data[entity_type][model_name] = np.array(f1_values)
    
    # Fallback: load from individual model summaries
    if not per_entity_data:
        for exp_dir in output_dir.glob('experiment_*'):
            summary_path = exp_dir / 'multiple_runs_summary.json'
            if summary_path.exists():
                with open(summary_path, 'r', encoding='utf-8') as f:
                    model_data = json.load(f)
                
                model_name = model_data['model']
                metrics = model_data.get('metrics', {})
                
                for key in metrics.keys():
                    if key.endswith('_f1') and key != 'weighted_f1' and key != 'macro_f1' and key != 'micro_f1':
                        entity_type = key.replace('_f1', '')
                        if entity_type not in per_entity_data:
                            per_entity_data[entity_type] = {}
                        f1_values = metrics[key].get('values', [])
                        if f1_values:
                            per_entity_data[entity_type][model_name] = np.array(f1_values)
    
    return per_entity_data


def paired_ttest(model1_scores, model2_scores):
    """Paired t-test."""
    if len(model1_scores) != len(model2_scores):
        return None
    
    if len(model1_scores) < 2:
        return None
    
    # Calculate differences
    differences = model1_scores - model2_scores
    
    # Perform paired t-test
    t_stat, p_value = stats.ttest_rel(model1_scores, model2_scores)
    
    # Degrees of freedom
    df = len(model1_scores) - 1
    
    # Mean difference
    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)
    
    return {
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'degrees_of_freedom': int(df),
        'mean_difference': float(mean_diff),
        'std_difference': float(std_diff)
    }


def wilcoxon_test(model1_scores, model2_scores):
    """Wilcoxon signed-rank test."""
    if len(model1_scores) != len(model2_scores):
        return None
    
    if len(model1_scores) < 3:  # Wilcoxon needs at least 3 pairs
        return None
    
    # Calculate differences
    differences = model1_scores - model2_scores
    
    # Remove zero differences
    differences = differences[differences != 0]
    
    if len(differences) == 0:
        return None
    
    # Perform Wilcoxon test
    try:
        w_stat, p_value = stats.wilcoxon(model1_scores, model2_scores)
        return {
            'w_statistic': float(w_stat),
            'p_value': float(p_value),
            'n_pairs': len(differences)
        }
    except:
        return None


def cohens_d(model1_scores, model2_scores):
    """Calculate Cohen's d effect size."""
    if len(model1_scores) < 2 or len(model2_scores) < 2:
        return None
    
    # Calculate means
    mean1 = np.mean(model1_scores)
    mean2 = np.mean(model2_scores)
    
    # Calculate pooled standard deviation
    std1 = np.std(model1_scores, ddof=1)
    std2 = np.std(model2_scores, ddof=1)
    n1 = len(model1_scores)
    n2 = len(model2_scores)
    
    # Pooled std
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return None
    
    # Cohen's d
    d = (mean1 - mean2) / pooled_std
    
    # Interpret effect size
    abs_d = abs(d)
    if abs_d < 0.2:
        interpretation = 'negligible'
    elif abs_d < 0.5:
        interpretation = 'small'
    elif abs_d < 0.8:
        interpretation = 'medium'
    else:
        interpretation = 'large'
    
    return {
        'cohens_d': float(d),
        'interpretation': interpretation,
        'mean1': float(mean1),
        'mean2': float(mean2),
        'pooled_std': float(pooled_std)
    }


def bonferroni_correction(p_values, alpha=0.05):
    """Apply Bonferroni correction."""
    n_comparisons = len(p_values)
    if n_comparisons == 0:
        return {
            'alpha_original': alpha,
            'alpha_adjusted': alpha,
            'n_comparisons': 0,
            'adjusted_p_values': []
        }
    
    alpha_adjusted = alpha / n_comparisons
    adjusted_p_values = [min(p * n_comparisons, 1.0) for p in p_values]
    
    return {
        'alpha_original': alpha,
        'alpha_adjusted': alpha_adjusted,
        'n_comparisons': n_comparisons,
        'adjusted_p_values': adjusted_p_values
    }


def escape_latex(text):
    """Escape special LaTeX characters."""
    if not isinstance(text, str):
        text = str(text)
    
    # Order matters: escape backslash first
    replacements = [
        ('\\', r'\textbackslash{}'),
        ('&', r'\&'),
        ('%', r'\%'),
        ('$', r'\$'),
        ('#', r'\#'),
        ('^', r'\^{}'),
        ('_', r'\_'),
        ('{', r'\{'),
        ('}', r'\}'),
        ('~', r'\~{}'),
    ]
    
    for char, escaped in replacements:
        text = text.replace(char, escaped)
    
    return text


def generate_latex_table_overall(all_tests, bonferroni_info):
    """Generate LaTeX table for overall statistical tests."""
    lines = []
    
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Statistical Significance Tests: Overall F1 Score}")
    lines.append("\\label{tab:statistical_tests_overall}")
    lines.append("\\begin{tabular}{lcccccc}")
    lines.append("\\toprule")
    lines.append("Model 1 & Model 2 & Mean Diff & t-stat & p-value & p-adj & Cohen's d & Significant \\\\")
    lines.append("\\midrule")
    
    for test in all_tests:
        model1 = escape_latex(test['model1'])
        model2 = escape_latex(test['model2'])
        ttest = test.get('ttest', {})
        cohen = test.get('cohens_d', {})
        p_adj = test.get('p_value_adjusted', 1.0)
        
        mean_diff = ttest.get('mean_difference', 0) if ttest else 0
        t_stat = ttest.get('t_statistic', 0) if ttest else 0
        p_val = ttest.get('p_value', 1.0) if ttest else 1.0
        d_val = cohen.get('cohens_d', 0) if cohen else 0
        
        # Significance
        significant = "Yes" if p_adj < bonferroni_info['alpha_adjusted'] else "No"
        
        lines.append(f"{model1} & {model2} & {mean_diff:.4f} & {t_stat:.4f} & {p_val:.4f} & {p_adj:.4f} & {d_val:.4f} & {significant} \\\\")
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    
    return "\n".join(lines)


def generate_latex_table_per_entity(per_entity_tests, bonferroni_info):
    """Generate LaTeX table for per-entity statistical tests."""
    lines = []
    
    # Group by entity type
    entity_types = sorted(set([t['entity_type'] for t in per_entity_tests]))
    
    for entity_type in entity_types:
        entity_tests = [t for t in per_entity_tests if t['entity_type'] == entity_type]
        
        # Calculate entity-specific bonferroni
        entity_p_values = [t.get('ttest', {}).get('p_value', 1.0) for t in entity_tests if 'ttest' in t]
        entity_bonferroni = bonferroni_correction(entity_p_values, alpha=0.05) if entity_p_values else bonferroni_info
        
        # Escape entity type name for LaTeX
        entity_type_escaped = escape_latex(entity_type)
        
        lines.append(f"\\begin{{table}}[htbp]")
        lines.append(f"\\centering")
        lines.append(f"\\caption{{Statistical Significance Tests: {entity_type_escaped} Entity F1 Score}}")
        lines.append(f"\\label{{tab:statistical_tests_{entity_type.lower().replace('-', '_')}}}")
        lines.append(f"\\begin{{tabular}}{{lcccccc}}")
        lines.append(f"\\toprule")
        lines.append(f"Model 1 & Model 2 & Mean Diff & t-stat & p-value & p-adj & Cohen's d & Significant \\\\")
        lines.append(f"\\midrule")
        
        for test in entity_tests:
            model1 = escape_latex(test['model1'])
            model2 = escape_latex(test['model2'])
            ttest = test.get('ttest', {})
            cohen = test.get('cohens_d', {})
            p_adj = test.get('p_value_adjusted', 1.0)
            
            mean_diff = ttest.get('mean_difference', 0) if ttest else 0
            t_stat = ttest.get('t_statistic', 0) if ttest else 0
            p_val = ttest.get('p_value', 1.0) if ttest else 1.0
            d_val = cohen.get('cohens_d', 0) if cohen else 0
            
            # Significance (use entity-specific bonferroni)
            significant = "Yes" if p_adj < entity_bonferroni['alpha_adjusted'] else "No"
            
            lines.append(f"{model1} & {model2} & {mean_diff:.4f} & {t_stat:.4f} & {p_val:.4f} & {p_adj:.4f} & {d_val:.4f} & {significant} \\\\")
        
        lines.append(f"\\bottomrule")
        lines.append(f"\\end{{tabular}}")
        lines.append(f"\\end{{table}}")
        lines.append("")  # Empty line between tables
    
    return "\n".join(lines)


def generate_comprehensive_report(all_tests, bonferroni_info):
    """Generate markdown report."""
    report_lines = []
    
    report_lines.append("# Comprehensive Statistical Tests Report\n")
    report_lines.append("Generated for Multiple Runs Analysis\n")
    report_lines.append(f"Number of comparisons: {len(all_tests)}\n")
    report_lines.append(f"Bonferroni adjusted α: {bonferroni_info['alpha_adjusted']:.6f}\n")
    report_lines.append(f"Original α: {bonferroni_info['alpha_original']}\n")
    report_lines.append("\n" + "="*80 + "\n")
    
    # Summary table
    report_lines.append("## Summary Table\n\n")
    report_lines.append("| Model 1 | Model 2 | Mean Diff | t-stat | p-value | ")
    report_lines.append("p-adjusted | W-stat | p-wilcoxon | Cohen's d | Effect | Significant |\n")
    report_lines.append("|---------|---------|-----------|---------|---------|")
    report_lines.append("------------|---------|------------|----------|--------|-------------|\n")
    
    for i, test in enumerate(all_tests):
        model1 = test['model1']
        model2 = test['model2']
        ttest = test.get('ttest', {})
        wilcoxon = test.get('wilcoxon', {})
        cohen = test.get('cohens_d', {})
        p_adj = test.get('p_value_adjusted', 1.0)
        
        mean_diff = ttest.get('mean_difference', 0) if ttest else 0
        t_stat = ttest.get('t_statistic', 0) if ttest else 0
        p_val = ttest.get('p_value', 1.0) if ttest else 1.0
        w_stat = wilcoxon.get('w_statistic', 0) if wilcoxon else 0
        p_wil = wilcoxon.get('p_value', 1.0) if wilcoxon else 1.0
        d_val = cohen.get('cohens_d', 0) if cohen else 0
        effect = cohen.get('interpretation', 'N/A') if cohen else 'N/A'
        
        # Significance
        significant = "Yes" if p_adj < bonferroni_info['alpha_adjusted'] else "No"
        
        report_lines.append(f"| {model1} | {model2} | {mean_diff:.4f} | ")
        report_lines.append(f"{t_stat:.4f} | {p_val:.6f} | {p_adj:.6f} | ")
        report_lines.append(f"{w_stat:.2f} | {p_wil:.6f} | {d_val:.4f} | ")
        report_lines.append(f"{effect} | {significant} |\n")
    
    report_lines.append("\n" + "="*80 + "\n")
    
    # Detailed results
    report_lines.append("## Detailed Results\n\n")
    
    for test in all_tests:
        model1 = test['model1']
        model2 = test['model2']
        
        report_lines.append(f"### {model1} vs {model2}\n\n")
        
        # T-test results
        if test.get('ttest'):
            ttest = test['ttest']
            report_lines.append("**Paired t-test:**\n")
            report_lines.append(f"- t-statistic: {ttest['t_statistic']:.4f}\n")
            report_lines.append(f"- p-value: {ttest['p_value']:.6f}\n")
            report_lines.append(f"- degrees of freedom: {ttest['degrees_of_freedom']}\n")
            report_lines.append(f"- mean difference: {ttest['mean_difference']:.4f} ± {ttest['std_difference']:.4f}\n")
            report_lines.append(f"- adjusted p-value: {test.get('p_value_adjusted', 1.0):.6f}\n\n")
        
        # Wilcoxon results
        if test.get('wilcoxon'):
            wilcoxon = test['wilcoxon']
            report_lines.append("**Wilcoxon signed-rank test:**\n")
            report_lines.append(f"- W-statistic: {wilcoxon['w_statistic']:.4f}\n")
            report_lines.append(f"- p-value: {wilcoxon['p_value']:.6f}\n")
            report_lines.append(f"- number of pairs: {wilcoxon['n_pairs']}\n\n")
        
        # Effect size
        if test.get('cohens_d'):
            cohen = test['cohens_d']
            report_lines.append("**Effect Size (Cohen's d):**\n")
            report_lines.append(f"- Cohen's d: {cohen['cohens_d']:.4f}\n")
            report_lines.append(f"- Interpretation: {cohen['interpretation']}\n")
            report_lines.append(f"- Mean {model1}: {cohen['mean1']:.4f}\n")
            report_lines.append(f"- Mean {model2}: {cohen['mean2']:.4f}\n\n")
        
        report_lines.append("---\n\n")
    
    # Interpretation guide
    report_lines.append("## Interpretation Guide\n\n")
    report_lines.append("### Significance Levels\n")
    report_lines.append("- **p < adjusted α**: Statistically significant after Bonferroni correction\n")
    report_lines.append("- **p < 0.05**: Statistically significant (before correction)\n")
    report_lines.append("- **p ≥ 0.05**: Not statistically significant\n\n")
    
    report_lines.append("### Effect Size (Cohen's d)\n")
    report_lines.append("- **|d| < 0.2**: Negligible effect\n")
    report_lines.append("- **0.2 ≤ |d| < 0.5**: Small effect\n")
    report_lines.append("- **0.5 ≤ |d| < 0.8**: Medium effect\n")
    report_lines.append("- **|d| ≥ 0.8**: Large effect\n\n")
    
    report_lines.append("### Test Selection\n")
    report_lines.append("- **Paired t-test**: Assumes normal distribution of differences\n")
    report_lines.append("- **Wilcoxon test**: Non-parametric, no distribution assumptions\n")
    report_lines.append("- Use Wilcoxon if data is not normally distributed\n\n")
    
    return "\n".join(report_lines)


def create_visualization(all_tests, bonferroni_info):
    """Create visualization heatmap of p-values and effect sizes."""
    if not all_tests:
        return
    
    # Extract model names
    models = sorted(set([t['model1'] for t in all_tests] + [t['model2'] for t in all_tests]))
    n_models = len(models)
    
    # Create matrices for p-values and effect sizes
    p_value_matrix = np.full((n_models, n_models), np.nan)
    effect_size_matrix = np.full((n_models, n_models), np.nan)
    
    model_to_idx = {model: i for i, model in enumerate(models)}
    
    for test in all_tests:
        idx1 = model_to_idx[test['model1']]
        idx2 = model_to_idx[test['model2']]
        
        # P-value (adjusted)
        p_adj = test.get('p_value_adjusted', np.nan)
        p_value_matrix[idx1, idx2] = p_adj
        
        # Effect size
        if test.get('cohens_d'):
            d_val = test['cohens_d']['cohens_d']
            effect_size_matrix[idx1, idx2] = d_val
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('Statistical Tests: P-values and Effect Sizes', 
                 fontsize=18, fontweight='bold')
    
    # Plot 1: P-values heatmap
    ax1 = axes[0]
    mask = np.isnan(p_value_matrix)
    sns.heatmap(p_value_matrix, annot=True, fmt='.4f', cmap='RdYlGn_r',
                mask=mask, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                xticklabels=models, yticklabels=models, ax=ax1, vmin=0, vmax=1)
    ax1.set_title('Adjusted P-values (Bonferroni)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Model 2', fontsize=12)
    ax1.set_ylabel('Model 1', fontsize=12)
    
    # Plot 2: Effect sizes heatmap
    ax2 = axes[1]
    mask = np.isnan(effect_size_matrix)
    sns.heatmap(effect_size_matrix, annot=True, fmt='.3f', cmap='RdBu_r',
                mask=mask, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                center=0, xticklabels=models, yticklabels=models, ax=ax2)
    ax2.set_title("Cohen's d Effect Sizes", fontsize=14, fontweight='bold')
    ax2.set_xlabel('Model 2', fontsize=12)
    ax2.set_ylabel('Model 1', fontsize=12)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = STATS_OUTPUT_DIR / 'statistical_tests_visualization.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved visualization to: {plot_path}")


def main():
    """Main function."""
    print("\n" + "="*60)
    print("Comprehensive Statistical Tests")
    print("="*60)
    
    # Load multiple runs data
    print("\nLoading multiple runs data...")
    all_data = load_multiple_runs_data(OUTPUT_BASE_DIR)
    
    if not all_data:
        print("ERROR: No multiple runs data found!")
        print("Please run multiple_runs_analysis.py first.")
        return
    
    print(f"Found data for {len(all_data)} models:")
    for model_name, scores in all_data.items():
        print(f"  {model_name}: {len(scores)} runs")
    
    # Perform statistical tests for all model pairs
    print("\nPerforming statistical tests for overall F1...")
    all_tests = []
    p_values = []
    
    model_names = list(all_data.keys())
    for model1, model2 in combinations(model_names, 2):
        scores1 = all_data[model1]
        scores2 = all_data[model2]
        
        # Ensure same length (should be same if same number of runs)
        min_len = min(len(scores1), len(scores2))
        if min_len < 2:
            continue
        
        scores1 = scores1[:min_len]
        scores2 = scores2[:min_len]
        
        test_result = {
            'model1': model1,
            'model2': model2
        }
        
        # Paired t-test
        ttest_result = paired_ttest(scores1, scores2)
        if ttest_result:
            test_result['ttest'] = ttest_result
            p_values.append(ttest_result['p_value'])
        
        # Wilcoxon test
        wilcoxon_result = wilcoxon_test(scores1, scores2)
        if wilcoxon_result:
            test_result['wilcoxon'] = wilcoxon_result
        
        # Cohen's d
        cohen_result = cohens_d(scores1, scores2)
        if cohen_result:
            test_result['cohens_d'] = cohen_result
        
        all_tests.append(test_result)
    
    # Apply Bonferroni correction
    print(f"\nApplying Bonferroni correction for {len(p_values)} comparisons...")
    bonferroni_info = bonferroni_correction(p_values, alpha=0.05)
    
    # Add adjusted p-values to tests
    p_idx = 0
    for test in all_tests:
        if 'ttest' in test:
            test['p_value_adjusted'] = bonferroni_info['adjusted_p_values'][p_idx]
            p_idx += 1
    
    # Load per-entity data and perform tests
    print("\nLoading per-entity data...")
    per_entity_data = load_per_entity_data(OUTPUT_BASE_DIR)
    
    per_entity_tests = []
    if per_entity_data:
        print(f"Found data for {len(per_entity_data)} entity types:")
        for entity_type, models_data in per_entity_data.items():
            print(f"  {entity_type}: {len(models_data)} models")
        
        print("\nPerforming statistical tests per entity...")
        for entity_type, models_data in per_entity_data.items():
            entity_p_values = []
            entity_model_names = list(models_data.keys())
            
            for model1, model2 in combinations(entity_model_names, 2):
                if model1 not in models_data or model2 not in models_data:
                    continue
                
                scores1 = models_data[model1]
                scores2 = models_data[model2]
                
                min_len = min(len(scores1), len(scores2))
                if min_len < 2:
                    continue
                
                scores1 = scores1[:min_len]
                scores2 = scores2[:min_len]
                
                test_result = {
                    'entity_type': entity_type,
                    'model1': model1,
                    'model2': model2
                }
                
                # Paired t-test
                ttest_result = paired_ttest(scores1, scores2)
                if ttest_result:
                    test_result['ttest'] = ttest_result
                    entity_p_values.append(ttest_result['p_value'])
                
                # Wilcoxon test
                wilcoxon_result = wilcoxon_test(scores1, scores2)
                if wilcoxon_result:
                    test_result['wilcoxon'] = wilcoxon_result
                
                # Cohen's d
                cohen_result = cohens_d(scores1, scores2)
                if cohen_result:
                    test_result['cohens_d'] = cohen_result
                
                per_entity_tests.append(test_result)
            
            # Apply Bonferroni correction per entity
            # Only process tests for current entity_type to avoid index mismatch
            if entity_p_values:
                entity_bonferroni = bonferroni_correction(entity_p_values, alpha=0.05)
                p_idx = 0
                # Filter tests for current entity_type only
                current_entity_tests = [t for t in per_entity_tests 
                                       if t['entity_type'] == entity_type and 'ttest' in t]
                for test in current_entity_tests:
                    test['p_value_adjusted'] = entity_bonferroni['adjusted_p_values'][p_idx]
                    p_idx += 1
    
    # Generate report
    print("Generating comprehensive report...")
    report = generate_comprehensive_report(all_tests, bonferroni_info)
    
    report_path = STATS_OUTPUT_DIR / 'statistical_tests_report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"Saved report to: {report_path}")
    
    # Add per-entity section to report
    if per_entity_tests:
        print("Generating per-entity report...")
        entity_report_lines = []
        entity_report_lines.append("\n\n" + "="*80 + "\n")
        entity_report_lines.append("# Per-Entity Statistical Tests\n\n")
        
        # Group by entity type
        entity_types = sorted(set([t['entity_type'] for t in per_entity_tests]))
        for entity_type in entity_types:
            entity_report_lines.append(f"## {entity_type} Entity\n\n")
            entity_tests = [t for t in per_entity_tests if t['entity_type'] == entity_type]
            
            entity_report_lines.append("| Model 1 | Model 2 | Mean Diff | t-stat | p-value | ")
            entity_report_lines.append("p-adjusted | W-stat | p-wilcoxon | Cohen's d | Effect | Significant |\n")
            entity_report_lines.append("|---------|---------|-----------|---------|---------|")
            entity_report_lines.append("------------|---------|------------|----------|--------|-------------|\n")
            
            for test in entity_tests:
                model1 = test['model1']
                model2 = test['model2']
                ttest = test.get('ttest', {})
                wilcoxon = test.get('wilcoxon', {})
                cohen = test.get('cohens_d', {})
                p_adj = test.get('p_value_adjusted', 1.0)
                
                mean_diff = ttest.get('mean_difference', 0) if ttest else 0
                t_stat = ttest.get('t_statistic', 0) if ttest else 0
                p_val = ttest.get('p_value', 1.0) if ttest else 1.0
                w_stat = wilcoxon.get('w_statistic', 0) if wilcoxon else 0
                p_wil = wilcoxon.get('p_value', 1.0) if wilcoxon else 1.0
                d_val = cohen.get('cohens_d', 0) if cohen else 0
                effect = cohen.get('interpretation', 'N/A') if cohen else 'N/A'
                
                # Significance (use entity-specific bonferroni)
                entity_p_vals = [t.get('ttest', {}).get('p_value', 1.0) for t in entity_tests if 'ttest' in t]
                if entity_p_vals:
                    entity_bonferroni = bonferroni_correction(entity_p_vals, alpha=0.05)
                    significant = "Yes" if p_adj < entity_bonferroni['alpha_adjusted'] else "No"
                else:
                    significant = "N/A"
                
                entity_report_lines.append(f"| {model1} | {model2} | {mean_diff:.4f} | ")
                entity_report_lines.append(f"{t_stat:.4f} | {p_val:.6f} | {p_adj:.6f} | ")
                entity_report_lines.append(f"{w_stat:.2f} | {p_wil:.6f} | {d_val:.4f} | ")
                entity_report_lines.append(f"{effect} | {significant} |\n")
            
            entity_report_lines.append("\n")
        
        with open(report_path, 'a', encoding='utf-8') as f:
            f.write("\n".join(entity_report_lines))
        print(f"Added per-entity results to report")
    
    # Save results as CSV
    results_data = []
    for test in all_tests:
        row = {
            'model1': test['model1'],
            'model2': test['model2'],
            'metric': 'overall_f1'
        }
        
        if 'ttest' in test:
            ttest = test['ttest']
            row['mean_difference'] = ttest['mean_difference']
            row['t_statistic'] = ttest['t_statistic']
            row['p_value'] = ttest['p_value']
            row['p_value_adjusted'] = test.get('p_value_adjusted', np.nan)
            row['degrees_of_freedom'] = ttest['degrees_of_freedom']
        
        if 'wilcoxon' in test:
            wilcoxon = test['wilcoxon']
            row['w_statistic'] = wilcoxon['w_statistic']
            row['p_value_wilcoxon'] = wilcoxon['p_value']
        
        if 'cohens_d' in test:
            cohen = test['cohens_d']
            row['cohens_d'] = cohen['cohens_d']
            row['effect_size_interpretation'] = cohen['interpretation']
        
        results_data.append(row)
    
    # Add per-entity results
    for test in per_entity_tests:
        row = {
            'model1': test['model1'],
            'model2': test['model2'],
            'metric': f"{test['entity_type']}_f1"
        }
        
        if 'ttest' in test:
            ttest = test['ttest']
            row['mean_difference'] = ttest['mean_difference']
            row['t_statistic'] = ttest['t_statistic']
            row['p_value'] = ttest['p_value']
            row['p_value_adjusted'] = test.get('p_value_adjusted', np.nan)
            row['degrees_of_freedom'] = ttest['degrees_of_freedom']
        
        if 'wilcoxon' in test:
            wilcoxon = test['wilcoxon']
            row['w_statistic'] = wilcoxon['w_statistic']
            row['p_value_wilcoxon'] = wilcoxon['p_value']
        
        if 'cohens_d' in test:
            cohen = test['cohens_d']
            row['cohens_d'] = cohen['cohens_d']
            row['effect_size_interpretation'] = cohen['interpretation']
        
        results_data.append(row)
    
    results_df = pd.DataFrame(results_data)
    results_path = STATS_OUTPUT_DIR / 'statistical_tests_results.csv'
    results_df.to_csv(results_path, index=False)
    print(f"Saved results table to: {results_path}")
    
    # Generate LaTeX tables
    print("Generating LaTeX tables...")
    latex_overall = generate_latex_table_overall(all_tests, bonferroni_info)
    latex_path_overall = STATS_OUTPUT_DIR / 'statistical_tests_overall.tex'
    with open(latex_path_overall, 'w', encoding='utf-8') as f:
        f.write(latex_overall)
    print(f"Saved LaTeX table (overall) to: {latex_path_overall}")
    
    if per_entity_tests:
        latex_per_entity = generate_latex_table_per_entity(per_entity_tests, bonferroni_info)
        latex_path_per_entity = STATS_OUTPUT_DIR / 'statistical_tests_per_entity.tex'
        with open(latex_path_per_entity, 'w', encoding='utf-8') as f:
            f.write(latex_per_entity)
        print(f"Saved LaTeX table (per-entity) to: {latex_path_per_entity}")
    
    # Create visualization
    print("Creating visualizations...")
    create_visualization(all_tests, bonferroni_info)
    
    print("\n" + "="*60)
    print("Statistical Tests Completed!")
    print("="*60)
    print(f"\nResults saved to: {STATS_OUTPUT_DIR}")


if __name__ == '__main__':
    main()

