import json
import re
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import spacy
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = Path('./outputs/linguistic_error_analysis')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Indonesian NER: prefixes, suffixes, particles, titles, org/loc indicators
INDONESIAN_PATTERNS = {
    'prefixes': ['me', 'ber', 'ter', 'pe', 'se', 'ke', 'di'],
    'suffixes': ['kan', 'i', 'an', 'nya', 'lah', 'kah'],
    'particles': ['yang', 'dan', 'atau', 'dengan', 'untuk', 'pada', 'dari', 'ke', 'di'],
    'titles': ['Bapak', 'Ibu', 'Pak', 'Bu', 'Dr', 'Prof', 'Ir', 'Hj', 'H', 'KH'],
    'common_org_words': ['PT', 'CV', 'Universitas', 'Institut', 'Rumah', 'Sakit', 'Bank', 
                         'Partai', 'Dewan', 'Komisi', 'Kementerian', 'Direktorat'],
    'location_indicators': ['Jalan', 'Jl', 'Kota', 'Kabupaten', 'Provinsi', 'Desa', 
                           'Kelurahan', 'Kecamatan', 'RT', 'RW']
}

class LinguisticErrorAnalyzer:
    def __init__(self):
        self.error_categories = defaultdict(list)
        self.linguistic_features = defaultdict(Counter)
        
    def analyze_error(self, token: str, true_tag: str, pred_tag: str, context: List[str], 
                     context_tags: List[str], position: int):
        """Analyze a single prediction error."""
        error = {
            'token': token,
            'true_tag': true_tag,
            'pred_tag': pred_tag,
            'context': context,
            'position': position,
            'error_type': self._categorize_error(true_tag, pred_tag)
        }
        features = self._extract_linguistic_features(token, context, position)
        error['features'] = features
        error_key = f"{true_tag} -> {pred_tag}"
        self.error_categories[error_key].append(error)
        for feature, value in features.items():
            if value:
                self.linguistic_features[error_key][feature] += 1
        
        return error
    
    def _categorize_error(self, true_tag: str, pred_tag: str) -> str:
        """Categorize error type."""
        if true_tag == 'O' and pred_tag != 'O':
            return 'false_positive'
        elif true_tag != 'O' and pred_tag == 'O':
            return 'false_negative'
        elif true_tag != 'O' and pred_tag != 'O':
            true_type = true_tag.split('-')[1] if '-' in true_tag else true_tag
            pred_type = pred_tag.split('-')[1] if '-' in pred_tag else pred_tag
            
            if true_type != pred_type:
                return 'type_confusion'
            return 'boundary_error'  # B/I tag confusion
        return 'other'
    
    def _extract_linguistic_features(self, token: str, context: List[str], position: int) -> Dict:
        """Extract linguistic features from token and context."""
        features = {}
        features['is_capitalized'] = token[0].isupper() if token else False
        features['is_all_caps'] = token.isupper() if len(token) > 1 else False
        features['contains_number'] = any(char.isdigit() for char in token)
        features['token_length'] = len(token)
        features['is_punctuation'] = not token.isalnum()
        token_lower = token.lower()
        for prefix in INDONESIAN_PATTERNS['prefixes']:
            if token_lower.startswith(prefix):
                features[f'has_prefix_{prefix}'] = True
                break
        for suffix in INDONESIAN_PATTERNS['suffixes']:
            if token_lower.endswith(suffix):
                features[f'has_suffix_{suffix}'] = True
                break
        features['is_particle'] = token_lower in INDONESIAN_PATTERNS['particles']
        features['is_title'] = token in INDONESIAN_PATTERNS['titles']
        if position > 0:
            prev_token = context[position - 1]
            features['prev_is_title'] = prev_token in INDONESIAN_PATTERNS['titles']
            features['prev_is_capitalized'] = prev_token[0].isupper() if prev_token else False
            features['prev_is_particle'] = prev_token.lower() in INDONESIAN_PATTERNS['particles']
        
        if position < len(context) - 1:
            next_token = context[position + 1]
            features['next_is_capitalized'] = next_token[0].isupper() if next_token else False
            features['next_is_particle'] = next_token.lower() in INDONESIAN_PATTERNS['particles']
        features['position_in_sentence'] = position / len(context) if context else 0
        features['is_sentence_start'] = position == 0
        features['is_sentence_end'] = position == len(context) - 1
        features['contains_org_word'] = any(org in token for org in INDONESIAN_PATTERNS['common_org_words'])
        features['contains_loc_indicator'] = any(loc in token for loc in INDONESIAN_PATTERNS['location_indicators'])
        features['is_reduplicated'] = '-' in token and len(token.split('-')) == 2 and token.split('-')[0] == token.split('-')[1]  # Indonesian reduplication
        
        return features
    
    def generate_report(self) -> Dict:
        """Generate comprehensive error analysis report."""
        report = {
            'error_distribution': {},
            'linguistic_patterns': {},
            'error_examples': {},
            'recommendations': []
        }
        for error_type, errors in self.error_categories.items():
            report['error_distribution'][error_type] = len(errors)
            if error_type in self.linguistic_features:
                top_features = self.linguistic_features[error_type].most_common(10)
                report['linguistic_patterns'][error_type] = top_features
            report['error_examples'][error_type] = errors[:5]
        report['recommendations'] = self._generate_recommendations()
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on error patterns."""
        recommendations = []
        title_errors = sum(1 for error_type in self.linguistic_features 
                          for feature, count in self.linguistic_features[error_type].items() 
                          if 'title' in feature and count > 10)
        if title_errors > 0:
            recommendations.append(
                "High error rate with Indonesian titles (Pak, Bu, etc.). "
                "Consider adding title-aware features or pre-processing."
            )
        morph_errors = sum(1 for error_type in self.linguistic_features 
                          for feature, count in self.linguistic_features[error_type].items() 
                          if ('prefix' in feature or 'suffix' in feature) and count > 20)
        if morph_errors > 0:
            recommendations.append(
                "Significant errors with morphologically complex words. "
                "Consider adding morphological analyzers or subword tokenization."
            )
        boundary_errors = sum(len(errors) for error_type, errors in self.error_categories.items() 
                             if any(errors) and errors[0]['error_type'] == 'boundary_error')
        if boundary_errors > 50:
            recommendations.append(
                "High boundary detection errors. "
                "Consider using stronger sequence labeling constraints or post-processing."
            )
        
        return recommendations

def analyze_model_errors(model_name: str, predictions_file: Path) -> Dict:
    """Analyze errors from a single model."""
    analyzer = LinguisticErrorAnalyzer()
    tokens, true_tags, pred_tags = [], [], []
    current_tokens, current_true, current_pred = [], [], []
    
    with open(predictions_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                if current_tokens:
                    tokens.append(current_tokens)
                    true_tags.append(current_true)
                    pred_tags.append(current_pred)
                    current_tokens, current_true, current_pred = [], [], []
            else:
                parts = line.split('\t')
                if len(parts) == 3:
                    current_tokens.append(parts[0])
                    current_true.append(parts[1])
                    current_pred.append(parts[2])
    total_errors = 0
    for sent_tokens, sent_true, sent_pred in zip(tokens, true_tags, pred_tags):
        for i, (token, true_tag, pred_tag) in enumerate(zip(sent_tokens, sent_true, sent_pred)):
            if true_tag != pred_tag:
                analyzer.analyze_error(token, true_tag, pred_tag, sent_tokens, sent_true, i)
                total_errors += 1
    report = analyzer.generate_report()
    report['model'] = model_name
    report['total_errors'] = total_errors
    
    return report

def create_error_visualizations(reports: List[Dict]):
    """Create comprehensive visualizations for error analysis."""
    plt.figure(figsize=(15, 8))
    
    error_data = []
    for report in reports:
        for error_type, count in report['error_distribution'].items():
            error_data.append({
                'model': report['model'],
                'error_type': error_type,
                'count': count
            })
    
    error_df = pd.DataFrame(error_data)
    if not error_df.empty:
        pivot_df = error_df.pivot(index='model', columns='error_type', values='count').fillna(0)
        
        plt.subplot(2, 2, 1)
        pivot_df.plot(kind='bar', stacked=True, ax=plt.gca())
        plt.title('Error Distribution by Type')
        plt.xlabel('Model')
        plt.ylabel('Number of Errors')
        plt.xticks(rotation=45, ha='right')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 2. Most common error transitions
    plt.subplot(2, 2, 2)
    all_transitions = Counter()
    for report in reports:
        for error_type, count in report['error_distribution'].items():
            all_transitions[error_type] += count
    
    top_transitions = all_transitions.most_common(10)
    if top_transitions:
        transitions, counts = zip(*top_transitions)
        plt.barh(range(len(transitions)), counts)
        plt.yticks(range(len(transitions)), transitions)
        plt.xlabel('Count')
        plt.title('Top 10 Error Transitions')
        plt.gca().invert_yaxis()
    
    # 3. Linguistic feature correlation with errors
    plt.subplot(2, 2, 3)
    feature_importance = defaultdict(int)
    for report in reports:
        for error_type, features in report['linguistic_patterns'].items():
            for feature, count in features:
                feature_importance[feature] += count
    
    if feature_importance:
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        features, counts = zip(*top_features)
        plt.barh(range(len(features)), counts)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Occurrence in Errors')
        plt.title('Top Linguistic Features in Errors')
        plt.gca().invert_yaxis()
    
    # 4. Model-specific error patterns
    plt.subplot(2, 2, 4)
    model_error_rates = []
    for report in reports:
        total = report['total_errors']
        if total > 0:
            error_types = defaultdict(int)
            for error_type, count in report['error_distribution'].items():
                true_type = error_type.split(' -> ')[0]
                if true_type != 'O' and '-' in true_type:
                    entity = true_type.split('-')[1]
                    error_types[entity] += count
            
            for entity, count in error_types.items():
                model_error_rates.append({
                    'model': report['model'],
                    'entity': entity,
                    'error_rate': count / total
                })
    
    if model_error_rates:
        mer_df = pd.DataFrame(model_error_rates)
        pivot_mer = mer_df.pivot(index='model', columns='entity', values='error_rate').fillna(0)
        pivot_mer.plot(kind='bar', ax=plt.gca())
        plt.xlabel('Model')
        plt.ylabel('Error Rate')
        plt.title('Entity-specific Error Rates')
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Entity Type')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'linguistic_error_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def analyze_cross_lingual_patterns(reports: List[Dict]):
    """Analyze patterns specific to Indonesian vs multilingual models."""
    indonesian_models = ['bilstm', 'bilstm_w2v', 'bilstm_w2v_cnn', 'indobert_bilstm']
    multilingual_models = ['mbert_bilstm', 'xlm_roberta_bilstm']
    
    indo_patterns = defaultdict(int)
    multi_patterns = defaultdict(int)
    
    for report in reports:
        patterns = report.get('linguistic_patterns', {})
        for error_type, features in patterns.items():
            for feature, count in features:
                if report['model'] in indonesian_models:
                    indo_patterns[feature] += count
                elif report['model'] in multilingual_models:
                    multi_patterns[feature] += count
    
    indo_specific = set(indo_patterns.keys()) - set(multi_patterns.keys())
    multi_specific = set(multi_patterns.keys()) - set(indo_patterns.keys())
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    if indo_specific:
        indo_specific_counts = [(f, indo_patterns[f]) for f in indo_specific]
        indo_specific_counts.sort(key=lambda x: x[1], reverse=True)
        features, counts = zip(*indo_specific_counts[:10])
        plt.barh(range(len(features)), counts, color='green')
        plt.yticks(range(len(features)), features)
        plt.xlabel('Count')
        plt.title('Indonesian Model-Specific Error Patterns')
        plt.gca().invert_yaxis()
    plt.subplot(1, 2, 2)
    if multi_specific:
        multi_specific_counts = [(f, multi_patterns[f]) for f in multi_specific]
        multi_specific_counts.sort(key=lambda x: x[1], reverse=True)
        features, counts = zip(*multi_specific_counts[:10])
        plt.barh(range(len(features)), counts, color='blue')
        plt.yticks(range(len(features)), features)
        plt.xlabel('Count')
        plt.title('Multilingual Model-Specific Error Patterns')
        plt.gca().invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'cross_lingual_patterns.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_detailed_error_report(reports: List[Dict]):
    """Generate detailed error report with examples and recommendations."""
    with open(OUTPUT_DIR / 'detailed_error_analysis.md', 'w', encoding='utf-8') as f:
        f.write("# Detailed Linguistic Error Analysis for Indonesian NER\n\n")
        f.write("## Executive Summary\n\n")
        total_errors = sum(r['total_errors'] for r in reports)
        f.write(f"- Total errors analyzed: {total_errors:,}\n")
        f.write(f"- Models analyzed: {len(reports)}\n")
        all_error_types = defaultdict(int)
        for report in reports:
            for error_type, count in report['error_distribution'].items():
                all_error_types[error_type] += count
        
        f.write("\n### Most Common Error Types:\n")
        for error_type, count in sorted(all_error_types.items(), key=lambda x: x[1], reverse=True)[:10]:
            percentage = (count / total_errors) * 100
            f.write(f"- {error_type}: {count:,} ({percentage:.2f}%)\n")
        f.write("\n## Model-Specific Analysis\n\n")
        for report in reports:
            f.write(f"### {report['model']}\n\n")
            f.write(f"Total errors: {report['total_errors']:,}\n\n")
            f.write("#### Top Error Patterns:\n")
            for error_type, count in sorted(report['error_distribution'].items(), 
                                          key=lambda x: x[1], reverse=True)[:5]:
                f.write(f"- {error_type}: {count}\n")
                if error_type in report['linguistic_patterns']:
                    f.write("  - Associated features:\n")
                    for feature, feat_count in report['linguistic_patterns'][error_type][:3]:
                        f.write(f"    - {feature}: {feat_count}\n")
            f.write("\n#### Example Errors:\n")
            shown_types = set()
            for error_type, examples in report['error_examples'].items():
                if len(shown_types) >= 3:
                    break
                shown_types.add(error_type)
                f.write(f"\n**{error_type}:**\n")
                for ex in examples[:2]:
                    context_start = max(0, ex['position'] - 2)
                    context_end = min(len(ex['context']), ex['position'] + 3)
                    context_str = ' '.join(ex['context'][context_start:context_end])
                    
                    f.write(f"- Token: '{ex['token']}'\n")
                    f.write(f"  - Context: ...{context_str}...\n")
                    f.write(f"  - Features: {', '.join(k for k, v in ex['features'].items() if v and isinstance(v, bool))}\n")
            
            f.write("\n")
        f.write("## Cross-Lingual Insights\n\n")
        indo_models = ['bilstm', 'bilstm_w2v', 'bilstm_w2v_cnn', 'indobert_bilstm']
        multi_models = ['mbert_bilstm', 'xlm_roberta_bilstm']
        
        indo_errors = sum(r['total_errors'] for r in reports if r['model'] in indo_models)
        multi_errors = sum(r['total_errors'] for r in reports if r['model'] in multi_models)
        
        if indo_errors > 0 and multi_errors > 0:
            f.write(f"- Indonesian-specific models: {indo_errors:,} total errors\n")
            f.write(f"- Multilingual models: {multi_errors:,} total errors\n\n")
        f.write("## Recommendations\n\n")
        all_recommendations = set()
        for report in reports:
            all_recommendations.update(report.get('recommendations', []))
        
        for i, rec in enumerate(all_recommendations, 1):
            f.write(f"{i}. {rec}\n\n")
        f.write("## Additional Insights\n\n")
        f.write("### Indonesian-Specific Challenges:\n\n")
        f.write("1. **Morphological Complexity**: Indonesian affixation creates challenges in boundary detection\n")
        f.write("2. **Title Handling**: Frequent errors with Indonesian titles (Pak, Bu, etc.)\n")
        f.write("3. **Reduplication**: Reduplicated forms (e.g., 'mata-mata') often misclassified\n")
        f.write("4. **Code-mixing**: Indonesian text often contains English terms, causing confusion\n\n")

def main():
    """Run comprehensive linguistic error analysis."""
    print("=== Linguistic Error Analysis for Indonesian NER ===\n")
    prediction_files = []
    models = ['bilstm', 'bilstm_w2v', 'bilstm_w2v_cnn', 'indobert', 'indobert_bilstm',
              'mbert_bilstm', 'xlm_roberta_bilstm', 'attention_fusion']
    for model in models:
        exp_pred = Path(f'./outputs/experiment_{model}/test_predictions.txt')
        if exp_pred.exists():
            prediction_files.append((model, exp_pred))
        for dataset in ['nerui', 'nerugm']:
            eval_pred = Path(f'./outputs/evaluation_{dataset}_{model}/test_predictions.txt')
            if eval_pred.exists():
                prediction_files.append((f'{model}_{dataset}', eval_pred))
    
    if not prediction_files:
        print("No prediction files found. Please run model training/evaluation first.")
        return
    
    print(f"Found {len(prediction_files)} prediction files to analyze.\n")
    reports = []
    for model_name, pred_file in prediction_files:
        print(f"Analyzing {model_name}...")
        report = analyze_model_errors(model_name, pred_file)
        reports.append(report)
        print(f"  Found {report['total_errors']} errors")
    print("\nCreating visualizations...")
    create_error_visualizations(reports)
    analyze_cross_lingual_patterns(reports)
    print("Generating detailed report...")
    generate_detailed_error_report(reports)
    summary_data = []
    for report in reports:
        summary_data.append({
            'model': report['model'],
            'total_errors': report['total_errors'],
            'unique_error_types': len(report['error_distribution']),
            'most_common_error': max(report['error_distribution'].items(), 
                                    key=lambda x: x[1])[0] if report['error_distribution'] else 'N/A'
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(OUTPUT_DIR / 'error_summary.csv', index=False)
    
    print(f"\nLinguistic error analysis completed!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print(f"- Visualizations: linguistic_error_analysis.png, cross_lingual_patterns.png")
    print(f"- Detailed report: detailed_error_analysis.md")
    print(f"- Summary statistics: error_summary.csv")

if __name__ == '__main__':
    main()
