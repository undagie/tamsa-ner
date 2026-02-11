import pandas as pd
import numpy as np
from pathlib import Path
import json
from collections import Counter, defaultdict

_ROOT = Path(__file__).resolve().parent.parent
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import cohen_kappa_score
import re

# Output directory
OUTPUT_DIR = _ROOT / "outputs" / "dataset_documentation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

class DatasetAnalyzer:
    def __init__(self, dataset_name, train_path, dev_path, test_path):
        self.dataset_name = dataset_name
        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
        self.train_data = self.load_bio_file(train_path)
        self.dev_data = self.load_bio_file(dev_path)
        self.test_data = self.load_bio_file(test_path)
        self.all_data = self.train_data + self.dev_data + self.test_data
        
    def load_bio_file(self, file_path):
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
    
    def get_basic_statistics(self):
        """Calculate basic dataset statistics."""
        stats = {
            'dataset_name': self.dataset_name,
            'total_sentences': len(self.all_data),
            'train_sentences': len(self.train_data),
            'dev_sentences': len(self.dev_data),
            'test_sentences': len(self.test_data),
            'total_tokens': sum(len(tokens) for tokens, _ in self.all_data),
            'train_tokens': sum(len(tokens) for tokens, _ in self.train_data),
            'dev_tokens': sum(len(tokens) for tokens, _ in self.dev_data),
            'test_tokens': sum(len(tokens) for tokens, _ in self.test_data),
        }
        
        # Sentence length statistics
        all_lengths = [len(tokens) for tokens, _ in self.all_data]
        stats['avg_sentence_length'] = np.mean(all_lengths)
        stats['std_sentence_length'] = np.std(all_lengths)
        stats['min_sentence_length'] = min(all_lengths)
        stats['max_sentence_length'] = max(all_lengths)
        stats['median_sentence_length'] = np.median(all_lengths)
        
        # Vocabulary statistics
        all_tokens = [token.lower() for tokens, _ in self.all_data for token in tokens]
        stats['vocabulary_size'] = len(set(all_tokens))
        stats['token_type_ratio'] = stats['vocabulary_size'] / len(all_tokens)
        
        return stats
    
    def get_entity_statistics(self):
        """Calculate entity-related statistics."""
        entity_stats = defaultdict(lambda: {'count': 0, 'tokens': 0})
        entity_lengths = defaultdict(list)
        
        for tokens, tags in self.all_data:
            current_entity = None
            current_length = 0
            
            for tag in tags:
                if tag.startswith('B-'):
                    # Save previous entity if exists
                    if current_entity and current_length > 0:
                        entity_lengths[current_entity].append(current_length)
                    
                    # Start new entity
                    current_entity = tag[2:]
                    current_length = 1
                    entity_stats[current_entity]['count'] += 1
                    entity_stats[current_entity]['tokens'] += 1
                    
                elif tag.startswith('I-') and current_entity:
                    current_length += 1
                    entity_stats[current_entity]['tokens'] += 1
                    
                else:  # O tag
                    if current_entity and current_length > 0:
                        entity_lengths[current_entity].append(current_length)
                    current_entity = None
                    current_length = 0
            
            # Don't forget the last entity
            if current_entity and current_length > 0:
                entity_lengths[current_entity].append(current_length)
        
        # Calculate average entity lengths
        for entity_type, lengths in entity_lengths.items():
            entity_stats[entity_type]['avg_length'] = np.mean(lengths)
            entity_stats[entity_type]['max_length'] = max(lengths)
            entity_stats[entity_type]['length_distribution'] = Counter(lengths)
        
        return dict(entity_stats), entity_lengths
    
    def analyze_label_distribution(self):
        """Analyze BIO label distribution."""
        all_tags = [tag for _, tags in self.all_data for tag in tags]
        tag_counts = Counter(all_tags)
        
        # Calculate percentages
        total_tags = len(all_tags)
        tag_percentages = {tag: (count/total_tags)*100 for tag, count in tag_counts.items()}
        
        # Analyze class imbalance
        o_percentage = tag_percentages.get('O', 0)
        entity_percentage = 100 - o_percentage
        
        return tag_counts, tag_percentages, {
            'o_percentage': o_percentage,
            'entity_percentage': entity_percentage,
            'imbalance_ratio': o_percentage / entity_percentage if entity_percentage > 0 else float('inf')
        }
    
    def analyze_annotation_patterns(self):
        """Analyze annotation consistency patterns."""
        patterns = {
            'single_token_entities': 0,
            'multi_token_entities': 0,
            'max_entity_length': 0,
            'avg_entities_per_sentence': 0,
            'sentences_without_entities': 0
        }
        
        entity_counts_per_sentence = []
        
        for tokens, tags in self.all_data:
            entities_in_sentence = 0
            current_length = 0
            
            for tag in tags:
                if tag.startswith('B-'):
                    if current_length == 1:
                        patterns['single_token_entities'] += 1
                    elif current_length > 1:
                        patterns['multi_token_entities'] += 1
                    
                    entities_in_sentence += 1
                    current_length = 1
                    
                elif tag.startswith('I-'):
                    current_length += 1
                    patterns['max_entity_length'] = max(patterns['max_entity_length'], current_length)
                    
                else:
                    if current_length == 1:
                        patterns['single_token_entities'] += 1
                    elif current_length > 1:
                        patterns['multi_token_entities'] += 1
                    current_length = 0
            
            # Handle last entity
            if current_length == 1:
                patterns['single_token_entities'] += 1
            elif current_length > 1:
                patterns['multi_token_entities'] += 1
            
            entity_counts_per_sentence.append(entities_in_sentence)
            if entities_in_sentence == 0:
                patterns['sentences_without_entities'] += 1
        
        patterns['avg_entities_per_sentence'] = np.mean(entity_counts_per_sentence)
        
        # Calculate average sentence length for entity density calculation
        all_sentence_lengths = [len(tokens) for tokens, _ in self.all_data]
        avg_sentence_length = np.mean(all_sentence_lengths) if all_sentence_lengths else 1.0
        patterns['entity_density'] = patterns['avg_entities_per_sentence'] / avg_sentence_length if avg_sentence_length > 0 else 0.0
        
        return patterns
    
    def calculate_inter_annotator_agreement(self, annotations1, annotations2):
        """Calculate Cohen's Kappa for inter-annotator agreement."""
        # Flatten annotations
        flat_anno1 = [tag for _, tags in annotations1 for tag in tags]
        flat_anno2 = [tag for _, tags in annotations2 for tag in tags]
        
        if len(flat_anno1) != len(flat_anno2):
            return None
        
        kappa = cohen_kappa_score(flat_anno1, flat_anno2)
        
        # Calculate agreement by entity type
        entity_kappas = {}
        for entity_type in ['PER', 'LOC', 'ORG', 'MISC']:
            # Convert to binary classification for each entity type
            binary_anno1 = [1 if entity_type in tag else 0 for tag in flat_anno1]
            binary_anno2 = [1 if entity_type in tag else 0 for tag in flat_anno2]
            
            if sum(binary_anno1) > 0 and sum(binary_anno2) > 0:
                entity_kappas[entity_type] = cohen_kappa_score(binary_anno1, binary_anno2)
        
        return kappa, entity_kappas
    
    def analyze_linguistic_characteristics(self):
        """Analyze Indonesian-specific linguistic patterns."""
        patterns = {
            'capitalized_tokens': 0,
            'all_caps_tokens': 0,
            'tokens_with_numbers': 0,
            'punctuation_tokens': 0,
            'reduplicated_words': 0,
            'tokens_with_prefixes': 0,
            'tokens_with_suffixes': 0,
            'average_token_length': 0
        }
        
        # Indonesian affixes
        prefixes = ['me', 'ber', 'ter', 'pe', 'se', 'ke', 'di']
        suffixes = ['kan', 'i', 'an', 'nya', 'lah', 'kah']
        
        all_tokens = []
        
        for tokens, _ in self.all_data:
            for token in tokens:
                all_tokens.append(token)
                
                if token[0].isupper():
                    patterns['capitalized_tokens'] += 1
                
                if token.isupper() and len(token) > 1:
                    patterns['all_caps_tokens'] += 1
                
                if any(char.isdigit() for char in token):
                    patterns['tokens_with_numbers'] += 1
                
                if not token.isalnum():
                    patterns['punctuation_tokens'] += 1
                
                # Check for reduplication (e.g., mata-mata)
                if '-' in token:
                    parts = token.split('-')
                    if len(parts) == 2 and parts[0] == parts[1]:
                        patterns['reduplicated_words'] += 1
                
                # Check affixes
                token_lower = token.lower()
                for prefix in prefixes:
                    if token_lower.startswith(prefix):
                        patterns['tokens_with_prefixes'] += 1
                        break
                
                for suffix in suffixes:
                    if token_lower.endswith(suffix):
                        patterns['tokens_with_suffixes'] += 1
                        break
        
        patterns['average_token_length'] = np.mean([len(token) for token in all_tokens])
        
        # Convert to percentages
        total_tokens = len(all_tokens)
        for key in ['capitalized_tokens', 'all_caps_tokens', 'tokens_with_numbers',
                   'punctuation_tokens', 'reduplicated_words', 'tokens_with_prefixes',
                   'tokens_with_suffixes']:
            patterns[key] = (patterns[key] / total_tokens) * 100
        
        return patterns

def create_dataset_visualizations(all_stats):
    """Create comprehensive dataset visualizations."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Dataset size comparison
    plt.subplot(3, 3, 1)
    datasets = list(all_stats.keys())
    train_sizes = [stats['basic']['train_sentences'] for stats in all_stats.values()]
    dev_sizes = [stats['basic']['dev_sentences'] for stats in all_stats.values()]
    test_sizes = [stats['basic']['test_sentences'] for stats in all_stats.values()]
    
    x = np.arange(len(datasets))
    width = 0.25
    
    plt.bar(x - width, train_sizes, width, label='Train', color='skyblue')
    plt.bar(x, dev_sizes, width, label='Dev', color='orange')
    plt.bar(x + width, test_sizes, width, label='Test', color='green')
    
    plt.xlabel('Dataset')
    plt.ylabel('Number of Sentences')
    plt.title('Dataset Split Sizes')
    plt.xticks(x, datasets)
    plt.legend()
    
    # 2. Entity distribution
    plt.subplot(3, 3, 2)
    entity_data = []
    for dataset, stats in all_stats.items():
        for entity_type, entity_info in stats['entities'].items():
            entity_data.append({
                'dataset': dataset,
                'entity_type': entity_type,
                'count': entity_info['count']
            })
    
    entity_df = pd.DataFrame(entity_data)
    entity_pivot = entity_df.pivot(index='dataset', columns='entity_type', values='count').fillna(0)
    
    entity_pivot.plot(kind='bar', ax=plt.gca())
    plt.xlabel('Dataset')
    plt.ylabel('Entity Count')
    plt.title('Entity Type Distribution')
    plt.xticks(rotation=45)
    plt.legend(title='Entity Type')
    
    # 3. Sentence length distribution
    plt.subplot(3, 3, 3)
    for dataset, stats in all_stats.items():
        analyzer = stats['analyzer']
        lengths = [len(tokens) for tokens, _ in analyzer.all_data]
        plt.hist(lengths, bins=50, alpha=0.5, label=dataset, density=True)
    
    plt.xlabel('Sentence Length (tokens)')
    plt.ylabel('Density')
    plt.title('Sentence Length Distribution')
    plt.legend()
    plt.xlim(0, 100)
    
    # 4. Label imbalance
    plt.subplot(3, 3, 4)
    o_percentages = [stats['label_dist']['imbalance']['o_percentage'] for stats in all_stats.values()]
    entity_percentages = [stats['label_dist']['imbalance']['entity_percentage'] for stats in all_stats.values()]
    
    x = np.arange(len(datasets))
    width = 0.35
    
    plt.bar(x - width/2, o_percentages, width, label='O tags', color='lightcoral')
    plt.bar(x + width/2, entity_percentages, width, label='Entity tags', color='lightgreen')
    
    plt.xlabel('Dataset')
    plt.ylabel('Percentage (%)')
    plt.title('Label Class Distribution')
    plt.xticks(x, datasets)
    plt.legend()
    
    # 5. Average entity length
    plt.subplot(3, 3, 5)
    for dataset, stats in all_stats.items():
        entity_types = []
        avg_lengths = []
        
        for entity_type, entity_info in stats['entities'].items():
            if 'avg_length' in entity_info:
                entity_types.append(entity_type)
                avg_lengths.append(entity_info['avg_length'])
        
        x = np.arange(len(entity_types))
        plt.bar(x + datasets.index(dataset) * 0.25, avg_lengths, 0.25, label=dataset)
    
    plt.xlabel('Entity Type')
    plt.ylabel('Average Length (tokens)')
    plt.title('Average Entity Length by Type')
    plt.xticks(x + 0.25, entity_types)
    plt.legend()
    
    # 6. Linguistic characteristics
    plt.subplot(3, 3, 6)
    ling_features = ['capitalized_tokens', 'tokens_with_numbers', 'reduplicated_words',
                     'tokens_with_prefixes', 'tokens_with_suffixes']
    
    ling_data = []
    for dataset, stats in all_stats.items():
        for feature in ling_features:
            if feature in stats['linguistic']:
                ling_data.append({
                    'dataset': dataset,
                    'feature': feature.replace('_', ' ').title(),
                    'percentage': stats['linguistic'][feature]
                })
    
    ling_df = pd.DataFrame(ling_data)
    ling_pivot = ling_df.pivot(index='feature', columns='dataset', values='percentage')
    
    ling_pivot.plot(kind='bar', ax=plt.gca())
    plt.xlabel('Linguistic Feature')
    plt.ylabel('Percentage (%)')
    plt.title('Linguistic Characteristics')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Dataset')
    
    # 7. Entity density
    plt.subplot(3, 3, 7)
    densities = []
    for dataset, stats in all_stats.items():
        total_entities = sum(info['count'] for info in stats['entities'].values())
        total_tokens = stats['basic']['total_tokens']
        density = (total_entities / total_tokens) * 100
        densities.append(density)
    
    plt.bar(datasets, densities, color='purple', alpha=0.7)
    plt.xlabel('Dataset')
    plt.ylabel('Entity Density (%)')
    plt.title('Entity Density (Entities per 100 tokens)')
    
    for i, v in enumerate(densities):
        plt.text(i, v + 0.1, f'{v:.1f}%', ha='center', va='bottom')
    
    # 8. Vocabulary statistics
    plt.subplot(3, 3, 8)
    vocab_sizes = [stats['basic']['vocabulary_size'] for stats in all_stats.values()]
    token_counts = [stats['basic']['total_tokens'] for stats in all_stats.values()]
    
    plt.scatter(token_counts, vocab_sizes, s=100, alpha=0.7)
    for i, dataset in enumerate(datasets):
        plt.annotate(dataset, (token_counts[i], vocab_sizes[i]),
                    xytext=(5, 5), textcoords='offset points')
    
    plt.xlabel('Total Tokens')
    plt.ylabel('Vocabulary Size')
    plt.title('Vocabulary Growth')
    plt.grid(True, alpha=0.3)
    
    # 9. Split consistency
    plt.subplot(3, 3, 9)
    split_data = []
    for dataset, stats in all_stats.items():
        for split in ['train', 'dev', 'test']:
            split_data.append({
                'dataset': dataset,
                'split': split,
                'percentage': stats['basic'][f'{split}_tokens'] / stats['basic']['total_tokens'] * 100
            })
    
    split_df = pd.DataFrame(split_data)
    split_pivot = split_df.pivot(index='dataset', columns='split', values='percentage')
    
    split_pivot.plot(kind='bar', stacked=True, ax=plt.gca())
    plt.xlabel('Dataset')
    plt.ylabel('Percentage of Total Tokens')
    plt.title('Train/Dev/Test Split Distribution')
    plt.legend(title='Split')
    plt.ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'dataset_analysis_comprehensive.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_dataset_report(all_stats):
    """Generate comprehensive dataset documentation."""
    with open(OUTPUT_DIR / 'dataset_documentation.md', 'w', encoding='utf-8') as f:
        f.write("# Comprehensive Dataset Documentation for Indonesian NER\n\n")
        
        # Executive Summary
        f.write("## Executive Summary\n\n")
        f.write("This document provides detailed documentation and analysis of three Indonesian NER datasets:\n")
        f.write("- **IDNer2k**: News domain dataset with 2,000 sentences\n")
        f.write("- **NER-UGM**: Mixed domain dataset from Universitas Gadjah Mada\n")
        f.write("- **NER-UI**: Wikipedia-based dataset from Universitas Indonesia\n\n")
        
        # Dataset Comparison Table
        f.write("## Dataset Overview\n\n")
        f.write("| Metric | IDNer2k | NER-UGM | NER-UI |\n")
        f.write("|--------|---------|---------|--------|\n")
        
        for metric in ['total_sentences', 'total_tokens', 'vocabulary_size', 
                      'avg_sentence_length', 'entity_percentage']:
            f.write(f"| {metric.replace('_', ' ').title()} | ")
            for dataset in ['idner2k', 'nerugm', 'nerui']:
                if metric == 'entity_percentage':
                    value = all_stats[dataset]['label_dist']['imbalance']['entity_percentage']
                    f.write(f"{value:.1f}% | ")
                elif metric == 'avg_sentence_length':
                    value = all_stats[dataset]['basic'][metric]
                    f.write(f"{value:.1f} | ")
                else:
                    value = all_stats[dataset]['basic'][metric]
                    f.write(f"{value:,} | ")
            f.write("\n")
        
        # Individual Dataset Details
        for dataset_name, stats in all_stats.items():
            f.write(f"\n## {dataset_name.upper()} Dataset\n\n")
            
            # Basic Information
            f.write("### Basic Statistics\n\n")
            basic = stats['basic']
            f.write(f"- **Total Sentences**: {basic['total_sentences']:,} ")
            f.write(f"(Train: {basic['train_sentences']:,}, Dev: {basic['dev_sentences']:,}, Test: {basic['test_sentences']:,})\n")
            f.write(f"- **Total Tokens**: {basic['total_tokens']:,}\n")
            f.write(f"- **Vocabulary Size**: {basic['vocabulary_size']:,}\n")
            f.write(f"- **Type-Token Ratio**: {basic['token_type_ratio']:.4f}\n")
            f.write(f"- **Sentence Length**: {basic['avg_sentence_length']:.1f} ± {basic['std_sentence_length']:.1f} tokens\n")
            f.write(f"- **Length Range**: [{basic['min_sentence_length']}, {basic['max_sentence_length']}]\n\n")
            
            # Entity Statistics
            f.write("### Entity Distribution\n\n")
            f.write("| Entity Type | Count | Tokens | Avg Length | Max Length |\n")
            f.write("|-------------|-------|--------|------------|------------|\n")
            
            for entity_type, info in sorted(stats['entities'].items()):
                f.write(f"| {entity_type} | {info['count']:,} | {info['tokens']:,} | ")
                if 'avg_length' in info:
                    f.write(f"{info['avg_length']:.2f} | {info['max_length']} |\n")
                else:
                    f.write("N/A | N/A |\n")
            
            # Label Distribution
            f.write("\n### Label Distribution\n\n")
            f.write("| Label | Count | Percentage |\n")
            f.write("|-------|-------|------------|\n")
            
            for label, count in sorted(stats['label_dist']['counts'].items(), 
                                      key=lambda x: x[1], reverse=True)[:10]:
                percentage = stats['label_dist']['percentages'][label]
                f.write(f"| {label} | {count:,} | {percentage:.2f}% |\n")
            
            # Annotation Patterns
            f.write("\n### Annotation Patterns\n\n")
            patterns = stats['patterns']
            f.write(f"- **Single-token entities**: {patterns['single_token_entities']:,}\n")
            f.write(f"- **Multi-token entities**: {patterns['multi_token_entities']:,}\n")
            f.write(f"- **Maximum entity length**: {patterns['max_entity_length']} tokens\n")
            f.write(f"- **Average entities per sentence**: {patterns['avg_entities_per_sentence']:.2f}\n")
            f.write(f"- **Sentences without entities**: {patterns['sentences_without_entities']:,} ")
            f.write(f"({patterns['sentences_without_entities']/basic['total_sentences']*100:.1f}%)\n\n")
            
            # Linguistic Characteristics
            f.write("### Linguistic Characteristics\n\n")
            ling = stats['linguistic']
            f.write(f"- **Capitalized tokens**: {ling['capitalized_tokens']:.1f}%\n")
            f.write(f"- **Tokens with numbers**: {ling['tokens_with_numbers']:.1f}%\n")
            f.write(f"- **Reduplicated words**: {ling['reduplicated_words']:.1f}%\n")
            f.write(f"- **Tokens with prefixes**: {ling['tokens_with_prefixes']:.1f}%\n")
            f.write(f"- **Tokens with suffixes**: {ling['tokens_with_suffixes']:.1f}%\n")
            f.write(f"- **Average token length**: {ling['average_token_length']:.1f} characters\n")
        
        # Inter-Annotator Agreement (simulated)
        f.write("\n## Annotation Quality\n\n")
        f.write("### Inter-Annotator Agreement (IAA)\n\n")
        f.write("*Note: Actual IAA scores would require multiple annotators' data*\n\n")
        f.write("Typical IAA scores for well-annotated NER datasets:\n")
        f.write("- **Overall Cohen's Kappa**: 0.85-0.95\n")
        f.write("- **PER (Person)**: 0.90-0.95 (highest agreement)\n")
        f.write("- **LOC (Location)**: 0.85-0.92\n")
        f.write("- **ORG (Organization)**: 0.80-0.88\n")
        f.write("- **MISC (Miscellaneous)**: 0.75-0.85 (lowest agreement)\n\n")
        
        # Annotation Guidelines Summary
        f.write("## Annotation Guidelines Summary\n\n")
        f.write("### Entity Type Definitions\n\n")
        f.write("1. **PER (Person)**\n")
        f.write("   - Individual persons, including fictional characters\n")
        f.write("   - Includes titles when part of name (e.g., 'Presiden Jokowi')\n")
        f.write("   - Groups of people labeled as MISC\n\n")
        
        f.write("2. **LOC (Location)**\n")
        f.write("   - Geographical locations: countries, cities, regions\n")
        f.write("   - Natural features: mountains, rivers, lakes\n")
        f.write("   - Man-made structures when referring to location\n\n")
        
        f.write("3. **ORG (Organization)**\n")
        f.write("   - Companies, institutions, organizations\n")
        f.write("   - Government bodies, political parties\n")
        f.write("   - Sports teams, bands, groups\n\n")
        
        f.write("4. **MISC (Miscellaneous)**\n")
        f.write("   - Nationalities, ethnic groups, religions\n")
        f.write("   - Events, products, works of art\n")
        f.write("   - Other named entities not fitting above categories\n\n")
        
        # Data Quality Issues
        f.write("## Data Quality Observations\n\n")
        f.write("### Strengths\n")
        f.write("- Consistent BIO tagging scheme across all datasets\n")
        f.write("- Good coverage of common Indonesian entity types\n")
        f.write("- Balanced representation of different entity categories\n\n")
        
        f.write("### Potential Issues\n")
        f.write("- High percentage of O tags indicates sparse entity annotation\n")
        f.write("- Some datasets show class imbalance between entity types\n")
        f.write("- Variation in annotation density across datasets suggests different guidelines\n\n")
        
        # Recommendations
        f.write("## Recommendations for Future Work\n\n")
        f.write("1. **Annotation Consistency**: Harmonize annotation guidelines across datasets\n")
        f.write("2. **Domain Diversity**: Include more diverse text sources (social media, legal, medical)\n")
        f.write("3. **Entity Coverage**: Consider adding more entity types (DATE, TIME, MONEY)\n")
        f.write("4. **Quality Control**: Implement systematic IAA measurement\n")
        f.write("5. **Documentation**: Provide detailed annotation manuals with examples\n")

def create_latex_dataset_table(all_stats):
    """Create LaTeX table for paper."""
    with open(OUTPUT_DIR / 'dataset_table.tex', 'w', encoding='utf-8') as f:
        f.write("% Dataset Statistics Table\n")
        f.write("\\begin{table}[ht]\n")
        f.write("\\centering\n")
        f.write("\\caption{Indonesian NER dataset statistics}\n")
        f.write("\\label{tab:dataset_stats}\n")
        f.write("\\begin{tabular}{l|rrr|rrr|r}\n")
        f.write("\\hline\n")
        f.write("\\multirow{2}{*}{Dataset} & \\multicolumn{3}{c|}{Sentences} & ")
        f.write("\\multicolumn{3}{c|}{Entities} & Vocab \\\\\n")
        f.write("& Train & Dev & Test & PER & LOC & ORG & Size \\\\\n")
        f.write("\\hline\n")
        
        for dataset_name, stats in all_stats.items():
            f.write(f"{dataset_name} & ")
            f.write(f"{stats['basic']['train_sentences']:,} & ")
            f.write(f"{stats['basic']['dev_sentences']:,} & ")
            f.write(f"{stats['basic']['test_sentences']:,} & ")
            
            # Entity counts
            for entity in ['PER', 'LOC', 'ORG']:
                count = stats['entities'].get(entity, {}).get('count', 0)
                f.write(f"{count:,} & ")
            
            f.write(f"{stats['basic']['vocabulary_size']:,} \\\\\n")
        
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

def main():
    """Run comprehensive dataset analysis."""
    print("=== Comprehensive Dataset Documentation ===\n")
    
    datasets = {
        'idner2k': {
            'train': _ROOT / "data" / "idner2k" / "train_bio.txt",
            'dev': _ROOT / "data" / "idner2k" / "dev_bio.txt",
            'test': _ROOT / "data" / "idner2k" / "test_bio.txt"
        },
        'nerugm': {
            'train': _ROOT / "data" / "nerugm" / "train_bio.txt",
            'dev': _ROOT / "data" / "nerugm" / "dev_bio.txt",
            'test': _ROOT / "data" / "nerugm" / "test_bio.txt"
        },
        'nerui': {
            'train': _ROOT / "data" / "nerui" / "train_bio.txt",
            'dev': _ROOT / "data" / "nerui" / "dev_bio.txt",
            'test': _ROOT / "data" / "nerui" / "test_bio.txt"
        }
    }
    
    all_stats = {}
    
    for dataset_name, paths in datasets.items():
        print(f"Analyzing {dataset_name}...")
        
        analyzer = DatasetAnalyzer(
            dataset_name,
            paths['train'],
            paths['dev'],
            paths['test']
        )
        
        # Gather all statistics
        stats = {
            'analyzer': analyzer,
            'basic': analyzer.get_basic_statistics(),
            'entities': analyzer.get_entity_statistics()[0],
            'entity_lengths': analyzer.get_entity_statistics()[1],
            'label_dist': {
                'counts': analyzer.analyze_label_distribution()[0],
                'percentages': analyzer.analyze_label_distribution()[1],
                'imbalance': analyzer.analyze_label_distribution()[2]
            },
            'patterns': analyzer.analyze_annotation_patterns(),
            'linguistic': analyzer.analyze_linguistic_characteristics()
        }
        
        all_stats[dataset_name] = stats
        
        # Print summary
        print(f"  - Total sentences: {stats['basic']['total_sentences']:,}")
        print(f"  - Total tokens: {stats['basic']['total_tokens']:,}")
        print(f"  - Entity percentage: {stats['label_dist']['imbalance']['entity_percentage']:.1f}%")
    
    # Save raw statistics
    stats_df = pd.DataFrame([{
        'dataset': name,
        **stats['basic'],
        'total_entities': sum(e['count'] for e in stats['entities'].values()),
        **stats['label_dist']['imbalance']
    } for name, stats in all_stats.items()])
    
    stats_df.to_csv(OUTPUT_DIR / 'dataset_statistics.csv', index=False)
    
    # Create visualizations
    print("\nCreating visualizations...")
    create_dataset_visualizations(all_stats)
    
    # Generate report
    print("Generating documentation...")
    generate_dataset_report(all_stats)
    
    # Create LaTeX table
    create_latex_dataset_table(all_stats)
    
    print(f"\nDataset documentation completed!")
    print(f"Results saved to: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
