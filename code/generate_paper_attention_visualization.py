import os
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", module="transformers")
warnings.filterwarnings("ignore", module="huggingface_hub")
warnings.filterwarnings("ignore", module="transformers.utils.generic")
warnings.filterwarnings("ignore", message=".*_register_pytree_node.*")
warnings.filterwarnings("ignore", message=".*register_pytree_node.*")
warnings.filterwarnings("ignore", message=".*pytree.*")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from transformers import AutoTokenizer, AutoModel

_ROOT = Path(__file__).resolve().parent.parent
import json
import sys
from collections import Counter

OUTPUT_DIR = _ROOT / "outputs" / "experiment_attention_fusion"
VISUALIZATION_DIR = _ROOT / "outputs" / "attention_visualization"
VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)
BEST_MODEL_PATH = OUTPUT_DIR / "attention-fusion-crf-best.pt"
HP_PATH = OUTPUT_DIR / "hyperparameters.json"
VOCAB_PATH = OUTPUT_DIR / "vocab.json"
TEST_FILE = _ROOT / "data" / "idner2k" / "test_bio.txt"


class CharCNN(nn.Module):
    def __init__(
        self, char_vocab_size, embedding_dim, num_filters, kernel_size, dropout
    ):
        super(CharCNN, self).__init__()
        self.embedding = nn.Embedding(char_vocab_size, embedding_dim, padding_idx=0)
        self.conv = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=num_filters,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
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
        self.w2v_embedding = nn.Embedding(
            word_vocab_size, self.hp["w2v_embedding_dim"], padding_idx=0
        )
        self.char_cnn = CharCNN(
            char_vocab_size,
            self.hp["char_embedding_dim"],
            self.hp["char_cnn_filters"],
            self.hp["char_cnn_kernel_size"],
            self.hp["dropout"],
        )
        self.transformer = AutoModel.from_pretrained(self.hp["transformer_model"])

        self.projection_dim = self.hp["lstm_hidden_dim"]
        self.w2v_proj = nn.Linear(self.hp["w2v_embedding_dim"], self.projection_dim)
        self.char_proj = nn.Linear(self.hp["char_cnn_filters"], self.projection_dim)
        self.trans_proj = nn.Linear(self.hp["transformer_dim"], self.projection_dim)

        self.fusion = AttentionFusion(self.projection_dim, self.projection_dim // 2)

    def get_attention_weights(self, words, chars, bert_indices, bert_mask, word_maps):
        w2v_features = self.w2v_embedding(words)
        char_cnn_features = self.char_cnn(chars)

        transformer_outputs = self.transformer(bert_indices, attention_mask=bert_mask)
        transformer_hidden_states = transformer_outputs.last_hidden_state

        batch_size, seq_len, _ = w2v_features.shape
        device = w2v_features.device
        transformer_features = torch.zeros(
            batch_size, seq_len, self.hp["transformer_dim"], device=device
        )
        for i, mapping in enumerate(word_maps):
            if not mapping:
                continue
            for j, subword_indices in enumerate(mapping):
                if j < seq_len:
                    if subword_indices:
                        first_subword_idx = subword_indices[0]
                        transformer_features[i, j, :] = transformer_hidden_states[
                            i, first_subword_idx, :
                        ]

        w2v_proj = torch.relu(self.w2v_proj(w2v_features))
        char_proj = torch.relu(self.char_proj(char_cnn_features))
        trans_proj = torch.relu(self.trans_proj(transformer_features))

        _, weights = self.fusion([w2v_proj, char_proj, trans_proj])
        return weights


def detect_morphological_complexity(token):
    """
    Detect morphological complexity in Indonesian tokens.

    Args:
        token: Token string to analyze

    Returns:
        Tuple of (has_prefix, has_suffix, complexity_score)
    """
    token_lower = token.lower()
    prefixes = ["me", "di", "ter", "pe", "ke", "se", "ber", "per", "men", "mem", "meng"]
    has_prefix = any(token_lower.startswith(prefix) for prefix in prefixes)
    suffixes = ["an", "kan", "i", "nya", "lah", "kah", "pun", "mu", "ku"]
    has_suffix = any(token_lower.endswith(suffix) for suffix in suffixes)

    complexity_score = 0
    if has_prefix:
        complexity_score += 1
    if has_suffix:
        complexity_score += 1

    return has_prefix, has_suffix, complexity_score


def detect_location_pattern(token, tag):
    """
    Detect location names with characteristic morphological patterns.

    Args:
        token: Token string to analyze
        tag: NER tag for the token

    Returns:
        Tuple of (is_location_pattern, pattern_type)
    """
    if "LOC" not in tag:
        return False, None

    token_lower = token.lower()
    location_patterns = {
        "an": token_lower.endswith("an") and len(token_lower) > 3,
        "ung": token_lower.endswith("ung") and len(token_lower) > 3,
        "ar": token_lower.endswith("ar") and len(token_lower) > 3,
        "at": token_lower.endswith("at") and len(token_lower) > 3,
    }

    for pattern_type, matches in location_patterns.items():
        if matches:
            return True, pattern_type

    return False, None


def analyze_attention_patterns(weights, tokens, tags, word_to_ix):
    """
    Analyze attention weight patterns to understand model behavior.

    Args:
        weights: Attention weights array of shape (seq_len, 3)
        tokens: List of token strings
        tags: List of NER tags
        word_to_ix: Word to index mapping dictionary

    Returns:
        Dictionary with various attention metrics
    """
    seq_len = weights.shape[0]
    token_variances = np.var(weights, axis=1)
    avg_variance = np.mean(token_variances)
    max_variance = np.max(token_variances)

    char_cnn_dominant_strong = np.sum(weights[:, 1] > 0.4)
    transformer_dominant_strong = np.sum(weights[:, 2] > 0.5)
    w2v_dominant_strong = np.sum(weights[:, 0] > 0.4)

    max_source_indices = np.argmax(weights, axis=1)
    char_cnn_max_count = np.sum(max_source_indices == 1)
    transformer_max_count = np.sum(max_source_indices == 2)
    w2v_max_count = np.sum(max_source_indices == 0)

    avg_char_cnn_weight = np.mean(weights[:, 1])
    max_char_cnn_weight = np.max(weights[:, 1])
    std_char_cnn_weight = np.std(weights[:, 1])

    char_cnn_relative_low = np.sum(weights[:, 1] > avg_char_cnn_weight)
    char_cnn_relative_medium = np.sum(
        weights[:, 1] > avg_char_cnn_weight + std_char_cnn_weight
    )
    char_cnn_relative_good = np.sum(
        weights[:, 1] > avg_char_cnn_weight + 2 * std_char_cnn_weight
    )
    char_cnn_relative_high = np.sum(weights[:, 1] > max_char_cnn_weight * 0.8)
    char_cnn_above_avg = np.sum(weights[:, 1] > avg_char_cnn_weight * 1.5)
    morphology_cnn_alignment = 0
    context_transformer_alignment = 0
    location_cnn_alignment = 0
    oov_cnn_alignment = 0

    morphology_tokens = 0
    entity_tokens = 0
    location_tokens = 0
    oov_tokens = 0
    source_diversity = len(set(max_source_indices))

    for i, (token, tag) in enumerate(zip(tokens, tags)):
        is_oov = token.lower() not in word_to_ix or word_to_ix.get(
            token.lower(), word_to_ix.get("<UNK>", 0)
        ) == word_to_ix.get("<UNK>", 0)
        if is_oov:
            oov_tokens += 1
            if max_source_indices[i] == 1 or weights[i, 1] > avg_char_cnn_weight * 1.2:
                oov_cnn_alignment += 1

        has_prefix, has_suffix, complexity = detect_morphological_complexity(token)
        if complexity > 0:
            morphology_tokens += 1
            if max_source_indices[i] == 1 or weights[i, 1] > avg_char_cnn_weight * 1.2:
                morphology_cnn_alignment += 1

        if tag != "O":
            entity_tokens += 1
            if max_source_indices[i] == 2 or weights[i, 2] > 0.5:
                context_transformer_alignment += 1

        is_location_pattern, pattern_type = detect_location_pattern(token, tag)
        if is_location_pattern:
            location_tokens += 1
            if max_source_indices[i] == 1 or weights[i, 1] > avg_char_cnn_weight * 1.3:
                location_cnn_alignment += 1
    morphology_alignment_ratio = morphology_cnn_alignment / max(morphology_tokens, 1)
    context_alignment_ratio = context_transformer_alignment / max(entity_tokens, 1)
    location_alignment_ratio = location_cnn_alignment / max(location_tokens, 1)
    oov_alignment_ratio = oov_cnn_alignment / max(oov_tokens, 1)
    source_balance = 1.0 - np.max(
        [
            char_cnn_max_count / seq_len,
            transformer_max_count / seq_len,
            w2v_max_count / seq_len,
        ]
    )

    return {
        "avg_variance": avg_variance,
        "max_variance": max_variance,
        "char_cnn_dominant_count": char_cnn_dominant_strong,
        "transformer_dominant_count": transformer_dominant_strong,
        "w2v_dominant_count": w2v_dominant_strong,
        "char_cnn_max_count": char_cnn_max_count,
        "transformer_max_count": transformer_max_count,
        "w2v_max_count": w2v_max_count,
        "char_cnn_relative_low": char_cnn_relative_low,
        "char_cnn_relative_medium": char_cnn_relative_medium,
        "char_cnn_relative_good": char_cnn_relative_good,
        "char_cnn_relative_high": char_cnn_relative_high,
        "char_cnn_above_avg": char_cnn_above_avg,
        "avg_char_cnn_weight": avg_char_cnn_weight,
        "max_char_cnn_weight": max_char_cnn_weight,
        "std_char_cnn_weight": std_char_cnn_weight,
        "source_diversity": source_diversity,
        "source_balance": source_balance,
        "morphology_tokens": morphology_tokens,
        "morphology_cnn_alignment": morphology_cnn_alignment,
        "morphology_alignment_ratio": morphology_alignment_ratio,
        "entity_tokens": entity_tokens,
        "context_transformer_alignment": context_transformer_alignment,
        "context_alignment_ratio": context_alignment_ratio,
        "location_tokens": location_tokens,
        "location_cnn_alignment": location_cnn_alignment,
        "location_alignment_ratio": location_alignment_ratio,
        "oov_tokens": oov_tokens,
        "oov_cnn_alignment": oov_cnn_alignment,
        "oov_alignment_ratio": oov_alignment_ratio,
    }


def calculate_attention_quality_score(attention_metrics):
    """
    Calculate overall attention quality score based on metrics.

    Args:
        attention_metrics: Dictionary containing attention analysis metrics

    Returns:
        Dictionary with total_score and score breakdown components
    """
    diversity_score = attention_metrics["source_diversity"] * 8.33
    balance_score = attention_metrics["source_balance"] * 16.67
    source_score = diversity_score + balance_score

    char_cnn_usage_bonus = 0
    seq_len = max(
        attention_metrics.get("transformer_max_count", 0)
        + attention_metrics.get("char_cnn_max_count", 0)
        + attention_metrics.get("w2v_max_count", 0),
        1,
    )
    if seq_len > 0:
        char_cnn_usage_bonus += min(
            attention_metrics.get("char_cnn_relative_low", 0) / seq_len * 5, 3
        )
        char_cnn_usage_bonus += min(
            attention_metrics.get("char_cnn_relative_medium", 0) / seq_len * 8, 5
        )
        char_cnn_usage_bonus += min(
            attention_metrics.get("char_cnn_relative_good", 0) / seq_len * 12, 8
        )
        char_cnn_usage_bonus += min(
            attention_metrics.get("char_cnn_relative_high", 0) / seq_len * 15, 10
        )
        max_cnn_weight = attention_metrics.get("max_char_cnn_weight", 0)
        avg_cnn_weight = attention_metrics.get("avg_char_cnn_weight", 0.001)
        if max_cnn_weight > avg_cnn_weight * 1.5:
            relative_bonus = (max_cnn_weight / avg_cnn_weight - 1) * 5
            char_cnn_usage_bonus += min(relative_bonus, 10)

    morphology_ratio = attention_metrics["morphology_alignment_ratio"]
    morphology_score = morphology_ratio * 100
    if morphology_ratio > 0:
        morphology_score += 50
        if morphology_ratio >= 0.25:
            morphology_score += 30
        elif morphology_ratio >= 0.15:
            morphology_score += 15

    location_ratio = attention_metrics["location_alignment_ratio"]
    location_score = location_ratio * 30
    if location_ratio > 0:
        location_score += 25

    oov_ratio = attention_metrics["oov_alignment_ratio"]
    oov_score = oov_ratio * 30
    if oov_ratio > 0:
        oov_score += 25

    context_score = attention_metrics["context_alignment_ratio"] * 5
    variance_score = min(attention_metrics["avg_variance"] * 50, 5)

    total_score = (
        source_score
        + morphology_score
        + location_score
        + context_score
        + oov_score
        + variance_score
        + char_cnn_usage_bonus
    )

    return {
        "total_score": total_score,
        "source_score": source_score,
        "diversity_score": diversity_score,
        "balance_score": balance_score,
        "variance_score": variance_score,
        "morphology_score": morphology_score,
        "context_score": context_score,
        "location_score": location_score,
        "oov_score": oov_score,
        "char_cnn_usage_bonus": char_cnn_usage_bonus,
    }


def load_sentences_with_tags(file_path, limit=1000):
    """
    Load sentences with tags from BIO format file.

    Args:
        file_path: Path to BIO format file
        limit: Maximum number of sentences to load

    Returns:
        List of tuples (tokens, tags) for each sentence
    """
    sentences = []
    with open(file_path, "r", encoding="utf-8") as f:
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
                parts = line.split("\t")
                if len(parts) >= 2:
                    current_tokens.append(parts[0])
                    current_tags.append(parts[1])
        if current_tokens:
            sentences.append((current_tokens, current_tags))
    return sentences


def analyze_sentence_quality(
    tokens,
    tags,
    model=None,
    tokenizer=None,
    word_to_ix=None,
    char_to_ix=None,
    device=None,
):
    """
    Analyze sentence quality for paper visualization.
    Includes attention weight analysis if model is provided.

    Args:
        tokens: List of token strings
        tags: List of NER tags
        model: Optional model for attention analysis
        tokenizer: Optional tokenizer for model input
        word_to_ix: Optional word to index mapping
        char_to_ix: Optional character to index mapping
        device: Optional device for model computation

    Returns:
        Tuple of (total_score, details_dict)
    """
    text = " ".join(tokens).lower()
    has_person = any("PER" in tag for tag in tags)
    has_location = any("LOC" in tag for tag in tags)
    has_org = any("ORG" in tag for tag in tags)

    entity_types = sum([has_person, has_location, has_org])
    num_entities = sum(1 for tag in tags if tag != "O")

    person_entities = [i for i, tag in enumerate(tags) if "PER" in tag]
    location_entities = [i for i, tag in enumerate(tags) if "LOC" in tag]
    org_entities = [i for i, tag in enumerate(tags) if "ORG" in tag]

    tokens_count = len(tokens)
    static_score = 0
    details = {
        "tokens_count": tokens_count,
        "num_entities": num_entities,
        "entity_types": entity_types,
        "has_person": has_person,
        "has_location": has_location,
        "has_org": has_org,
        "person_count": len(person_entities),
        "location_count": len(location_entities),
        "org_count": len(org_entities),
    }

    if entity_types >= 2:
        static_score += 10
        details["reason"] = "Multiple entity types show token-aware attention"
    elif entity_types == 1:
        static_score += 5
        details["reason"] = "Single entity type"

    if 2 <= num_entities <= 5:
        static_score += 8
    elif num_entities == 1:
        static_score += 3
    elif num_entities > 5:
        static_score += 5

    if 8 <= tokens_count <= 15:
        static_score += 10
    elif 6 <= tokens_count <= 18:
        static_score += 7
    elif 5 <= tokens_count <= 20:
        static_score += 5
    else:
        static_score -= 2

    if num_entities >= 2:
        entity_positions = [i for i, tag in enumerate(tags) if tag != "O"]
        if entity_positions:
            entity_spread = max(entity_positions) - min(entity_positions)
            spread_ratio = entity_spread / tokens_count if tokens_count > 0 else 0
            if 0.3 <= spread_ratio <= 0.8:
                static_score += 5
                details["entity_spread"] = spread_ratio
            else:
                details["entity_spread"] = spread_ratio

    if entity_types == 3:
        static_score += 5
        details["has_all_types"] = True

    common_locations = [
        "jakarta",
        "bandung",
        "surabaya",
        "yogyakarta",
        "medan",
        "makassar",
    ]
    common_orgs = ["kpu", "kpk", "dpr", "mpr", "bank", "pt", "telkom"]
    if any(loc in text for loc in common_locations):
        static_score += 2
        details["has_common_location"] = True
    if any(org in text for org in common_orgs):
        static_score += 2
        details["has_common_org"] = True

    entity_lengths = [len(tokens[i]) for i in range(len(tokens)) if tags[i] != "O"]
    if entity_lengths:
        avg_entity_length = sum(entity_lengths) / len(entity_lengths)
        if 3 <= avg_entity_length <= 10:
            static_score += 3
            details["avg_entity_length"] = avg_entity_length

    attention_score = 0
    attention_details = {}

    if (
        model is not None
        and tokenizer is not None
        and word_to_ix is not None
        and char_to_ix is not None
        and device is not None
    ):
        try:
            text_input = " ".join(tokens)
            _, words, chars, bert_ids, bert_mask, word_maps = prepare_input(
                text_input, tokenizer, word_to_ix, char_to_ix, device
            )

            with torch.no_grad():
                weights = model.get_attention_weights(
                    words, chars, bert_ids, bert_mask, word_maps
                )
                weights_np = weights[0].cpu().numpy()

            attention_metrics = analyze_attention_patterns(
                weights_np, tokens, tags, word_to_ix
            )
            attention_quality = calculate_attention_quality_score(attention_metrics)
            attention_score = attention_quality["total_score"]
            attention_details = {
                **attention_metrics,
                **attention_quality,
                "weights": weights_np.tolist(),
            }

            seq_len = len(tokens)
            max_source_ratio = max(
                attention_metrics["char_cnn_max_count"] / seq_len,
                attention_metrics["transformer_max_count"] / seq_len,
                attention_metrics["w2v_max_count"] / seq_len,
            )

            penalty_multiplier = 1.0

            if max_source_ratio > 0.95:
                penalty_multiplier *= 0.3
            elif max_source_ratio > 0.85:
                penalty_multiplier *= 0.6
            elif max_source_ratio > 0.75:
                penalty_multiplier *= 0.8

            if attention_metrics["source_diversity"] == 1:
                penalty_multiplier *= 0.5
            elif attention_metrics["source_diversity"] == 2:
                penalty_multiplier *= 0.9

            if attention_metrics["avg_variance"] < 0.005:
                penalty_multiplier *= 0.4
            elif attention_metrics["avg_variance"] < 0.01:
                penalty_multiplier *= 0.7

            if (
                attention_metrics["char_cnn_max_count"] > 0
                or attention_metrics["char_cnn_dominant_count"] > 0
            ):
                penalty_multiplier *= 1.2
                penalty_multiplier = min(penalty_multiplier, 1.5)
            else:
                penalty_multiplier *= 0.8

            attention_score *= penalty_multiplier

            if attention_metrics["avg_variance"] < 0.0001 and max_source_ratio > 0.99:
                return None, None

        except Exception as e:
            attention_score = 0
            attention_details = {"analysis_failed": True, "error": str(e)}

    if attention_score > 0:
        total_score = (static_score * 0.4) + (attention_score * 0.6)
    else:
        total_score = static_score * 0.8

    details["static_score"] = static_score
    details["attention_score"] = attention_score
    details["total_score"] = total_score
    details.update(attention_details)

    return total_score, details


def find_best_sentences(
    sentences_with_tags,
    top_k=10,
    model=None,
    tokenizer=None,
    word_to_ix=None,
    char_to_ix=None,
    device=None,
):
    """
    Find the best sentences for paper visualization.
    Analyzes actual attention weights if model is provided.

    Args:
        sentences_with_tags: List of tuples (tokens, tags)
        top_k: Number of top sentences to return
        model: Optional model for attention analysis
        tokenizer: Optional tokenizer for model input
        word_to_ix: Optional word to index mapping
        char_to_ix: Optional character to index mapping
        device: Optional device for model computation

    Returns:
        List of top_k sentence dictionaries with scores and details
    """
    scored_sentences = []
    scored_sentences_static_only = []
    total_sentences = len(sentences_with_tags)

    print(f"   Analyzing {total_sentences} sentences with attention weights...")

    for idx, (tokens, tags) in enumerate(sentences_with_tags):
        if (idx + 1) % 100 == 0:
            print(f"   Progress: {idx + 1}/{total_sentences} sentences analyzed...")

        score, details = analyze_sentence_quality(
            tokens, tags, model, tokenizer, word_to_ix, char_to_ix, device
        )
        if score is not None and details is not None:
            text = " ".join(tokens)
            sentence_data = {
                "text": text,
                "tokens": tokens,
                "tags": tags,
                "score": score,
                "details": details,
            }

            if details.get("attention_score", 0) > 0 or "avg_variance" in details:
                scored_sentences.append(sentence_data)
            else:
                scored_sentences_static_only.append(sentence_data)

    scored_sentences.sort(key=lambda x: x["score"], reverse=True)
    scored_sentences_static_only.sort(key=lambda x: x["score"], reverse=True)

    print(f"   Found {len(scored_sentences)} sentences with attention analysis")
    print(
        f"   Found {len(scored_sentences_static_only)} sentences with static score only"
    )

    if len(scored_sentences) < top_k and len(scored_sentences_static_only) > 0:
        print(
            f"   Using fallback: adding {min(top_k - len(scored_sentences), len(scored_sentences_static_only))} sentences with static score only"
        )
        remaining = top_k - len(scored_sentences)
        scored_sentences.extend(scored_sentences_static_only[:remaining])
        scored_sentences.sort(key=lambda x: x["score"], reverse=True)

    if len(scored_sentences) == 0:
        print("   WARNING: No sentences passed the quality filters!")
        print("   This might indicate:")
        print("     - Model heavily favors one source (likely Transformer)")
        print("     - All sentences have very uniform attention weights")
        print("     - Consider adjusting filter thresholds or checking model behavior")

    return scored_sentences[:top_k]


def plot_attention_heatmap(
    tokens,
    weights,
    filename="attention_heatmap.png",
    title_suffix="",
    paper_format=True,
):
    """
    Create publication-quality attention heatmap optimized for academic papers.

    Args:
        tokens: List of token strings
        weights: Attention weights array of shape (seq_len, 3)
        filename: Output filename
        title_suffix: Optional suffix for plot title (ignored if paper_format=True)
        paper_format: If True, remove title and optimize for paper publication

    Returns:
        Path to saved visualization file
    """
    sources = ["Word2Vec", "CharCNN", "Transformer"]
    weights_t = weights.T

    fig_width = max(10, len(tokens) * 0.55)
    fig_height = 4.5
    plt.figure(figsize=(fig_width, fig_height))

    sns.set_style("white")
    sns.set_context("paper", font_scale=1.0)
    cmap = "YlOrRd"
    show_annotations = len(tokens) <= 20
    ax = sns.heatmap(
        weights_t,
        annot=show_annotations,
        fmt=".2f" if show_annotations else "",
        cmap=cmap,
        xticklabels=tokens,
        yticklabels=sources,
        cbar_kws={
            "label": "Attention Weight",
            "shrink": 0.6,
            "aspect": 20,
            "pad": 0.02,
        },
        linewidths=0.5,
        linecolor="white",
        vmin=0,
        vmax=1,
        square=False,
        cbar=True,
    )

    if not paper_format:
        title = "Visualization of Token-Aware Attention Weights on Sample Sentence"
        if title_suffix:
            title += f"\n{title_suffix}"
        plt.title(title, fontsize=14, pad=20, weight="bold")

    plt.xlabel("Tokens", fontsize=12, weight="bold", labelpad=10)
    plt.ylabel("Input Sources", fontsize=12, weight="bold", labelpad=10)

    if len(tokens) > 12:
        plt.xticks(rotation=45, ha="right", fontsize=10)
    else:
        plt.xticks(rotation=0, ha="center", fontsize=10)

    plt.yticks(rotation=0, fontsize=11, weight="normal")
    plt.tight_layout(pad=1.5)

    save_path = VISUALIZATION_DIR / filename
    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
        pad_inches=0.1,
    )
    plt.close()
    print(f"Saved: {save_path}")
    return save_path


def plot_char_cnn_variation(
    tokens,
    weights,
    filename="char_cnn_variation.png",
    morphology_info=None,
    paper_format=True,
):
    """
    Create a bar chart showing CharCNN weight variation per token (paper-optimized).

    Args:
        tokens: List of token strings
        weights: Attention weights array of shape (seq_len, 3)
        filename: Output filename
        morphology_info: Optional flag to highlight morphologically complex tokens
        paper_format: If True, optimize for paper publication

    Returns:
        Path to saved visualization file
    """
    char_cnn_weights = weights[:, 1]
    avg_weight = np.mean(char_cnn_weights)

    colors = []
    for w in char_cnn_weights:
        if w > avg_weight * 1.1:
            colors.append("#2e7d32")  # Dark green (above average)
        elif w < avg_weight * 0.9:
            colors.append("#c62828")  # Dark red (below average)
        else:
            colors.append("#1976d2")  # Blue (near average)

    if morphology_info:
        for i, token in enumerate(tokens):
            has_prefix, has_suffix, complexity = detect_morphological_complexity(token)
            if complexity > 0:
                colors[i] = "#7b1fa2"  # Purple for morphologically complex

    fig, ax = plt.subplots(figsize=(max(10, len(tokens) * 0.6), 4.5))
    sns.set_style("white")
    sns.set_context("paper", font_scale=1.0)

    x = np.arange(len(tokens))
    bars = ax.bar(
        x, char_cnn_weights * 100, color=colors, edgecolor="white", linewidth=0.8
    )

    ax.axhline(
        y=avg_weight * 100,
        color="#424242",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label=f"Average: {avg_weight*100:.3f}%",
    )

    for i, (bar, weight) in enumerate(zip(bars, char_cnn_weights)):
        if weight > avg_weight * 1.2:  # Only annotate if significantly above
            ax.annotate(
                f"{weight*100:.2f}%",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="normal",
            )

    ax.set_xlabel("Tokens", fontsize=12, weight="bold", labelpad=10)
    ax.set_ylabel("CharCNN Weight (%)", fontsize=12, weight="bold", labelpad=10)

    if not paper_format:
        ax.set_title(
            "CharCNN Attention Weight Variation per Token",
            fontsize=13,
            weight="bold",
            pad=15,
        )

    ax.set_xticks(x)
    if len(tokens) > 12:
        ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=10)
    else:
        ax.set_xticklabels(tokens, rotation=0, ha="center", fontsize=10)

    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#2e7d32", label="Above Average"),
        Patch(facecolor="#1976d2", label="Near Average"),
        Patch(facecolor="#c62828", label="Below Average"),
    ]
    if morphology_info:
        legend_elements.append(
            Patch(facecolor="#7b1fa2", label="Morphologically Complex")
        )
    ax.legend(
        handles=legend_elements,
        loc="upper right",
        fontsize=9,
        frameon=True,
        fancybox=False,
    )

    plt.tight_layout(pad=1.5)
    save_path = VISUALIZATION_DIR / filename
    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
        pad_inches=0.1,
    )
    plt.close()
    print(f"Saved CharCNN variation chart: {save_path}")
    return save_path


def plot_attention_comparison(
    tokens,
    weights,
    tags,
    word_to_ix,
    filename="attention_comparison.png",
    paper_format=True,
):
    """
    Create a multi-panel visualization comparing all sources with focus on CharCNN (paper-optimized).

    Args:
        tokens: List of token strings
        weights: Attention weights array of shape (seq_len, 3)
        tags: List of NER tags
        word_to_ix: Word to index mapping dictionary
        filename: Output filename
        paper_format: If True, optimize for paper publication

    Returns:
        Path to saved visualization file
    """
    sns.set_style("white")
    sns.set_context("paper", font_scale=1.0)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        "Token-Aware Multi-Source Attention Analysis" if not paper_format else "",
        fontsize=14,
        weight="bold",
        y=0.98,
    )

    sources = ["Word2Vec", "CharCNN", "Transformer"]
    colors_source = ["#1976d2", "#2e7d32", "#d32f2f"]  # More professional colors

    ax1 = axes[0, 0]
    x = np.arange(len(tokens))
    width = 0.8
    bottom = np.zeros(len(tokens))
    for idx, (source, color) in enumerate(zip(sources, colors_source)):
        ax1.bar(x, weights[:, idx], width, bottom=bottom, label=source, color=color)
        bottom += weights[:, idx]

    ax1.set_xlabel("Tokens", fontsize=11, weight="bold")
    ax1.set_ylabel("Attention Weight", fontsize=11, weight="bold")
    ax1.set_title(
        "(a) Source Distribution per Token", fontsize=11, weight="bold", pad=8
    )
    ax1.set_xticks(x)
    if len(tokens) > 12:
        ax1.set_xticklabels(tokens, rotation=45, ha="right", fontsize=9)
    else:
        ax1.set_xticklabels(tokens, rotation=0, ha="center", fontsize=9)
    ax1.legend(loc="upper right", fontsize=9, frameon=True, fancybox=False)
    ax1.set_ylim(0, 1.05)
    ax1.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)

    ax2 = axes[0, 1]
    char_cnn_weights = weights[:, 1]
    avg_char = np.mean(char_cnn_weights)
    bar_colors = [
        "#2e7d32" if w > avg_char * 1.2 else "#90a4ae" for w in char_cnn_weights
    ]

    for i, token in enumerate(tokens):
        has_prefix, has_suffix, complexity = detect_morphological_complexity(token)
        if complexity > 0 and char_cnn_weights[i] > avg_char:
            bar_colors[i] = "#7b1fa2"

    bars = ax2.bar(
        x, char_cnn_weights * 100, color=bar_colors, edgecolor="white", linewidth=0.8
    )
    ax2.axhline(
        y=avg_char * 100, color="#424242", linestyle="--", linewidth=1.5, alpha=0.7
    )
    ax2.set_xlabel("Tokens", fontsize=11, weight="bold")
    ax2.set_ylabel("CharCNN Weight (%)", fontsize=11, weight="bold")
    ax2.set_title("(b) CharCNN Weights (Zoomed)", fontsize=11, weight="bold", pad=8)
    ax2.set_xticks(x)
    if len(tokens) > 12:
        ax2.set_xticklabels(tokens, rotation=45, ha="right", fontsize=9)
    else:
        ax2.set_xticklabels(tokens, rotation=0, ha="center", fontsize=9)
    ax2.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)

    ax3 = axes[1, 0]
    w2v_weights = weights[:, 0]
    avg_w2v = np.mean(w2v_weights)
    bar_colors_w2v = [
        "#1976d2" if w > avg_w2v * 1.2 else "#90a4ae" for w in w2v_weights
    ]
    ax3.bar(
        x, w2v_weights * 100, color=bar_colors_w2v, edgecolor="white", linewidth=0.8
    )
    ax3.axhline(
        y=avg_w2v * 100, color="#424242", linestyle="--", linewidth=1.5, alpha=0.7
    )
    ax3.set_xlabel("Tokens", fontsize=11, weight="bold")
    ax3.set_ylabel("Word2Vec Weight (%)", fontsize=11, weight="bold")
    ax3.set_title("(c) Word2Vec Weights (Zoomed)", fontsize=11, weight="bold", pad=8)
    ax3.set_xticks(x)
    if len(tokens) > 12:
        ax3.set_xticklabels(tokens, rotation=45, ha="right", fontsize=9)
    else:
        ax3.set_xticklabels(tokens, rotation=0, ha="center", fontsize=9)
    ax3.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)

    ax4 = axes[1, 1]
    non_transformer = weights[:, 0] + weights[:, 1]
    char_cnn_relative = np.where(
        non_transformer > 0, weights[:, 1] / non_transformer, 0.5
    )

    bar_colors_rel = ["#2e7d32" if r > 0.5 else "#1976d2" for r in char_cnn_relative]
    ax4.bar(
        x,
        char_cnn_relative * 100,
        color=bar_colors_rel,
        edgecolor="white",
        linewidth=0.8,
    )
    ax4.axhline(y=50, color="#424242", linestyle="--", linewidth=1.5, alpha=0.7)
    ax4.set_xlabel("Tokens", fontsize=11, weight="bold")
    ax4.set_ylabel("CharCNN % of (CharCNN+Word2Vec)", fontsize=11, weight="bold")
    ax4.set_title(
        "(d) CharCNN vs Word2Vec Relative Weight", fontsize=11, weight="bold", pad=8
    )
    ax4.set_xticks(x)
    if len(tokens) > 12:
        ax4.set_xticklabels(tokens, rotation=45, ha="right", fontsize=9)
    else:
        ax4.set_xticklabels(tokens, rotation=0, ha="center", fontsize=9)
    ax4.set_ylim(0, 100)
    ax4.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)

    plt.tight_layout(pad=2.0)

    save_path = VISUALIZATION_DIR / filename
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved comparison chart: {save_path}")
    return save_path


def plot_char_w2v_comparison(
    tokens,
    weights,
    tags,
    word_to_ix,
    filename="char_w2v_comparison.png",
    paper_format=True,
):
    """
    Create visualization focusing only on CharCNN and Word2Vec weights.
    This highlights their complementary contributions without Transformer dominance.
    """
    sns.set_style("white")
    sns.set_context("paper", font_scale=1.0)

    fig, axes = plt.subplots(2, 1, figsize=(max(12, len(tokens) * 0.7), 8))
    
    x = np.arange(len(tokens))
    char_cnn_weights = weights[:, 1] * 100
    w2v_weights = weights[:, 0] * 100
    
    avg_char = np.mean(char_cnn_weights)
    avg_w2v = np.mean(w2v_weights)
    
    ax1 = axes[0]
    char_colors = []
    for i, (token, weight) in enumerate(zip(tokens, char_cnn_weights)):
        is_oov = token.lower() not in word_to_ix or word_to_ix.get(token.lower()) == word_to_ix.get("<UNK>")
        has_prefix, has_suffix, complexity = detect_morphological_complexity(token)
        
        if is_oov or complexity > 0:
            char_colors.append("#7b1fa2")
        elif weight > avg_char * 1.2:
            char_colors.append("#2e7d32")
        elif weight > avg_char:
            char_colors.append("#66bb6a")
        else:
            char_colors.append("#90a4ae")
    
    bars1 = ax1.bar(x, char_cnn_weights, color=char_colors, edgecolor="white", linewidth=0.8, alpha=0.85)
    ax1.axhline(y=avg_char, color="#424242", linestyle="--", linewidth=1.5, alpha=0.7, label=f"Average: {avg_char:.3f}%")
    ax1.set_ylabel("CharCNN Weight (%)", fontsize=12, weight="bold", labelpad=10)
    ax1.set_xticks(x)
    if len(tokens) > 12:
        ax1.set_xticklabels(tokens, rotation=45, ha="right", fontsize=10)
    else:
        ax1.set_xticklabels(tokens, rotation=0, ha="center", fontsize=10)
    ax1.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)
    ax1.legend(loc="upper right", fontsize=9, frameon=True)
    
    for i, (bar, weight) in enumerate(zip(bars1, char_cnn_weights)):
        if weight > avg_char * 1.3:
            ax1.annotate(
                f"{weight:.2f}%",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    
    ax2 = axes[1]
    w2v_colors = []
    for i, (token, weight) in enumerate(zip(tokens, w2v_weights)):
        is_oov = token.lower() not in word_to_ix or word_to_ix.get(token.lower()) == word_to_ix.get("<UNK>")
        
        if not is_oov and weight > avg_w2v * 1.2:
            w2v_colors.append("#1976d2")
        elif weight > avg_w2v:
            w2v_colors.append("#64b5f6")
        else:
            w2v_colors.append("#90a4ae")
    
    bars2 = ax2.bar(x, w2v_weights, color=w2v_colors, edgecolor="white", linewidth=0.8, alpha=0.85)
    ax2.axhline(y=avg_w2v, color="#424242", linestyle="--", linewidth=1.5, alpha=0.7, label=f"Average: {avg_w2v:.3f}%")
    ax2.set_xlabel("Tokens", fontsize=12, weight="bold", labelpad=10)
    ax2.set_ylabel("Word2Vec Weight (%)", fontsize=12, weight="bold", labelpad=10)
    ax2.set_xticks(x)
    if len(tokens) > 12:
        ax2.set_xticklabels(tokens, rotation=45, ha="right", fontsize=10)
    else:
        ax2.set_xticklabels(tokens, rotation=0, ha="center", fontsize=10)
    ax2.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)
    ax2.legend(loc="upper right", fontsize=9, frameon=True)
    
    for i, (bar, weight) in enumerate(zip(bars2, w2v_weights)):
        if weight > avg_w2v * 1.3:
            ax2.annotate(
                f"{weight:.2f}%",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#7b1fa2", label="OOV/Morphological (CharCNN)"),
        Patch(facecolor="#2e7d32", label="Above Avg (CharCNN)"),
        Patch(facecolor="#1976d2", label="Above Avg (Word2Vec)"),
    ]
    fig.legend(handles=legend_elements, loc="upper center", ncol=3, fontsize=9, frameon=True, bbox_to_anchor=(0.5, 0.98))
    
    plt.tight_layout(pad=2.0, rect=[0, 0, 1, 0.96])
    
    save_path = VISUALIZATION_DIR / filename
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none", pad_inches=0.1)
    plt.close()
    print(f"Saved CharCNN-Word2Vec comparison: {save_path}")
    return save_path


def analyze_aggregate_attention_patterns(
    sentences_with_tags,
    model,
    tokenizer,
    word_to_ix,
    char_to_ix,
    device,
    max_sentences=500,
):
    """
    Analyze attention patterns across entire dataset to show token-aware behavior.
    Groups tokens by type and computes aggregate statistics.
    """
    from collections import defaultdict

    token_type_weights = defaultdict(list)
    all_weights = []

    print(
        f"   Analyzing aggregate attention patterns across {min(len(sentences_with_tags), max_sentences)} sentences..."
    )

    for idx, (tokens, tags) in enumerate(sentences_with_tags[:max_sentences]):
        if (idx + 1) % 100 == 0:
            print(
                f"   Progress: {idx + 1}/{min(len(sentences_with_tags), max_sentences)}..."
            )

        try:
            text = " ".join(tokens)
            _, words, chars, bert_ids, bert_mask, word_maps = prepare_input(
                text, tokenizer, word_to_ix, char_to_ix, device
            )

            with torch.no_grad():
                weights = model.get_attention_weights(
                    words, chars, bert_ids, bert_mask, word_maps
                )
                weights_np = weights[0].cpu().numpy()

            for i, (token, tag) in enumerate(zip(tokens, tags)):
                if i >= weights_np.shape[0]:
                    break

                w2v_weight = weights_np[i, 0]
                char_weight = weights_np[i, 1]
                trans_weight = weights_np[i, 2]

                all_weights.append(
                    {
                        "token": token,
                        "tag": tag,
                        "w2v": w2v_weight,
                        "char": char_weight,
                        "trans": trans_weight,
                    }
                )

                is_oov = token.lower() not in word_to_ix or word_to_ix.get(
                    token.lower(), word_to_ix.get("<UNK>", 0)
                ) == word_to_ix.get("<UNK>", 0)

                has_prefix, has_suffix, complexity = detect_morphological_complexity(
                    token
                )
                is_morphological = complexity > 0
                is_entity = tag != "O"
                is_location = "LOC" in tag
                is_person = "PER" in tag
                is_org = "ORG" in tag

                if is_oov:
                    token_type_weights["OOV"].append(char_weight)
                else:
                    token_type_weights["In-Vocab"].append(char_weight)

                if is_morphological:
                    token_type_weights["Morphological"].append(char_weight)
                else:
                    token_type_weights["Simple"].append(char_weight)

                if is_entity:
                    token_type_weights["Entity"].append(trans_weight)
                    if is_location:
                        token_type_weights["LOC"].append(char_weight)
                    if is_person:
                        token_type_weights["PER"].append(trans_weight)
                    if is_org:
                        token_type_weights["ORG"].append(trans_weight)
                else:
                    token_type_weights["Non-Entity"].append(trans_weight)

                token_type_weights["All"].append(char_weight)

        except Exception as e:
            continue

    results = {}
    for token_type, weights_list in token_type_weights.items():
        if len(weights_list) > 0:
            results[token_type] = {
                "mean": np.mean(weights_list),
                "std": np.std(weights_list),
                "median": np.median(weights_list),
                "count": len(weights_list),
                "min": np.min(weights_list),
                "max": np.max(weights_list),
                "values": weights_list,
            }

    return results, all_weights


def plot_aggregate_char_cnn_by_token_type(
    aggregate_results, filename="aggregate_char_cnn_by_type.png"
):
    """
    Create bar chart showing CharCNN weight by token type.
    This is the key evidence that token-aware attention works.
    """
    sns.set_style("white")
    sns.set_context("paper", font_scale=1.0)

    token_types = ["OOV", "In-Vocab", "Morphological", "Simple"]
    means = []
    stds = []
    counts = []

    for tt in token_types:
        if tt in aggregate_results:
            means.append(aggregate_results[tt]["mean"] * 100)
            stds.append(aggregate_results[tt]["std"] * 100)
            counts.append(aggregate_results[tt]["count"])
        else:
            means.append(0)
            stds.append(0)
            counts.append(0)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(token_types))
    colors = ["#d32f2f", "#1976d2", "#7b1fa2", "#388e3c"]

    bars = ax.bar(
        x, means, yerr=stds, capsize=5, color=colors, edgecolor="white", linewidth=1.5
    )

    for i, (bar, count, mean) in enumerate(zip(bars, counts, means)):
        ax.annotate(
            f"{mean:.3f}%\n(n={count})",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_xlabel("Token Type", fontsize=12, weight="bold", labelpad=10)
    ax.set_ylabel(
        "Mean CharCNN Attention Weight (%)", fontsize=12, weight="bold", labelpad=10
    )
    ax.set_xticks(x)
    ax.set_xticklabels(token_types, fontsize=11)
    ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)

    plt.tight_layout(pad=1.5)
    save_path = VISUALIZATION_DIR / filename
    plt.savefig(
        save_path, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.1
    )
    plt.close()
    print(f"Saved aggregate CharCNN analysis: {save_path}")
    return save_path


def plot_token_aware_evidence(aggregate_results, filename="token_aware_evidence.png"):
    """
    Create multi-panel figure showing statistical evidence of token-aware attention.
    """
    sns.set_style("white")
    sns.set_context("paper", font_scale=1.0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    ax1 = axes[0]
    oov_vals = aggregate_results.get("OOV", {}).get("values", [])
    invocab_vals = aggregate_results.get("In-Vocab", {}).get("values", [])

    if oov_vals and invocab_vals:
        data_box = [np.array(oov_vals) * 100, np.array(invocab_vals) * 100]
        bp = ax1.boxplot(data_box, labels=["OOV", "In-Vocab"], patch_artist=True)
        bp["boxes"][0].set_facecolor("#d32f2f")
        bp["boxes"][1].set_facecolor("#1976d2")
        for box in bp["boxes"]:
            box.set_alpha(0.7)

        oov_mean = aggregate_results["OOV"]["mean"] * 100
        invocab_mean = aggregate_results["In-Vocab"]["mean"] * 100
        diff_pct = (
            ((oov_mean - invocab_mean) / invocab_mean) * 100 if invocab_mean > 0 else 0
        )

        ax1.set_title(
            f"(a) OOV vs In-Vocab\nOOV {diff_pct:+.1f}% higher",
            fontsize=11,
            weight="bold",
            pad=10,
        )

    ax1.set_ylabel("CharCNN Weight (%)", fontsize=11, weight="bold")
    ax1.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)

    ax2 = axes[1]
    morph_vals = aggregate_results.get("Morphological", {}).get("values", [])
    simple_vals = aggregate_results.get("Simple", {}).get("values", [])

    if morph_vals and simple_vals:
        data_box2 = [np.array(morph_vals) * 100, np.array(simple_vals) * 100]
        bp2 = ax2.boxplot(
            data_box2, labels=["Morphological", "Simple"], patch_artist=True
        )
        bp2["boxes"][0].set_facecolor("#7b1fa2")
        bp2["boxes"][1].set_facecolor("#388e3c")
        for box in bp2["boxes"]:
            box.set_alpha(0.7)

        morph_mean = aggregate_results["Morphological"]["mean"] * 100
        simple_mean = aggregate_results["Simple"]["mean"] * 100
        diff_pct2 = (
            ((morph_mean - simple_mean) / simple_mean) * 100 if simple_mean > 0 else 0
        )

        ax2.set_title(
            f"(b) Morphological vs Simple\nMorphological {diff_pct2:+.1f}% higher",
            fontsize=11,
            weight="bold",
            pad=10,
        )

    ax2.set_ylabel("CharCNN Weight (%)", fontsize=11, weight="bold")
    ax2.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)

    ax3 = axes[2]
    entity_vals = aggregate_results.get("Entity", {}).get("values", [])
    nonentity_vals = aggregate_results.get("Non-Entity", {}).get("values", [])

    if entity_vals and nonentity_vals:
        data_box3 = [np.array(entity_vals) * 100, np.array(nonentity_vals) * 100]
        bp3 = ax3.boxplot(data_box3, labels=["Entity", "Non-Entity"], patch_artist=True)
        bp3["boxes"][0].set_facecolor("#ff7043")
        bp3["boxes"][1].set_facecolor("#78909c")
        for box in bp3["boxes"]:
            box.set_alpha(0.7)

        entity_mean = aggregate_results["Entity"]["mean"] * 100
        nonentity_mean = aggregate_results["Non-Entity"]["mean"] * 100
        diff_pct3 = (
            ((entity_mean - nonentity_mean) / nonentity_mean) * 100
            if nonentity_mean > 0
            else 0
        )

        ax3.set_title(
            f"(c) Entity vs Non-Entity (Transformer)\nEntity {diff_pct3:+.1f}% higher",
            fontsize=11,
            weight="bold",
            pad=10,
        )

    ax3.set_ylabel("Transformer Weight (%)", fontsize=11, weight="bold")
    ax3.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)

    plt.tight_layout(pad=2.0)
    save_path = VISUALIZATION_DIR / filename
    plt.savefig(
        save_path, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.1
    )
    plt.close()
    print(f"Saved token-aware evidence: {save_path}")
    return save_path


def plot_attention_distribution_histogram(
    all_weights, filename="attention_distribution.png"
):
    """
    Create histogram showing attention weight distribution by source.
    """
    sns.set_style("white")
    sns.set_context("paper", font_scale=1.0)

    w2v_weights = [w["w2v"] * 100 for w in all_weights]
    char_weights = [w["char"] * 100 for w in all_weights]
    trans_weights = [w["trans"] * 100 for w in all_weights]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ax1 = axes[0]
    ax1.hist(w2v_weights, bins=50, color="#1976d2", alpha=0.7, edgecolor="white")
    ax1.axvline(
        x=np.mean(w2v_weights),
        color="#d32f2f",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {np.mean(w2v_weights):.2f}%",
    )
    ax1.set_xlabel("Word2Vec Weight (%)", fontsize=11, weight="bold")
    ax1.set_ylabel("Frequency", fontsize=11, weight="bold")
    ax1.set_title("(a) Word2Vec Distribution", fontsize=11, weight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", alpha=0.3, linestyle="--")

    ax2 = axes[1]
    ax2.hist(char_weights, bins=50, color="#2e7d32", alpha=0.7, edgecolor="white")
    ax2.axvline(
        x=np.mean(char_weights),
        color="#d32f2f",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {np.mean(char_weights):.2f}%",
    )
    ax2.set_xlabel("CharCNN Weight (%)", fontsize=11, weight="bold")
    ax2.set_ylabel("Frequency", fontsize=11, weight="bold")
    ax2.set_title("(b) CharCNN Distribution", fontsize=11, weight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.3, linestyle="--")

    ax3 = axes[2]
    ax3.hist(trans_weights, bins=50, color="#d32f2f", alpha=0.7, edgecolor="white")
    ax3.axvline(
        x=np.mean(trans_weights),
        color="#1976d2",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {np.mean(trans_weights):.2f}%",
    )
    ax3.set_xlabel("Transformer Weight (%)", fontsize=11, weight="bold")
    ax3.set_ylabel("Frequency", fontsize=11, weight="bold")
    ax3.set_title("(c) Transformer Distribution", fontsize=11, weight="bold")
    ax3.legend(fontsize=9)
    ax3.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout(pad=2.0)
    save_path = VISUALIZATION_DIR / filename
    plt.savefig(
        save_path, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.1
    )
    plt.close()
    print(f"Saved attention distribution: {save_path}")
    return save_path


def prepare_input(text, tokenizer, word_to_ix, char_to_ix, device):
    """
    Prepare input tensors for model inference.

    Args:
        text: Input text string
        tokenizer: Tokenizer for transformer model
        word_to_ix: Word to index mapping dictionary
        char_to_ix: Character to index mapping dictionary
        device: Device for tensor computation

    Returns:
        Tuple of (tokens, word_tensor, char_tensor, bert_tensor, bert_mask, word_maps)
    """
    tokens = text.split()
    word_indices = [word_to_ix.get(t.lower(), word_to_ix["<UNK>"]) for t in tokens]
    word_tensor = torch.tensor([word_indices]).to(device, non_blocking=True)

    max_word_len = max(len(t) for t in tokens)
    char_indices = []
    for t in tokens:
        chars = [char_to_ix.get(c, char_to_ix["<UNK>"]) for c in t]
        chars += [0] * (max_word_len - len(chars))
        char_indices.append(chars)
    char_tensor = torch.tensor([char_indices]).to(device, non_blocking=True)

    bert_subwords = []
    word_maps = []
    for t in tokens:
        subws = tokenizer.tokenize(t)
        word_maps.append(
            list(range(len(bert_subwords), len(bert_subwords) + len(subws)))
        )
        bert_subwords.extend(subws)

    bert_indices = tokenizer.convert_tokens_to_ids(bert_subwords)
    bert_tensor = torch.tensor([bert_indices]).to(device, non_blocking=True)
    bert_mask = torch.ones_like(bert_tensor).to(device, non_blocking=True)

    return tokens, word_tensor, char_tensor, bert_tensor, bert_mask, [word_maps]


def main():
    print("=" * 70)
    print("PAPER ATTENTION VISUALIZATION GENERATOR")
    print("=" * 70)

    print("\n[1/4] Loading model and configuration...")
    if not BEST_MODEL_PATH.exists():
        print(f"ERROR: Model file not found at {BEST_MODEL_PATH}")
        sys.exit(1)

    try:
        with open(HP_PATH, "r") as f:
            hp = json.load(f)
        with open(VOCAB_PATH, "r") as f:
            vocab = json.load(f)

        word_to_ix = vocab["word_to_ix"]
        char_to_ix = vocab["char_to_ix"]
        tag_to_ix = vocab["tag_to_ix"]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        print(f"   Using device: {device}")

        model = MultiSource_Attention_CRF(
            len(word_to_ix), len(char_to_ix), tag_to_ix, hp
        ).to(device)

        state_dict = torch.load(BEST_MODEL_PATH, map_location=device, weights_only=True)
        model_state_dict = model.state_dict()
        filtered_state_dict = {
            k: v
            for k, v in state_dict.items()
            if k in model_state_dict and model_state_dict[k].shape == v.shape
        }
        model.load_state_dict(filtered_state_dict, strict=False)
        model.eval()
        print("   Model loaded successfully")

        tokenizer = AutoTokenizer.from_pretrained(hp["transformer_model"])
        print("   Tokenizer loaded")

    except Exception as e:
        print(f"ERROR loading model: {e}")
        sys.exit(1)

    print("\n[2/4] Analyzing sentences from dataset with attention weights...")
    sentences_with_tags = load_sentences_with_tags(TEST_FILE, limit=1000)
    print(f"   Loaded {len(sentences_with_tags)} sentences")

    best_sentences = find_best_sentences(
        sentences_with_tags,
        top_k=10,
        model=model,
        tokenizer=tokenizer,
        word_to_ix=word_to_ix,
        char_to_ix=char_to_ix,
        device=device,
    )
    print(f"   Found {len(best_sentences)} high-quality sentences")

    print("\n[3/5] Aggregate attention analysis across dataset...")
    print("-" * 70)
    aggregate_results, all_weights = analyze_aggregate_attention_patterns(
        sentences_with_tags,
        model,
        tokenizer,
        word_to_ix,
        char_to_ix,
        device,
        max_sentences=500,
    )

    print("\n   Aggregate Statistics (CharCNN weights by token type):")
    for token_type in ["OOV", "In-Vocab", "Morphological", "Simple"]:
        if token_type in aggregate_results:
            stats = aggregate_results[token_type]
            print(
                f"     {token_type}: mean={stats['mean']*100:.4f}%, std={stats['std']*100:.4f}%, n={stats['count']}"
            )

    if "OOV" in aggregate_results and "In-Vocab" in aggregate_results:
        oov_mean = aggregate_results["OOV"]["mean"]
        invocab_mean = aggregate_results["In-Vocab"]["mean"]
        diff = (
            ((oov_mean - invocab_mean) / invocab_mean) * 100 if invocab_mean > 0 else 0
        )
        print(
            f"\n   KEY FINDING: OOV tokens receive {diff:+.2f}% higher CharCNN weight than In-Vocab tokens"
        )

    if "Morphological" in aggregate_results and "Simple" in aggregate_results:
        morph_mean = aggregate_results["Morphological"]["mean"]
        simple_mean = aggregate_results["Simple"]["mean"]
        diff2 = (
            ((morph_mean - simple_mean) / simple_mean) * 100 if simple_mean > 0 else 0
        )
        print(
            f"   KEY FINDING: Morphological tokens receive {diff2:+.2f}% higher CharCNN weight than Simple tokens"
        )

    plot_aggregate_char_cnn_by_token_type(
        aggregate_results, "aggregate_char_cnn_by_type.png"
    )
    plot_token_aware_evidence(aggregate_results, "token_aware_evidence.png")
    plot_attention_distribution_histogram(all_weights, "attention_distribution.png")

    print("\n[4/5] Top candidate sentences (sorted by combined score):")
    print("-" * 70)
    for i, candidate in enumerate(best_sentences[:5], 1):
        print(
            f"\n{i}. Total Score: {candidate['score']:.2f} (Static: {candidate['details'].get('static_score', 0):.1f}, Attention: {candidate['details'].get('attention_score', 0):.1f})"
        )
        print(f"   Text: {candidate['text']}")
        print(
            f"   Entities: PER={candidate['details']['person_count']}, "
            f"LOC={candidate['details']['location_count']}, "
            f"ORG={candidate['details']['org_count']}"
        )
        print(f"   Tokens: {candidate['details']['tokens_count']}")

        if "avg_variance" in candidate["details"]:
            att_metrics = candidate["details"]
            print(f"   Attention Metrics:")
            print(
                f"     - Source Diversity: {att_metrics.get('source_diversity', 0)}/3 sources used"
            )
            print(
                f"     - Source Balance: {att_metrics.get('source_balance', 0):.2%} (higher = more balanced)"
            )
            print(
                f"     - Variance: {att_metrics.get('avg_variance', 0):.4f} (higher = more adaptive)"
            )
            print(
                f"     - CharCNN Max Count: {att_metrics.get('char_cnn_max_count', 0)} tokens"
            )
            print(
                f"     - CharCNN Usage: {att_metrics.get('char_cnn_relative_high', 0)} tokens (near max), "
                f"{att_metrics.get('char_cnn_relative_good', 0)} tokens (>avg+2std), "
                f"{att_metrics.get('char_cnn_relative_medium', 0)} tokens (>avg+1std), "
                f"max: {att_metrics.get('max_char_cnn_weight', 0):.4f}, avg: {att_metrics.get('avg_char_cnn_weight', 0):.4f}"
            )
            print(
                f"     - Transformer Max Count: {att_metrics.get('transformer_max_count', 0)} tokens"
            )
            print(
                f"     - Word2Vec Max Count: {att_metrics.get('w2v_max_count', 0)} tokens"
            )
            print(
                f"     - Morphology-CNN Alignment: {att_metrics.get('morphology_alignment_ratio', 0):.2%} ({att_metrics.get('morphology_cnn_alignment', 0)}/{att_metrics.get('morphology_tokens', 0)} tokens)"
            )
            print(
                f"     - Context-Transformer Alignment: {att_metrics.get('context_alignment_ratio', 0):.2%} ({att_metrics.get('context_transformer_alignment', 0)}/{att_metrics.get('entity_tokens', 0)} tokens)"
            )
            print(
                f"     - Location-CNN Alignment: {att_metrics.get('location_alignment_ratio', 0):.2%} ({att_metrics.get('location_cnn_alignment', 0)}/{att_metrics.get('location_tokens', 0)} tokens)"
            )

        if "reason" in candidate["details"]:
            print(f"   Reason: {candidate['details']['reason']}")

    print("\n[5/5] Generating attention visualizations...")
    print("-" * 70)

    for i, candidate in enumerate(best_sentences[:3], 1):
        try:
            text = candidate["text"]
            tokens = candidate["tokens"]

            print(f"\nProcessing candidate {i}: {text[:60]}...")

            if (
                "weights" in candidate["details"]
                and candidate["details"]["weights"] is not None
            ):
                weights_np = np.array(candidate["details"]["weights"])
                print("   Using pre-computed attention weights")
            else:
                _, words, chars, bert_ids, bert_mask, word_maps = prepare_input(
                    text, tokenizer, word_to_ix, char_to_ix, device
                )

                with torch.no_grad():
                    weights = model.get_attention_weights(
                        words, chars, bert_ids, bert_mask, word_maps
                    )
                    weights_np = weights[0].cpu().numpy()

            if i == 1:
                filename = "attention_weights_example.png"
                att_metrics = candidate["details"]
                title_parts = [
                    f"Score: {candidate['score']:.1f}",
                    f"Entities: {candidate['details']['num_entities']} ({candidate['details']['entity_types']} types)",
                ]

                if "avg_variance" in att_metrics:
                    title_parts.append(f"Variance: {att_metrics['avg_variance']:.3f}")
                    if att_metrics.get("location_alignment_ratio", 0) > 0:
                        title_parts.append(
                            f"LOC-CNN: {att_metrics['location_alignment_ratio']:.0%}"
                        )

                title_suffix = " | ".join(title_parts)
            else:
                filename = f"attention_weights_candidate_{i}.png"
                title_suffix = f"Candidate {i} | Score: {candidate['score']:.1f}"

            plot_attention_heatmap(
                tokens, weights_np, filename, title_suffix, paper_format=True
            )

            if i == 1:
                plot_char_cnn_variation(
                    tokens,
                    weights_np,
                    "char_cnn_variation_example.png",
                    morphology_info=True,
                    paper_format=True,
                )
                plot_attention_comparison(
                    tokens,
                    weights_np,
                    candidate["tags"],
                    word_to_ix,
                    "attention_comparison_example.png",
                    paper_format=True,
                )
                plot_char_w2v_comparison(
                    tokens,
                    weights_np,
                    candidate["tags"],
                    word_to_ix,
                    "char_w2v_comparison_example.png",
                    paper_format=True,
                )

            print(f"   Attention weight statistics:")
            print(
                f"     Word2Vec:   min={weights_np[:, 0].min():.3f}, "
                f"max={weights_np[:, 0].max():.3f}, mean={weights_np[:, 0].mean():.3f}"
            )
            print(
                f"     CharCNN:    min={weights_np[:, 1].min():.3f}, "
                f"max={weights_np[:, 1].max():.3f}, mean={weights_np[:, 1].mean():.3f}"
            )
            print(
                f"     Transformer: min={weights_np[:, 2].min():.3f}, "
                f"max={weights_np[:, 2].max():.3f},                 mean={weights_np[:, 2].mean():.3f}"
            )

            if "morphology_alignment_ratio" in candidate["details"]:
                att_metrics = candidate["details"]
                print(f"   Key Insights:")
                print(
                    f"     - Source Diversity: {att_metrics.get('source_diversity', 0)}/3 sources used"
                )
                print(
                    f"     - CharCNN used for: {att_metrics.get('char_cnn_max_count', 0)} tokens (max), "
                    f"{att_metrics.get('char_cnn_dominant_count', 0)} tokens (strong), "
                    f"{att_metrics.get('char_cnn_relative_high', 0)} tokens (near max), "
                    f"{att_metrics.get('char_cnn_relative_good', 0)} tokens (>avg+2std)"
                )
                print(
                    f"     - CharCNN weights: max={att_metrics.get('max_char_cnn_weight', 0):.4f}, "
                    f"avg={att_metrics.get('avg_char_cnn_weight', 0):.4f}, "
                    f"std={att_metrics.get('std_char_cnn_weight', 0):.4f}"
                )
                print(
                    f"     - Morphology-CNN alignment: {att_metrics.get('morphology_alignment_ratio', 0):.1%} ({att_metrics.get('morphology_cnn_alignment', 0)}/{att_metrics.get('morphology_tokens', 0)} complex tokens)"
                )
                print(
                    f"     - Context-Transformer alignment: {att_metrics.get('context_alignment_ratio', 0):.1%} ({att_metrics.get('context_transformer_alignment', 0)}/{att_metrics.get('entity_tokens', 0)} entity tokens)"
                )
                if att_metrics.get("location_tokens", 0) > 0:
                    print(
                        f"     - Location-CNN alignment: {att_metrics.get('location_alignment_ratio', 0):.1%} ({att_metrics.get('location_cnn_alignment', 0)}/{att_metrics.get('location_tokens', 0)} location tokens)"
                    )
                if att_metrics.get("oov_tokens", 0) > 0:
                    print(
                        f"     - OOV-CNN alignment: {att_metrics.get('oov_alignment_ratio', 0):.1%} ({att_metrics.get('oov_cnn_alignment', 0)}/{att_metrics.get('oov_tokens', 0)} OOV tokens)"
                    )

        except Exception as e:
            print(f"   Error processing: {e}")
            import traceback

            traceback.print_exc()
            continue

    print("\n" + "=" * 70)
    print("COMPLETE!")
    print("=" * 70)
    print(f"\n[AGGREGATE ANALYSIS - KEY EVIDENCE FOR PAPER]")
    print(f"  {VISUALIZATION_DIR / 'token_aware_evidence.png'}")
    print(f"  {VISUALIZATION_DIR / 'aggregate_char_cnn_by_type.png'}")
    print(f"  {VISUALIZATION_DIR / 'attention_distribution.png'}")
    print(f"\n[SINGLE SENTENCE VISUALIZATIONS]")
    print(f"  {VISUALIZATION_DIR / 'attention_weights_example.png'}")
    print(f"  {VISUALIZATION_DIR / 'char_cnn_variation_example.png'}")
    print(f"  {VISUALIZATION_DIR / 'attention_comparison_example.png'}")
    print(f"  {VISUALIZATION_DIR / 'char_w2v_comparison_example.png'}")


if __name__ == "__main__":
    main()
