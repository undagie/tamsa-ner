import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from collections import Counter
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="torch")
from torchcrf import CRF
import time
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from flair.embeddings import WordEmbeddings
from flair.data import Sentence as FlairSentence
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys

sys.path.insert(0, str(Path(__file__).parent))
_ROOT = Path(__file__).resolve().parent.parent
from train_attention_fusion import (
    MultiSource_Attention_CRF,
    CharCNN,
    AttentionFusion,
    load_bio_file,
    build_vocab,
    NERMultiSourceDataset,
    collate_fn_multisource,
)

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix
from seqeval.metrics import classification_report as seqeval_classification_report
from seqeval.scheme import IOB2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

OUTPUT_DIR = _ROOT / "outputs" / "ablation_studies"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = _ROOT / "data" / "idner2k" / "train_bio.txt"
DEV_FILE = _ROOT / "data" / "idner2k" / "dev_bio.txt"
TEST_FILE = _ROOT / "data" / "idner2k" / "test_bio.txt"

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
    "early_stopping_patience": 15,
}


class TAMSA_NoTokenAwareAttention(MultiSource_Attention_CRF):
    """TAMSA without token-aware attention - uses uniform (1/3, 1/3, 1/3) attention."""

    def _get_features(self, words, chars, bert_indices, bert_mask, word_maps):
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
            for j, subword_idx in enumerate(mapping):
                if j < seq_len:
                    transformer_features[i, j, :] = transformer_hidden_states[
                        i, subword_idx, :
                    ]

        w2v_proj = torch.relu(self.w2v_proj(w2v_features))
        char_proj = torch.relu(self.char_proj(char_cnn_features))
        trans_proj = torch.relu(self.trans_proj(transformer_features))

        # Uniform attention: simple average (1/3 each)
        fused_features = (w2v_proj + char_proj + trans_proj) / 3.0
        fused_features = self.dropout(fused_features)

        lstm_out, _ = self.lstm(fused_features)
        lstm_out = self.dropout(lstm_out)
        emissions = self.hidden2tag(lstm_out)
        return emissions


class TAMSA_UniformAttention(MultiSource_Attention_CRF):
    """TAMSA with uniform attention weights (fixed, not learned)."""

    def _get_features(self, words, chars, bert_indices, bert_mask, word_maps):
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
            for j, subword_idx in enumerate(mapping):
                if j < seq_len:
                    transformer_features[i, j, :] = transformer_hidden_states[
                        i, subword_idx, :
                    ]

        w2v_proj = torch.relu(self.w2v_proj(w2v_features))
        char_proj = torch.relu(self.char_proj(char_cnn_features))
        trans_proj = torch.relu(self.trans_proj(transformer_features))

        # Uniform attention: fixed weights (1/3, 1/3, 1/3) per token
        # Same as NoTokenAwareAttention, but keep the name for clarity
        fused_features = (w2v_proj + char_proj + trans_proj) / 3.0
        fused_features = self.dropout(fused_features)

        lstm_out, _ = self.lstm(fused_features)
        lstm_out = self.dropout(lstm_out)
        emissions = self.hidden2tag(lstm_out)
        return emissions


class TAMSA_NoCharCNN(MultiSource_Attention_CRF):
    """TAMSA without Character CNN component."""

    def __init__(self, word_vocab_size, char_vocab_size, tag_to_ix, hp):
        super().__init__(word_vocab_size, char_vocab_size, tag_to_ix, hp)
        # Reinitialize fusion for 2 sources only (w2v + transformer)
        self.fusion = AttentionFusion(self.projection_dim, self.projection_dim // 2)

    def _get_features(self, words, chars, bert_indices, bert_mask, word_maps):
        w2v_features = self.w2v_embedding(words)
        # No char_cnn_features - skip char_cnn

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
            for j, subword_idx in enumerate(mapping):
                if j < seq_len:
                    transformer_features[i, j, :] = transformer_hidden_states[
                        i, subword_idx, :
                    ]

        w2v_proj = torch.relu(self.w2v_proj(w2v_features))
        trans_proj = torch.relu(self.trans_proj(transformer_features))

        # Fusion with only 2 sources (w2v and transformer)
        fused_features, _ = self.fusion([w2v_proj, trans_proj])
        fused_features = self.dropout(fused_features)

        lstm_out, _ = self.lstm(fused_features)
        lstm_out = self.dropout(lstm_out)
        emissions = self.hidden2tag(lstm_out)
        return emissions


class TAMSA_NoWord2Vec(MultiSource_Attention_CRF):
    """TAMSA without Word2Vec embeddings."""

    def __init__(self, word_vocab_size, char_vocab_size, tag_to_ix, hp):
        super().__init__(word_vocab_size, char_vocab_size, tag_to_ix, hp)
        # Reinitialize fusion for 2 sources only (char_cnn + transformer)
        self.fusion = AttentionFusion(self.projection_dim, self.projection_dim // 2)

    def _get_features(self, words, chars, bert_indices, bert_mask, word_maps):
        # Skip w2v_features - don't use w2v_embedding
        char_cnn_features = self.char_cnn(chars)

        transformer_outputs = self.transformer(bert_indices, attention_mask=bert_mask)
        transformer_hidden_states = transformer_outputs.last_hidden_state

        batch_size = chars.size(0)
        seq_len = chars.size(1)
        device = chars.device
        transformer_features = torch.zeros(
            batch_size, seq_len, self.hp["transformer_dim"], device=device
        )
        for i, mapping in enumerate(word_maps):
            if not mapping:
                continue
            for j, subword_idx in enumerate(mapping):
                if j < seq_len:
                    transformer_features[i, j, :] = transformer_hidden_states[
                        i, subword_idx, :
                    ]

        char_proj = torch.relu(self.char_proj(char_cnn_features))
        trans_proj = torch.relu(self.trans_proj(transformer_features))

        # Fusion with only 2 sources (char_cnn and transformer)
        fused_features, _ = self.fusion([char_proj, trans_proj])
        fused_features = self.dropout(fused_features)

        lstm_out, _ = self.lstm(fused_features)
        lstm_out = self.dropout(lstm_out)
        emissions = self.hidden2tag(lstm_out)
        return emissions


class TAMSA_NoIndoBERT(MultiSource_Attention_CRF):
    """TAMSA without IndoBERT transformer component."""

    def __init__(self, word_vocab_size, char_vocab_size, tag_to_ix, hp):
        super().__init__(word_vocab_size, char_vocab_size, tag_to_ix, hp)
        # Reinitialize fusion for 2 sources only (w2v + char_cnn)
        self.fusion = AttentionFusion(self.projection_dim, self.projection_dim // 2)

    def _get_features(self, words, chars, bert_indices, bert_mask, word_maps):
        w2v_features = self.w2v_embedding(words)
        char_cnn_features = self.char_cnn(chars)
        # Skip transformer_features - don't use transformer

        w2v_proj = torch.relu(self.w2v_proj(w2v_features))
        char_proj = torch.relu(self.char_proj(char_cnn_features))

        # Fusion with only 2 sources (w2v and char_cnn)
        fused_features, _ = self.fusion([w2v_proj, char_proj])
        fused_features = self.dropout(fused_features)

        lstm_out, _ = self.lstm(fused_features)
        lstm_out = self.dropout(lstm_out)
        emissions = self.hidden2tag(lstm_out)
        return emissions


def evaluate_model(model, loader, ix_to_tag, device):
    """Evaluate model and return classification report."""
    model.eval()
    all_preds, all_true = [], []

    with torch.no_grad():
        for (
            tokens,
            words,
            chars,
            tags,
            word_mask,
            bert_indices,
            bert_mask,
            word_maps,
        ) in loader:
            words = words.to(device, non_blocking=True)
            chars = chars.to(device, non_blocking=True)
            tags = tags.to(device, non_blocking=True)
            word_mask = word_mask.to(device, non_blocking=True)
            bert_indices = bert_indices.to(device, non_blocking=True)
            bert_mask = bert_mask.to(device, non_blocking=True)

            preds = model.predict(
                words, chars, word_mask, bert_indices, bert_mask, word_maps
            )

            for i in range(len(preds)):
                seq_len = int(word_mask[i].sum())
                true_tags = [
                    ix_to_tag.get(tags[i][j].item(), "O") for j in range(seq_len)
                ]
                pred_tags = [ix_to_tag.get(p, "O") for p in preds[i][:seq_len]]

                all_true.append(true_tags)
                all_preds.append(pred_tags)

    report_dict = seqeval_classification_report(
        all_true, all_preds, scheme=IOB2, digits=4, output_dict=True, zero_division=0
    )
    return report_dict


def train_model(
    model, train_loader, dev_loader, ix_to_tag, epochs, learning_rate, model_name
):
    """Train model with early stopping."""
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    best_f1 = 0
    patience_counter = 0
    history = []
    model_saved = False

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for (
            tokens,
            words,
            chars,
            tags,
            word_mask,
            bert_indices,
            bert_mask,
            word_maps,
        ) in train_loader:
            words = words.to(DEVICE, non_blocking=True)
            chars = chars.to(DEVICE, non_blocking=True)
            tags = tags.to(DEVICE, non_blocking=True)
            word_mask = word_mask.to(DEVICE, non_blocking=True)
            bert_indices = bert_indices.to(DEVICE, non_blocking=True)
            bert_mask = bert_mask.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()
            loss = model.forward(
                words, chars, tags, word_mask, bert_indices, bert_mask, word_maps
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Evaluate on dev set
        dev_report = evaluate_model(model, dev_loader, ix_to_tag, DEVICE)
        dev_f1 = dev_report.get("weighted avg", {}).get("f1-score", 0)

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": total_loss / len(train_loader),
                "dev_f1": dev_f1,
            }
        )

        print(
            f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}, Dev F1: {dev_f1:.4f}"
        )

        if dev_f1 > best_f1:
            best_f1 = dev_f1
            patience_counter = 0
            torch.save(model.state_dict(), OUTPUT_DIR / f"best_{model_name}.pt")
            model_saved = True
        else:
            patience_counter += 1
            if patience_counter >= HP["early_stopping_patience"]:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    if not model_saved:
        print(f"Warning: No improvement during training. Saving final model state.")
        torch.save(model.state_dict(), OUTPUT_DIR / f"best_{model_name}.pt")

    return history


def main():
    print("=== TAMSA Ablation Study ===\n")

    # Load data
    print("1. Loading data...")
    train_data = load_bio_file(TRAIN_FILE)
    dev_data = load_bio_file(DEV_FILE)
    test_data = load_bio_file(TEST_FILE)

    # Build vocabularies
    word_to_ix, tag_to_ix, ix_to_tag, char_to_ix = build_vocab(train_data + dev_data)
    tokenizer = AutoTokenizer.from_pretrained(HP["transformer_model"])

    # Create datasets
    train_dataset = NERMultiSourceDataset(
        train_data, word_to_ix, tag_to_ix, char_to_ix, tokenizer
    )
    dev_dataset = NERMultiSourceDataset(
        dev_data, word_to_ix, tag_to_ix, char_to_ix, tokenizer
    )
    test_dataset = NERMultiSourceDataset(
        test_data, word_to_ix, tag_to_ix, char_to_ix, tokenizer
    )

    loader_args_train = {
        "batch_size": HP["batch_size"],
        "shuffle": True,
        "collate_fn": collate_fn_multisource,
    }
    loader_args_eval = {"batch_size": 16, "collate_fn": collate_fn_multisource}
    if DEVICE.type == "cuda":
        loader_args_train.update(
            {
                "num_workers": min(4, os.cpu_count() or 1),
                "pin_memory": True,
                "persistent_workers": True,
                "prefetch_factor": 2,
            }
        )
        loader_args_eval.update(
            {
                "num_workers": min(4, os.cpu_count() or 1),
                "pin_memory": True,
                "persistent_workers": True,
                "prefetch_factor": 2,
            }
        )

    train_loader = DataLoader(train_dataset, **loader_args_train)
    dev_loader = DataLoader(dev_dataset, **loader_args_eval)
    test_loader = DataLoader(test_dataset, **loader_args_eval)

    # Ablation configurations
    ablation_configs = [
        {
            "name": "TAMSA (Full)",
            "variant_key": "tamsa_full",
            "model_class": MultiSource_Attention_CRF,
            "model_args": {
                "word_vocab_size": len(word_to_ix),
                "char_vocab_size": len(char_to_ix),
                "tag_to_ix": tag_to_ix,
                "hp": HP,
            },
            "load_from_trained": True,  # Load from trained TAMSA model
            "description": "Full TAMSA model with token-aware attention",
        },
        {
            "name": "w/o Token-Aware Attention",
            "variant_key": "tamsa_no_token_aware",
            "model_class": TAMSA_NoTokenAwareAttention,
            "model_args": {
                "word_vocab_size": len(word_to_ix),
                "char_vocab_size": len(char_to_ix),
                "tag_to_ix": tag_to_ix,
                "hp": HP,
            },
            "load_from_trained": False,
            "description": "TAMSA with uniform attention (1/3 each) instead of token-aware",
        },
        {
            "name": "w/o Character CNN",
            "variant_key": "tamsa_no_charcnn",
            "model_class": TAMSA_NoCharCNN,
            "model_args": {
                "word_vocab_size": len(word_to_ix),
                "char_vocab_size": len(char_to_ix),
                "tag_to_ix": tag_to_ix,
                "hp": HP,
            },
            "load_from_trained": False,
            "description": "TAMSA without Character CNN component",
        },
        {
            "name": "w/o Word2Vec",
            "variant_key": "tamsa_no_w2v",
            "model_class": TAMSA_NoWord2Vec,
            "model_args": {
                "word_vocab_size": len(word_to_ix),
                "char_vocab_size": len(char_to_ix),
                "tag_to_ix": tag_to_ix,
                "hp": HP,
            },
            "load_from_trained": False,
            "description": "TAMSA without Word2Vec embeddings",
        },
        {
            "name": "w/o IndoBERT",
            "variant_key": "tamsa_no_indobert",
            "model_class": TAMSA_NoIndoBERT,
            "model_args": {
                "word_vocab_size": len(word_to_ix),
                "char_vocab_size": len(char_to_ix),
                "tag_to_ix": tag_to_ix,
                "hp": HP,
            },
            "load_from_trained": False,
            "description": "TAMSA without IndoBERT transformer component",
        },
        {
            "name": "Uniform Attention",
            "variant_key": "tamsa_uniform_attention",
            "model_class": TAMSA_UniformAttention,
            "model_args": {
                "word_vocab_size": len(word_to_ix),
                "char_vocab_size": len(char_to_ix),
                "tag_to_ix": tag_to_ix,
                "hp": HP,
            },
            "load_from_trained": False,
            "description": "TAMSA with fixed uniform attention weights (no token-aware)",
        },
    ]

    results = []

    # Process TAMSA Full first (baseline)
    print("\n2. Processing TAMSA (Full) baseline...")
    full_config = ablation_configs[0]

    # Try to load from trained model
    tamsa_model_path = _ROOT / "outputs" / "experiment_attention_fusion" / "attention-fusion-crf-best.pt"
    tamsa_vocab_path = _ROOT / "outputs" / "experiment_attention_fusion" / "vocab.json"
    tamsa_hp_path = _ROOT / "outputs" / "experiment_attention_fusion" / "hyperparameters.json"

    if full_config["load_from_trained"] and tamsa_model_path.exists():
        print("   Loading trained TAMSA model...")
        model = full_config["model_class"](**full_config["model_args"]).to(DEVICE)

        # Load vocab and HP if available
        if tamsa_vocab_path.exists():
            with open(tamsa_vocab_path, "r") as f:
                vocab_data = json.load(f)
            # Use vocab from trained model if compatible

        if tamsa_hp_path.exists():
            with open(tamsa_hp_path, "r") as f:
                saved_hp = json.load(f)
            # Use saved HP if available

        model.load_state_dict(
            torch.load(tamsa_model_path, map_location=DEVICE, weights_only=True)
        )
        print("   TAMSA model loaded successfully!")

        # Evaluate directly without training
        print("   Evaluating TAMSA (Full) on test set...")
        test_report = evaluate_model(model, test_loader, ix_to_tag, DEVICE)

        result = {
            "variant": full_config["name"],
            "variant_key": full_config["variant_key"],
            "description": full_config["description"],
            "f1_score": test_report.get("weighted avg", {}).get("f1-score", 0),
            "precision": test_report.get("weighted avg", {}).get("precision", 0),
            "recall": test_report.get("weighted avg", {}).get("recall", 0),
            "trained": False,  # Loaded from pre-trained
        }
        results.append(result)
        print(f"   Test F1-Score: {result['f1_score']:.4f}")
    else:
        print("   Trained TAMSA model not found. Training from scratch...")
        model = full_config["model_class"](**full_config["model_args"]).to(DEVICE)

        start_time = time.time()
        history = train_model(
            model,
            train_loader,
            dev_loader,
            ix_to_tag,
            HP["epochs"],
            HP["learning_rate"],
            full_config["variant_key"],
        )
        training_time = time.time() - start_time

        # Load best model
        model.load_state_dict(
            torch.load(
                OUTPUT_DIR / f"best_{full_config['variant_key']}.pt", weights_only=True
            )
        )

        # Evaluate on test set
        test_report = evaluate_model(model, test_loader, ix_to_tag, DEVICE)

        result = {
            "variant": full_config["name"],
            "variant_key": full_config["variant_key"],
            "description": full_config["description"],
            "f1_score": test_report.get("weighted avg", {}).get("f1-score", 0),
            "precision": test_report.get("weighted avg", {}).get("precision", 0),
            "recall": test_report.get("weighted avg", {}).get("recall", 0),
            "training_time": training_time,
            "trained": True,
        }
        results.append(result)
        print(f"   Test F1-Score: {result['f1_score']:.4f}")

    full_tamsa_f1 = results[0]["f1_score"]

    # Process ablation variants
    print("\n3. Training and evaluating ablation variants...")
    for config in ablation_configs[1:]:  # Skip full model (already processed)
        print(f"\n   Processing: {config['name']}")
        print(f"   {config['description']}")

        # Initialize model
        try:
            model = config["model_class"](**config["model_args"]).to(DEVICE)
        except Exception as e:
            print(f"   ERROR: Failed to initialize model: {e}")
            continue

        # Count parameters
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   Parameters: {num_params:,}")

        # Train model
        start_time = time.time()
        history = train_model(
            model,
            train_loader,
            dev_loader,
            ix_to_tag,
            HP["epochs"],
            HP["learning_rate"],
            config["variant_key"],
        )
        training_time = time.time() - start_time

        # Load best model
        model_path = OUTPUT_DIR / f"best_{config['variant_key']}.pt"
        if model_path.exists():
            model.load_state_dict(
                torch.load(model_path, map_location=DEVICE, weights_only=True)
            )
        else:
            print(f"   Warning: Model file not found, using current state")

        # Evaluate on test set
        test_report = evaluate_model(model, test_loader, ix_to_tag, DEVICE)

        f1_score = test_report.get("weighted avg", {}).get("f1-score", 0)
        delta_from_full = full_tamsa_f1 - f1_score

        result = {
            "variant": config["name"],
            "variant_key": config["variant_key"],
            "description": config["description"],
            "f1_score": f1_score,
            "precision": test_report.get("weighted avg", {}).get("precision", 0),
            "recall": test_report.get("weighted avg", {}).get("recall", 0),
            "delta_from_full": delta_from_full,
            "num_params": num_params,
            "training_time": training_time,
            "trained": True,
        }
        results.append(result)

        print(f"   Test F1-Score: {f1_score:.4f}")
        print(f"   Δ from Full: -{delta_from_full:.4f}")

    # Add delta for full model (0.0)
    results[0]["delta_from_full"] = 0.0

    # Create results dataframe
    results_df = pd.DataFrame(results)

    # Sort by delta_from_full (largest drop first, full model last)
    results_df_sorted = results_df.sort_values("delta_from_full", ascending=False)
    # Move full model to first row
    full_row = results_df_sorted[results_df_sorted["variant"] == "TAMSA (Full)"]
    other_rows = results_df_sorted[results_df_sorted["variant"] != "TAMSA (Full)"]
    results_df_final = pd.concat([full_row, other_rows]).reset_index(drop=True)

    # Format F1-score (add ± if we have std, otherwise just the value)
    # For now, single run, so no std
    results_df_final["f1_score_formatted"] = results_df_final["f1_score"].apply(
        lambda x: f"{x:.2f}"
    )

    # Select columns for final CSV (according to mapping document)
    csv_columns = ["variant", "f1_score", "delta_from_full"]
    final_csv = results_df_final[csv_columns].copy()
    final_csv.columns = ["Variant", "F1-Score", "Δ from Full"]

    # Save CSV
    final_csv.to_csv(OUTPUT_DIR / "ablation_results.csv", index=False)
    print(f"\n4. Results saved to: {OUTPUT_DIR / 'ablation_results.csv'}")

    # Also save full results
    results_df_final.to_csv(OUTPUT_DIR / "ablation_results_full.csv", index=False)

    # Create visualization (Figure 5 requirement)
    print("\n5. Creating visualization...")

    plt.figure(figsize=(10, 6))

    # Prepare data
    variants = results_df_final["variant"].tolist()
    f1_scores = results_df_final["f1_score"].tolist()
    deltas = results_df_final["delta_from_full"].tolist()

    # Create horizontal bar chart
    colors = [
        (
            "green"
            if v == "TAMSA (Full)"
            else "red" if "Token-Aware" in v or "Uniform" in v else "steelblue"
        )
        for v in variants
    ]

    bars = plt.barh(range(len(variants)), f1_scores, color=colors, alpha=0.8)

    # Highlight TAMSA Full
    if "TAMSA (Full)" in variants:
        full_idx = variants.index("TAMSA (Full)")
        bars[full_idx].set_edgecolor("darkgreen")
        bars[full_idx].set_linewidth(2)

    # Add delta annotations (except for full model)
    for i, (variant, delta) in enumerate(zip(variants, deltas)):
        if variant != "TAMSA (Full)":
            plt.text(
                f1_scores[i] + 0.3,
                i,
                f"Δ = -{delta:.2f}",
                va="center",
                fontsize=9,
                fontweight="bold",
            )

    # Customize plot
    plt.yticks(range(len(variants)), variants)
    plt.xlabel("F1-Score", fontsize=12)
    plt.title(
        "Ablation Study: Contribution of Each Component in TAMSA",
        fontsize=14,
        fontweight="bold",
    )
    plt.grid(axis="x", alpha=0.3, linestyle="--")
    plt.xlim(0, max(f1_scores) * 1.15)

    # Add legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="green", alpha=0.8, label="TAMSA (Full)"),
        Patch(facecolor="red", alpha=0.8, label="Attention variants (key)"),
        Patch(facecolor="steelblue", alpha=0.8, label="Other variants"),
    ]
    plt.legend(handles=legend_elements, loc="lower right", fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ablation_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"   Visualization saved to: {OUTPUT_DIR / 'ablation_comparison.png'}")

    # Print summary
    print("\n" + "=" * 70)
    print("ABLATION STUDY SUMMARY")
    print("=" * 70)
    print(f"\n{'Variant':<35} {'F1-Score':<12} {'Δ from Full':<12}")
    print("-" * 70)
    for _, row in results_df_final.iterrows():
        delta_str = (
            f"-{row['delta_from_full']:.2f}" if row["delta_from_full"] > 0 else "-"
        )
        print(f"{row['variant']:<35} {row['f1_score']:<12.4f} {delta_str:<12}")

    print("\n" + "=" * 70)
    print(f"Results saved to: {OUTPUT_DIR}")
    print(f"  - ablation_results.csv (for paper)")
    print(f"  - ablation_results_full.csv (detailed)")
    print(f"  - ablation_comparison.png (Figure 5)")


if __name__ == "__main__":
    main()
