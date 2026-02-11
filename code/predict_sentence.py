import json
import torch
import torch.nn as nn
from pathlib import Path
from transformers import AutoTokenizer, AutoModel

_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT_DIR = _ROOT / "outputs" / "experiment_attention_fusion"
BEST_MODEL_PATH = EXPERIMENT_DIR / "attention-fusion-crf-best.pt"
VOCAB_PATH = EXPERIMENT_DIR / "vocab.json"
HP_PATH = EXPERIMENT_DIR / "hyperparameters.json"

if not all([BEST_MODEL_PATH.exists(), VOCAB_PATH.exists(), HP_PATH.exists()]):
    print(f"Error: Model, vocab, or hyperparameter file not found in {EXPERIMENT_DIR}")
    print("Make sure you have run the 'Attention Fusion' experiment first.")
    exit()

with open(HP_PATH, "r") as f:
    HP = json.load(f)

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

print(f"Using device: {DEVICE}")

# Architecture must match train_attention_fusion.py.


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
    """Attention mechanism for fusing multiple feature sources."""

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
    """Multi-source attention fusion model with CRF for Named Entity Recognition."""

    def __init__(self, word_vocab_size, char_vocab_size, tag_to_ix, hp):
        super(MultiSource_Attention_CRF, self).__init__()
        import warnings

        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=UserWarning, module="torch")
        warnings.filterwarnings("ignore", category=DeprecationWarning, module="torch")
        from torchcrf import CRF

        self.hp = hp
        self.tagset_size = len(tag_to_ix)
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
        self.lstm = nn.LSTM(
            self.projection_dim,
            self.hp["lstm_hidden_dim"] // 2,
            num_layers=self.hp["lstm_layers"],
            bidirectional=True,
            batch_first=True,
        )
        self.dropout = nn.Dropout(self.hp["dropout"])
        self.hidden2tag = nn.Linear(self.hp["lstm_hidden_dim"], self.tagset_size)
        self.crf = CRF(self.tagset_size, batch_first=True)

    def _get_features(self, words, chars, bert_indices, bert_mask, word_maps):
        w2v_features = self.w2v_embedding(words)
        char_cnn_features = self.char_cnn(chars)
        transformer_outputs = self.transformer(bert_indices, attention_mask=bert_mask)
        transformer_hidden_states = transformer_outputs.last_hidden_state
        batch_size, seq_len, _ = w2v_features.shape
        transformer_features = torch.zeros(
            batch_size, seq_len, self.hp["transformer_dim"], device=DEVICE
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
        fused_features, _ = self.fusion([w2v_proj, char_proj, trans_proj])
        fused_features = self.dropout(fused_features)
        lstm_out, _ = self.lstm(fused_features)
        lstm_out = self.dropout(lstm_out)
        emissions = self.hidden2tag(lstm_out)
        return emissions

    def predict(self, words, chars, word_mask, bert_indices, bert_mask, word_maps):
        emissions = self._get_features(words, chars, bert_indices, bert_mask, word_maps)
        return self.crf.decode(emissions, mask=word_mask)


def predict_sentence(model, sentence, word_to_ix, char_to_ix, ix_to_tag, tokenizer):
    """Takes model and sentence string, returns list of (token, tag)."""
    model.eval()

    tokens = sentence.split()
    if not tokens:
        return []

    word_indices = [word_to_ix.get(t.lower(), 1) for t in tokens]
    char_indices = [[char_to_ix.get(c, 1) for c in token] for token in tokens]

    bert_subwords = []
    word_to_subword_map = []
    for token in tokens:
        subws = tokenizer.tokenize(token)
        word_to_subword_map.append(len(bert_subwords))
        bert_subwords.extend(subws)
    bert_indices = tokenizer.convert_tokens_to_ids(bert_subwords)

    # Create batch with size 1
    # Convert all data to tensors and add batch dimension
    padded_words = torch.tensor(word_indices, dtype=torch.long).unsqueeze(0)

    max_char_len = max(len(c) for c in char_indices) if char_indices else 1
    padded_chars = torch.zeros(1, len(tokens), max_char_len, dtype=torch.long)
    for i, cl in enumerate(char_indices):
        l = len(cl)
        padded_chars[0, i, :l] = torch.tensor(cl)

    padded_bert_indices = torch.tensor(bert_indices, dtype=torch.long).unsqueeze(0)
    bert_attention_mask = torch.ones_like(padded_bert_indices)
    word_mask = torch.ones(1, len(tokens), dtype=torch.bool)
    word_maps = [word_to_subword_map]

    # Move all tensors to appropriate device
    padded_words, padded_chars, word_mask = (
        padded_words.to(DEVICE, non_blocking=True),
        padded_chars.to(DEVICE, non_blocking=True),
        word_mask.to(DEVICE, non_blocking=True),
    )
    padded_bert_indices, bert_attention_mask = padded_bert_indices.to(
        DEVICE, non_blocking=True
    ), bert_attention_mask.to(DEVICE, non_blocking=True)

    # Perform prediction
    with torch.no_grad():
        predicted_indices = model.predict(
            padded_words,
            padded_chars,
            word_mask,
            padded_bert_indices,
            bert_attention_mask,
            word_maps,
        )

    # Convert indices back to tags
    predicted_tags = [ix_to_tag.get(p, "<UNK>") for p in predicted_indices[0]]

    return list(zip(tokens, predicted_tags))


def main():
    """Main function to load model and run interactive loop."""
    print("\nLoading vocabulary and tokenizer...")
    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        vocab_data = json.load(f)
    word_to_ix = vocab_data["word_to_ix"]
    tag_to_ix = vocab_data["tag_to_ix"]
    ix_to_tag = {int(k): v for k, v in vocab_data["ix_to_tag"].items()}
    char_to_ix = vocab_data["char_to_ix"]
    tokenizer = AutoTokenizer.from_pretrained(HP["transformer_model"])

    print("Loading best Attention Fusion model...")
    model = MultiSource_Attention_CRF(len(word_to_ix), len(char_to_ix), tag_to_ix, HP)
    model.load_state_dict(
        torch.load(BEST_MODEL_PATH, map_location=DEVICE, weights_only=True)
    )
    model.to(DEVICE)

    print("Model ready.")

    print("\n--- Welcome to Interactive Prediction Mode ---")
    print("Type your sentence and press Enter. Type 'exit' or 'quit' to exit.")

    while True:
        sentence = input("\nSentence > ")
        if sentence.lower() in ["exit", "quit"]:
            break

        predictions = predict_sentence(
            model, sentence, word_to_ix, char_to_ix, ix_to_tag, tokenizer
        )

        print("\nPrediction Results:")
        # Display results in neat format
        for token, tag in predictions:
            print(f"{token:<20} {tag}")

    print("\nThank you! Exiting program.")


if __name__ == "__main__":
    main()
