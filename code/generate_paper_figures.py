import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from pathlib import Path
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
matplotlib.rcParams["font.family"] = "Times New Roman"
matplotlib.rcParams["font.size"] = 10

OUTPUT_DIR = _ROOT / "outputs"
FIGURES_DIR = _ROOT / "paper_figures"
FIGURES_DIR.mkdir(exist_ok=True)

MODELS = {
    "experiment_bilstm": "BiLSTM-CRF",
    "experiment_bilstm_w2v": "BiLSTM+W2V",
    "experiment_bilstm_w2v_cnn": "BiLSTM+W2V+CNN",
    "experiment_indobert": "IndoBERT-CRF",
    "experiment_indobert_bilstm": "IndoBERT+BiLSTM",
    "experiment_mbert_bilstm": "mBERT",
    "experiment_xlm_roberta_bilstm": "XLM-RoBERTa+BiLSTM",
    "experiment_attention_fusion": "TAMSA (Ours)",
}

COLORS = sns.color_palette("husl", len(MODELS))
MODEL_COLORS = dict(zip(MODELS.values(), COLORS))


def load_results():
    """Load experimental results from output directories."""
    results = {}
    for dir_name, display_name in MODELS.items():
        path = OUTPUT_DIR / dir_name
        if not path.exists():
            continue

        result_data = {}

        # Load classification report
        report_path = path / "classification_report.json"
        if report_path.exists():
            with open(report_path, "r") as f:
                result_data["report"] = json.load(f)

        # Load summary report (for efficiency metrics)
        summary_path = path / "summary_report.json"
        if summary_path.exists():
            with open(summary_path, "r") as f:
                result_data["summary"] = json.load(f)

        # Load training history
        history_path = path / "training_history.csv"
        if history_path.exists():
            result_data["history"] = pd.read_csv(history_path)

        if result_data:
            results[display_name] = result_data

    return results


def plot_performance_comparison(results):
    """Generate performance comparison plot from actual experimental results."""
    print("Generating Figure 3: Performance Comparison...")
    data = []

    model_order = [
        "BiLSTM-CRF",
        "BiLSTM+W2V",
        "BiLSTM+W2V+CNN",
        "IndoBERT-CRF",
        "IndoBERT+BiLSTM",
        "XLM-RoBERTa+BiLSTM",
        "TAMSA (Ours)",
    ]

    for model_name in model_order:
        if model_name not in results:
            continue
        content = results[model_name]
        if "report" not in content:
            continue

        report = content["report"]
        weighted_avg = report.get("weighted avg", {})

        if weighted_avg:
            data.append(
                {
                    "Model": model_name,
                    "F1-Score": weighted_avg.get("f1-score", 0),
                    "Precision": weighted_avg.get("precision", 0),
                    "Recall": weighted_avg.get("recall", 0),
                }
            )

    if not data:
        print("Warning: No performance data found. Skipping comparison plot.")
        return

    models = [d["Model"] for d in data]
    f1_scores = [d["F1-Score"] for d in data]
    precision_scores = [d["Precision"] for d in data]
    recall_scores = [d["Recall"] for d in data]

    fig_width = 6.5
    fig_height = 3.6
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    x = np.arange(len(models))
    width = 0.28

    bars1 = ax.bar(
        x - width,
        f1_scores,
        width,
        label="F1-Score",
        color="#1f77b4",
        edgecolor="none",
        linewidth=0,
    )
    bars2 = ax.bar(
        x,
        precision_scores,
        width,
        label="Precision",
        color="#ff7f0e",
        edgecolor="none",
        linewidth=0,
    )
    bars3 = ax.bar(
        x + width,
        recall_scores,
        width,
        label="Recall",
        color="#2ca02c",
        edgecolor="none",
        linewidth=0,
    )

    ax.set_ylabel("Score", fontsize=10, fontname="Times New Roman")
    ax.set_xlabel("Model", fontsize=10, fontname="Times New Roman")
    ax.set_xticks(x)
    ax.set_xticklabels(
        models, rotation=45, ha="right", fontsize=9, fontname="Times New Roman"
    )
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(
        ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"],
        fontsize=9,
        fontname="Times New Roman",
    )

    ax.legend(
        loc="upper center",
        fontsize=9,
        frameon=False,
        prop={"family": "Times New Roman", "size": 9},
        ncol=3,
        columnspacing=1.2,
        handletextpad=0.5,
        bbox_to_anchor=(0.5, 1.15),
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)

    ax.grid(True, axis="y", linestyle="--", linewidth=0.3, alpha=0.2, color="gray")
    ax.set_axisbelow(True)

    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    plt.tight_layout(pad=0.5, rect=[0, 0, 1, 0.92])

    plt.savefig(
        FIGURES_DIR / "figure3_performance_comparison.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.05,
        facecolor="white",
        edgecolor="none",
    )
    plt.close()


def plot_learning_curves(results):
    """Generate learning curves plot from actual training history."""
    print("Generating Figure 4: Learning Curves...")

    has_data = False
    plt.figure(figsize=(14, 6))

    # Validation F1
    plt.subplot(1, 2, 1)
    for model_name, content in results.items():
        if "history" in content:
            df = content["history"]
            if "epoch" in df.columns and "dev_f1" in df.columns:
                plt.plot(
                    df["epoch"],
                    df["dev_f1"],
                    label=model_name,
                    marker="o",
                    markersize=4,
                    linewidth=2,
                )
                has_data = True

    if not has_data:
        plt.close()
        print("Warning: No training history data found. Skipping learning curves plot.")
        return

    plt.title("Validation F1-Score over Epochs", fontsize=14)
    plt.xlabel("Epoch")
    plt.ylabel("F1-Score")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)

    # Training Loss
    plt.subplot(1, 2, 2)
    for model_name, content in results.items():
        if "history" in content:
            df = content["history"]
            if "epoch" in df.columns and "train_loss" in df.columns:
                plt.plot(
                    df["epoch"],
                    df["train_loss"],
                    label=model_name,
                    linestyle="--",
                    linewidth=2,
                )

    plt.title("Training Loss over Epochs", fontsize=14)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout(pad=2.0)
    plt.savefig(
        FIGURES_DIR / "figure4_learning_curves.png", dpi=300, bbox_inches="tight"
    )
    plt.close()


def plot_confusion_matrix_tamsa():
    """Generate confusion matrix plot for TAMSA model."""
    print("Generating Figure 5: Confusion Matrix (TAMSA)...")

    tamsa_dir = OUTPUT_DIR / "experiment_attention_fusion"
    cm_path = tamsa_dir / "confusion_matrix.csv"

    if not cm_path.exists():
        for model_dir in [
            OUTPUT_DIR / "experiment_attention_fusion",
            OUTPUT_DIR / "experiment_indobert_bilstm",
            OUTPUT_DIR / "experiment_bilstm",
        ]:
            alt_path = model_dir / "confusion_matrix.csv"
            if alt_path.exists():
                cm_path = alt_path
                model_name = model_dir.name.replace("experiment_", "").replace("_", " ")
                title = f"Confusion Matrix ({model_name.title()})"
                break
        else:
            print("Warning: No confusion matrix found. Skipping confusion matrix plot.")
            return
    else:
        title = "Confusion Matrix (TAMSA)"

    try:
        cm_df = pd.read_csv(cm_path, index_col=0)

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.title(title, fontsize=16, pad=20)
        plt.ylabel("Actual Label", fontsize=12)
        plt.xlabel("Predicted Label", fontsize=12)
        plt.tight_layout(pad=2.0)
        plt.savefig(
            FIGURES_DIR / "figure5_confusion_matrix.png", dpi=300, bbox_inches="tight"
        )
        plt.close()
    except Exception as e:
        print(f"Error generating confusion matrix plot: {e}")
        import traceback

        traceback.print_exc()


def plot_ablation_study():
    """Generate ablation study figure from actual experimental results."""
    print("Generating Figure 6: Ablation Study...")

    ablation_path = _ROOT / "outputs" / "ablation_studies" / "ablation_results.csv"

    if not ablation_path.exists():
        print(f"Warning: {ablation_path} not found. Skipping ablation study plot.")
        return

    try:
        df = pd.read_csv(ablation_path)

        if "Variant" not in df.columns or "F1-Score" not in df.columns:
            print(
                f"Warning: Expected columns 'Variant' and 'F1-Score' not found in {ablation_path}"
            )
            return

        plt.figure(figsize=(12, 7))
        ax = sns.barplot(
            data=df,
            x="Variant",
            y="F1-Score",
            hue="Variant",
            palette="rocket",
            legend=False,
        )
        plt.title("Ablation Study: Impact of Components", fontsize=16, pad=20)

        y_min = df["F1-Score"].min() * 0.98
        y_max = df["F1-Score"].max() * 1.02
        plt.ylim(max(0.8, y_min), min(1.0, y_max))

        for i, row in df.iterrows():
            ax.text(
                i,
                row["F1-Score"] + (y_max - y_min) * 0.01,
                f"{row['F1-Score']:.4f}",
                ha="center",
                fontsize=12,
            )

        plt.xticks(rotation=45, ha="right")
        plt.subplots_adjust(left=0.12, right=0.95, top=0.88, bottom=0.25)
        plt.savefig(
            FIGURES_DIR / "figure6_ablation_study.png", dpi=300, bbox_inches="tight"
        )
        plt.close()
    except Exception as e:
        print(f"Error generating ablation study plot: {e}")
        import traceback

        traceback.print_exc()


def plot_efficiency_tradeoff(results):
    """Generate efficiency vs accuracy trade-off plot from actual experimental results."""
    print("Generating Figure 7: Efficiency vs Accuracy...")

    data = []
    baseline_time = None

    for model_name, content in results.items():
        if "report" not in content or "summary" not in content:
            continue

        report = content["report"]
        f1_score = report.get("weighted avg", {}).get("f1-score", 0)
        summary = content["summary"]
        training_time = summary.get("training_duration_seconds", 0)

        if training_time > 0 and f1_score > 0:
            if baseline_time is None:
                baseline_time = training_time

            data.append(
                {
                    "Model": model_name,
                    "F1": f1_score,
                    "Time": (
                        training_time / baseline_time
                        if baseline_time > 0
                        else training_time
                    ),
                }
            )

    if not data:
        print("Warning: No efficiency data found. Skipping efficiency trade-off plot.")
        return

    df = pd.DataFrame(data)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df, x="Time", y="F1", s=200, hue="Model", style="Model", palette="deep"
    )

    for i, row in df.iterrows():
        plt.text(row["Time"] + 0.1, row["F1"], row["Model"], fontsize=11)

    plt.title("Efficiency vs Accuracy Trade-off", fontsize=16)
    plt.xlabel("Normalized Training Time (relative to baseline)", fontsize=12)
    plt.ylabel("F1-Score", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.15)
    plt.savefig(
        FIGURES_DIR / "figure7_efficiency_tradeoff.png", dpi=300, bbox_inches="tight"
    )
    plt.close()


def main():
    """Main function to generate all paper figures from experimental results."""
    print("Starting figure generation...")
    results = load_results()

    if not results:
        print("Warning: No results found! Please run experiments first.")
        return

    plot_performance_comparison(results)
    plot_learning_curves(results)
    plot_confusion_matrix_tamsa()
    plot_ablation_study()
    plot_efficiency_tradeoff(results)

    print(f"\nAll figures generated in {FIGURES_DIR.absolute()}")


if __name__ == "__main__":
    main()
