import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

ENTITY_TYPES = ["PER", "ORG", "LOC", "MISC"]

# Model configurations
MODELS = {
    "bilstm": "BiLSTM-CRF",
    "bilstm_w2v": "BiLSTM + Word2Vec",
    "bilstm_w2v_cnn": "BiLSTM + Word2Vec + CharCNN",
    "indobert": "IndoBERT-CRF",
    "indobert_bilstm": "IndoBERT + BiLSTM-CRF",
    "mbert_bilstm": "mBERT + BiLSTM-CRF",
    "xlm_roberta_bilstm": "XLM-RoBERTa + BiLSTM-CRF",
    "attention_fusion": "TAMSA",
}


def load_per_entity_results(model_key: str, dataset: str) -> Dict:
    """Load per-entity evaluation results from classification report."""
    result_path = Path(f"outputs/{model_key}/eval_{dataset}/classification_report.json")
    
    if not result_path.exists():
        result_path = Path(f"outputs/{model_key}/eval_{dataset}/results.json")
    
    if not result_path.exists():
        result_path = Path(f"outputs/experiment_{model_key}/classification_report.json")
    
    if not result_path.exists():
        return None

    try:
        with open(result_path, "r") as f:
            results = json.load(f)
        return results
    except Exception as e:
        print(f"Error loading {result_path}: {e}")
        return None


def create_per_entity_dataframe() -> pd.DataFrame:
    """Create dataframe with per-entity metrics for all models."""
    results = []

    datasets = ["idner2k", "nerugm", "nerui"]
    skip_keys = {"macro avg", "weighted avg", "micro avg", "accuracy", "O"}

    for model_key, model_name in MODELS.items():
        for dataset in datasets:
            result = load_per_entity_results(model_key, dataset)

            if result and isinstance(result, dict):
                found_entities = False
                for key, metrics in result.items():
                    if key in skip_keys:
                        continue
                    
                    if isinstance(metrics, dict) and "f1-score" in metrics:
                        found_entities = True
                        entity_type = key
                        if "-" in key:
                            parts = key.split("-")
                            if len(parts) == 2 and parts[0] in ["B", "I"]:
                                entity_type = parts[1]
                            else:
                                entity_type = parts[-1]
                        
                        results.append(
                            {
                                "Model": model_name,
                                "Dataset": dataset.upper(),
                                "Entity Type": entity_type,
                                "Precision": metrics.get("precision", 0),
                                "Recall": metrics.get("recall", 0),
                                "F1-Score": metrics.get("f1-score", 0),
                                "Support": metrics.get("support", 0),
                            }
                        )
                
                if not found_entities:
                    for entity_type in ENTITY_TYPES:
                        results.append(
                            {
                                "Model": model_name,
                                "Dataset": dataset.upper(),
                                "Entity Type": entity_type,
                                "Precision": np.nan,
                                "Recall": np.nan,
                                "F1-Score": np.nan,
                                "Support": np.nan,
                            }
                        )
            else:
                for entity_type in ENTITY_TYPES:
                    results.append(
                        {
                            "Model": model_name,
                            "Dataset": dataset.upper(),
                            "Entity Type": entity_type,
                            "Precision": np.nan,
                            "Recall": np.nan,
                            "F1-Score": np.nan,
                            "Support": np.nan,
                        }
                    )

    return pd.DataFrame(results)


def create_summary_tables(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Create summary tables for per-entity analysis."""
    summaries = {}

    # Average performance per entity type across all models
    entity_avg = (
        df.groupby("Entity Type")[["Precision", "Recall", "F1-Score"]]
        .mean()
        .reset_index()
    )
    entity_avg = entity_avg.sort_values("F1-Score", ascending=False)
    summaries["entity_average"] = entity_avg

    # Best model per entity type
    best_per_entity = (
        df.groupby("Entity Type")
        .apply(lambda x: x.loc[x["F1-Score"].idxmax()])[
            ["Model", "Dataset", "F1-Score"]
        ]
        .reset_index()
    )
    summaries["best_per_entity"] = best_per_entity

    # Model performance per entity type
    model_entity = df.groupby(["Model", "Entity Type"])["F1-Score"].mean().reset_index()
    model_entity_pivot = model_entity.pivot(
        index="Model", columns="Entity Type", values="F1-Score"
    )
    summaries["model_entity_matrix"] = model_entity_pivot

    # Hardest entities (lowest average F1)
    hardest_entities = (
        df.groupby("Entity Type")["F1-Score"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    hardest_entities = hardest_entities.sort_values("mean")
    summaries["hardest_entities"] = hardest_entities

    # Entity distribution (support)
    entity_dist = df.groupby("Entity Type")["Support"].sum().reset_index()
    entity_dist = entity_dist.sort_values("Support", ascending=False)
    summaries["entity_distribution"] = entity_dist

    return summaries


def create_visualizations(
    df: pd.DataFrame, summaries: Dict[str, pd.DataFrame], output_dir: Path
):
    """Create visualizations for per-entity analysis."""
    sns.set_style("whitegrid")
    plt.rcParams["figure.figsize"] = (12, 8)

    # Heatmap: Model performance per entity type
    if "model_entity_matrix" in summaries:
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            summaries["model_entity_matrix"],
            annot=True,
            fmt=".3f",
            cmap="YlOrRd",
            cbar_kws={"label": "F1-Score"},
        )
        plt.title(
            "Model Performance per Entity Type (F1-Score)",
            fontsize=14,
            fontweight="bold",
        )
        plt.xlabel("Entity Type", fontsize=12)
        plt.ylabel("Model", fontsize=12)
        plt.tight_layout()
        plt.savefig(
            output_dir / "model_entity_heatmap.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    # Bar plot: Average F1 per entity type
    if "entity_average" in summaries:
        plt.figure(figsize=(10, 6))
        entity_avg = summaries["entity_average"]
        plt.bar(entity_avg["Entity Type"], entity_avg["F1-Score"], color="steelblue")
        plt.xlabel("Entity Type", fontsize=12)
        plt.ylabel("Average F1-Score", fontsize=12)
        plt.title(
            "Average F1-Score per Entity Type (Across All Models)",
            fontsize=14,
            fontweight="bold",
        )
        plt.xticks(rotation=45)
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "entity_average_f1.png", dpi=300, bbox_inches="tight")
        plt.close()

    # Comparison: Precision vs Recall per entity
    if "entity_average" in summaries:
        plt.figure(figsize=(10, 6))
        entity_avg = summaries["entity_average"]
        x = np.arange(len(entity_avg))
        width = 0.35

        plt.bar(
            x - width / 2,
            entity_avg["Precision"],
            width,
            label="Precision",
            color="skyblue",
        )
        plt.bar(
            x + width / 2,
            entity_avg["Recall"],
            width,
            label="Recall",
            color="lightcoral",
        )

        plt.xlabel("Entity Type", fontsize=12)
        plt.ylabel("Score", fontsize=12)
        plt.title("Precision vs Recall per Entity Type", fontsize=14, fontweight="bold")
        plt.xticks(x, entity_avg["Entity Type"], rotation=45)
        plt.legend()
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            output_dir / "precision_recall_comparison.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    print(f"Visualizations saved to {output_dir}/")


def save_results(
    df: pd.DataFrame, summaries: Dict[str, pd.DataFrame], output_dir: Path
):
    """Save all results to files."""
    df.to_csv(output_dir / "per_entity_full_results.csv", index=False)
    df.to_excel(output_dir / "per_entity_full_results.xlsx", index=False)

    # Save summaries
    with pd.ExcelWriter(output_dir / "per_entity_summaries.xlsx") as writer:
        for name, summary_df in summaries.items():
            summary_df.to_excel(writer, sheet_name=name, index=False)

    # Save JSON
    results_dict = {
        "full_results": df.to_dict("records"),
        "summaries": {
            k: v.to_dict("records") if isinstance(v, pd.DataFrame) else v.to_dict()
            for k, v in summaries.items()
        },
    }

    with open(output_dir / "per_entity_results.json", "w") as f:
        json.dump(results_dict, f, indent=2, default=str)

    print(f"Results saved to {output_dir}/")


def print_summary_statistics(df: pd.DataFrame, summaries: Dict[str, pd.DataFrame]):
    """Print summary statistics to console."""
    print("\n" + "=" * 80)
    print("PER-ENTITY TYPE ANALYSIS SUMMARY")
    print("=" * 80)

    print("\nAverage Performance per Entity Type:")
    print(summaries["entity_average"].to_string(index=False))

    print("\nBest Model per Entity Type:")
    print(summaries["best_per_entity"].to_string(index=False))

    print("\nHardest Entities (Lowest Average F1):")
    print(summaries["hardest_entities"].to_string(index=False))

    print("\nEntity Distribution (Support):")
    print(summaries["entity_distribution"].to_string(index=False))

    print("\n" + "=" * 80)


def main():
    """Main function to run per-entity analysis."""
    print("Starting Per-Entity Type Analysis...")
    print("This will analyze model performance for each entity type.\n")

    df = create_per_entity_dataframe()

    if df.empty or df["F1-Score"].isna().all():
        print("WARNING: No per-entity results found!")
        print(
            "Note: You may need to modify your evaluation scripts to output per-entity metrics."
        )
        print(
            "This script expects results in format with 'per_entity' field in JSON results."
        )
        return

    summaries = create_summary_tables(df)
    print_summary_statistics(df, summaries)

    output_dir = Path("outputs/per_entity_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    create_visualizations(df, summaries, output_dir)
    save_results(df, summaries, output_dir)

    print("\nPer-entity analysis completed!")


if __name__ == "__main__":
    main()
