import os
import sys
import json
import subprocess
import time
import shutil
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

_ROOT = Path(__file__).resolve().parent.parent
from datetime import datetime
import torch

warnings.filterwarnings("ignore", category=RuntimeWarning, module="scipy.stats")

CHECKPOINT_FILE = "multiple_runs_progress.json"
OUTPUT_BASE_DIR = _ROOT / "outputs"
MULTIPLE_RUNS_DIR = OUTPUT_BASE_DIR / "multiple_runs"
MULTIPLE_RUNS_DIR.mkdir(parents=True, exist_ok=True)

# Default settings
DEFAULT_NUM_RUNS = 5
DEFAULT_SEEDS = [42, 123, 456, 789, 1011]

# Training scripts mapping
TRAINING_SCRIPTS = {
    "train_bilstm.py": "experiment_bilstm",
    "train_bilstm_w2v.py": "experiment_bilstm_w2v",
    "train_bilstm_w2v_cnn.py": "experiment_bilstm_w2v_cnn",
    "train_indobert.py": "experiment_indobert",
    "train_indobert_bilstm.py": "experiment_indobert_bilstm",
    "train_mbert_bilstm.py": "experiment_mbert_bilstm",
    "train_xlm_roberta_bilstm.py": "experiment_xlm_roberta_bilstm",
    "train_attention_fusion.py": "experiment_attention_fusion",
}

# Plot style
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 12,
        "figure.dpi": 300,
    }
)


def load_checkpoint():
    """Load progress from checkpoint file."""
    if Path(CHECKPOINT_FILE).exists():
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "models": {},
        "start_time": datetime.now().isoformat(),
        "num_runs": DEFAULT_NUM_RUNS,
        "seeds": DEFAULT_SEEDS,
    }


def save_checkpoint(progress):
    """Save progress to checkpoint file."""
    progress["last_updated"] = datetime.now().isoformat()
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2)


def get_model_name_from_script(script_name):
    """Get model name from script name."""
    return (
        script_name.replace("train_", "").replace(".py", "").replace("_", " ").title()
    )


def check_existing_runs(model_output_dir, num_runs):
    """Return list of completed run IDs for resume support."""
    completed_runs = []
    for run_id in range(1, num_runs + 1):
        run_dir = model_output_dir / f"run_{run_id}"
        summary_path = run_dir / "summary_report.json"
        if run_dir.exists() and summary_path.exists():
            try:
                with open(summary_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if "final_test_metrics" in data:
                        completed_runs.append(run_id)
            except:
                pass
    return completed_runs


def run_training_with_seed(script_name, seed, run_id, output_subdir):
    """Run training script with given seed and save outputs to subdirectory."""
    print(f"\n{'='*60}", flush=True)
    print(
        f"Running: {script_name} | Run {run_id}/{DEFAULT_NUM_RUNS} | Seed: {seed}",
        flush=True,
    )
    print(f"{'='*60}", flush=True)
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(seed)
    env["RANDOM_SEED"] = str(seed)
    env["TORCH_SEED"] = str(seed)
    wrapper_script_content = f"""#!/usr/bin/env python
# Wrapper: set RNG seeds then exec training script
import random
import numpy as np
import torch
import os
import sys

seed = {seed}
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['RANDOM_SEED'] = str(seed)
os.environ['TORCH_SEED'] = str(seed)
exec(open(r'{script_name}', encoding='utf-8').read())
"""
    wrapper_path = Path(f"_temp_wrapper_{run_id}_{script_name}")
    with open(wrapper_path, "w", encoding="utf-8") as f:
        f.write(wrapper_script_content)

    try:
        start_time = time.time()
        process = subprocess.Popen(
            [sys.executable, str(wrapper_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
            env=env,
        )
        for line in process.stdout:
            print(line, end="")

        process.wait()
        duration = time.time() - start_time
        if process.returncode == 0:
            exp_name = TRAINING_SCRIPTS[script_name]
            exp_dir = OUTPUT_BASE_DIR / exp_name
            run_dir = Path(output_subdir)
            if exp_dir.exists():
                run_dir.mkdir(parents=True, exist_ok=True)
                files_to_copy = [
                    "summary_report.json",
                    "classification_report.txt",
                    "classification_report.json",
                    "training_history.csv",
                    "hyperparameters.json",
                    "vocab.json",
                    "test_predictions.txt",
                    "confusion_matrix.csv",
                ]
                for file_name in files_to_copy:
                    src = exp_dir / file_name
                    if src.exists():
                        dst = run_dir / file_name
                        shutil.copy2(src, dst)
                model_patterns = ["*-best.pt", "*.pt"]
                for pattern in model_patterns:
                    for model_file in exp_dir.glob(pattern):
                        dst = run_dir / model_file.name
                        shutil.copy2(model_file, dst)
                for img_file in exp_dir.glob("*.png"):
                    dst = run_dir / img_file.name
                    shutil.copy2(img_file, dst)

            hours = int(duration // 3600)
            minutes = int((duration % 3600) // 60)
            print(
                f"\n[SUCCESS] Run {run_id} completed in {hours}h {minutes}m", flush=True
            )
            print(f"Results copied to: {run_dir}", flush=True)
            return True
        else:
            print(
                f"\n[FAILED] Run {run_id} exited with code {process.returncode}",
                flush=True,
            )
            return False

    except Exception as e:
        print(f"\n[ERROR] Run {run_id} failed: {str(e)}", flush=True)
        import traceback

        traceback.print_exc()
        return False
    finally:
        if wrapper_path.exists():
            wrapper_path.unlink()


def collect_metrics_from_run(output_dir, run_id):
    """Collect metrics from a single run."""
    run_dir = output_dir / f"run_{run_id}"
    summary_path = run_dir / "summary_report.json"

    if not summary_path.exists():
        return None

    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        metrics = {
            "run_id": run_id,
            "training_time": data.get("training_duration_seconds", 0),
            "inference_time": data.get("evaluation_time_seconds", 0),
            "num_params": data.get("model_info", {}).get("num_trainable_params", 0),
        }
        final_metrics = data.get("final_test_metrics", {})
        if isinstance(final_metrics, dict):
            metrics["weighted_f1"] = final_metrics.get("weighted avg", {}).get(
                "f1-score", 0
            )
            metrics["macro_f1"] = final_metrics.get("macro avg", {}).get("f1-score", 0)
            metrics["micro_f1"] = final_metrics.get("micro avg", {}).get("f1-score", 0)
            metrics["weighted_precision"] = final_metrics.get("weighted avg", {}).get(
                "precision", 0
            )
            metrics["weighted_recall"] = final_metrics.get("weighted avg", {}).get(
                "recall", 0
            )
            for key, value in final_metrics.items():
                if isinstance(value, dict) and "f1-score" in value:
                    metrics[f"{key}_f1"] = value["f1-score"]
                    metrics[f"{key}_precision"] = value.get("precision", 0)
                    metrics[f"{key}_recall"] = value.get("recall", 0)

        return metrics

    except Exception as e:
        print(f"Warning: Could not collect metrics from run_{run_id}: {e}")
        return None


def aggregate_multiple_runs(model_name, script_name, num_runs, seeds):
    """Aggregate results from multiple runs."""
    exp_name = TRAINING_SCRIPTS[script_name]
    model_output_dir = OUTPUT_BASE_DIR / exp_name

    print(f"\n{'='*60}", flush=True)
    print(f"Aggregating results for: {model_name}", flush=True)
    print(f"{'='*60}", flush=True)
    all_runs_data = []
    for run_id in range(1, num_runs + 1):
        metrics = collect_metrics_from_run(model_output_dir, run_id)
        if metrics:
            all_runs_data.append(metrics)

    if not all_runs_data:
        print(f"Warning: No completed runs found for {model_name}", flush=True)
        return None
    df = pd.DataFrame(all_runs_data)

    # Calculate statistics for each metric
    aggregated = {
        "model": model_name,
        "script": script_name,
        "num_runs": len(all_runs_data),
        "completed_runs": [m["run_id"] for m in all_runs_data],
        "metrics": {},
    }
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col == "run_id":
            continue

        values = df[col].values
        valid_values = values[np.isfinite(values)]

        if len(valid_values) > 0:
            mean_val = np.mean(valid_values)
            std_val = np.std(valid_values, ddof=1)
            min_val = np.min(valid_values)
            max_val = np.max(valid_values)
            cv = std_val / mean_val if mean_val != 0 else 0

            if len(valid_values) > 1 and std_val > 0:
                try:
                    sem = stats.sem(valid_values)
                    if np.isfinite(sem) and sem > 0:
                        ci = stats.t.interval(
                            0.95, len(valid_values) - 1, loc=mean_val, scale=sem
                        )
                        ci_lower, ci_upper = ci
                        if not (np.isfinite(ci_lower) and np.isfinite(ci_upper)):
                            ci_lower, ci_upper = mean_val, mean_val
                    else:
                        ci_lower, ci_upper = mean_val, mean_val
                except (ValueError, RuntimeWarning):
                    ci_lower, ci_upper = mean_val, mean_val
            else:
                ci_lower, ci_upper = mean_val, mean_val

            aggregated["metrics"][col] = {
                "mean": float(mean_val),
                "std": float(std_val),
                "min": float(min_val),
                "max": float(max_val),
                "cv": float(cv),
                "ci_95_lower": float(ci_lower),
                "ci_95_upper": float(ci_upper),
                "values": [float(v) for v in valid_values],
            }
    model_summary_path = model_output_dir / "multiple_runs_summary.json"
    with open(model_summary_path, "w", encoding="utf-8") as f:
        json.dump(aggregated, f, indent=2)

    print(f"Saved summary to: {model_summary_path}", flush=True)

    return aggregated, df


def generate_statistics_table(all_aggregated_data):
    """Generate table with mean ± std and confidence intervals."""
    rows = []

    for agg_data in all_aggregated_data:
        if agg_data is None:
            continue

        model_name = agg_data["model"]
        metrics = agg_data["metrics"]

        row = {"Model": model_name}
        key_metrics = [
            "weighted_f1",
            "macro_f1",
            "micro_f1",
            "weighted_precision",
            "weighted_recall",
            "training_time",
            "inference_time",
            "num_params",
        ]

        for metric in key_metrics:
            if metric in metrics:
                m = metrics[metric]
                row[metric] = f"{m['mean']:.4f} ± {m['std']:.4f}"
                row[f"{metric}_ci"] = (
                    f"[{m['ci_95_lower']:.4f}, {m['ci_95_upper']:.4f}]"
                )

        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def escape_latex(text):
    """Escape special LaTeX characters."""
    if not isinstance(text, str):
        text = str(text)

    # Order matters: escape backslash first
    replacements = [
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("^", r"\^{}"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\~{}"),
    ]

    for char, escaped in replacements:
        text = text.replace(char, escaped)

    return text


def generate_latex_table_mean_std(all_aggregated_data):
    """Generate LaTeX table for mean ± std results."""
    lines = []

    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append(
        "\\caption{Model Performance: Mean ± Standard Deviation (Multiple Runs)}"
    )
    lines.append("\\label{tab:multiple_runs_mean_std}")
    lines.append("\\begin{tabular}{lcccc}")
    lines.append("\\toprule")
    lines.append("Model & Weighted F1 & Macro F1 & Precision & Recall \\\\")
    lines.append("\\midrule")

    for agg_data in all_aggregated_data:
        if agg_data is None:
            continue

        model_name = escape_latex(agg_data["model"])
        metrics = agg_data["metrics"]

        weighted_f1 = metrics.get("weighted_f1", {})
        macro_f1 = metrics.get("macro_f1", {})
        precision = metrics.get("weighted_precision", {})
        recall = metrics.get("weighted_recall", {})

        f1_str = (
            f"{weighted_f1.get('mean', 0):.4f} $\\pm$ {weighted_f1.get('std', 0):.4f}"
            if weighted_f1
            else "N/A"
        )
        macro_str = (
            f"{macro_f1.get('mean', 0):.4f} $\\pm$ {macro_f1.get('std', 0):.4f}"
            if macro_f1
            else "N/A"
        )
        prec_str = (
            f"{precision.get('mean', 0):.4f} $\\pm$ {precision.get('std', 0):.4f}"
            if precision
            else "N/A"
        )
        rec_str = (
            f"{recall.get('mean', 0):.4f} $\\pm$ {recall.get('std', 0):.4f}"
            if recall
            else "N/A"
        )

        lines.append(
            f"{model_name} & {f1_str} & {macro_str} & {prec_str} & {rec_str} \\\\"
        )

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def generate_latex_table_per_entity(all_aggregated_data):
    """Generate LaTeX table for per-entity mean ± std results."""
    lines = []
    entity_types = set()
    for agg_data in all_aggregated_data:
        if agg_data is None:
            continue
        metrics = agg_data["metrics"]
        for key in metrics.keys():
            if (
                key.endswith("_f1")
                and key != "weighted_f1"
                and key != "macro_f1"
                and key != "micro_f1"
            ):
                entity_type = key.replace("_f1", "")
                entity_types.add(entity_type)

    if not entity_types:
        return ""

    entity_types = sorted(entity_types)
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append(
        "\\caption{Per-Entity F1 Score: Mean ± Standard Deviation (Multiple Runs)}"
    )
    lines.append("\\label{tab:multiple_runs_per_entity}")
    lines.append("\\begin{tabular}{l" + "c" * len(entity_types) + "}")
    lines.append("\\toprule")
    lines.append(
        "Model & " + " & ".join([escape_latex(et) for et in entity_types]) + " \\\\"
    )
    lines.append("\\midrule")

    for agg_data in all_aggregated_data:
        if agg_data is None:
            continue

        model_name = escape_latex(agg_data["model"])
        metrics = agg_data["metrics"]

        entity_values = []
        for entity_type in entity_types:
            metric_key = f"{entity_type}_f1"
            if metric_key in metrics:
                m = metrics[metric_key]
                entity_values.append(f"{m['mean']:.4f} $\\pm$ {m['std']:.4f}")
            else:
                entity_values.append("N/A")

        lines.append(f"{model_name} & " + " & ".join(entity_values) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def plot_results_with_error_bars(all_aggregated_data):
    """Plot metrics with error bars."""
    if not all_aggregated_data:
        print("No data to plot", flush=True)
        return
    valid_data = [d for d in all_aggregated_data if d is not None]
    if not valid_data:
        return
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "Multiple Runs Analysis: Performance Metrics with Error Bars",
        fontsize=18,
        fontweight="bold",
    )
    models = [d["model"] for d in valid_data]
    weighted_f1_means = [
        d["metrics"].get("weighted_f1", {}).get("mean", 0) for d in valid_data
    ]
    weighted_f1_stds = [
        d["metrics"].get("weighted_f1", {}).get("std", 0) for d in valid_data
    ]
    macro_f1_means = [
        d["metrics"].get("macro_f1", {}).get("mean", 0) for d in valid_data
    ]
    macro_f1_stds = [d["metrics"].get("macro_f1", {}).get("std", 0) for d in valid_data]
    ax1 = axes[0, 0]
    x_pos = np.arange(len(models))
    ax1.bar(
        x_pos,
        weighted_f1_means,
        yerr=weighted_f1_stds,
        capsize=5,
        alpha=0.7,
        color="steelblue",
        edgecolor="black",
    )
    ax1.set_xlabel("Model", fontsize=12)
    ax1.set_ylabel("Weighted F1 Score", fontsize=12)
    ax1.set_title("Weighted F1 Score (Mean ± Std)", fontsize=14, fontweight="bold")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(models, rotation=45, ha="right")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 100])
    ax2 = axes[0, 1]
    ax2.bar(
        x_pos,
        macro_f1_means,
        yerr=macro_f1_stds,
        capsize=5,
        alpha=0.7,
        color="coral",
        edgecolor="black",
    )
    ax2.set_xlabel("Model", fontsize=12)
    ax2.set_ylabel("Macro F1 Score", fontsize=12)
    ax2.set_title("Macro F1 Score (Mean ± Std)", fontsize=14, fontweight="bold")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(models, rotation=45, ha="right")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 100])
    ax3 = axes[1, 0]
    f1_data = []
    f1_labels = []
    for d in valid_data:
        if "weighted_f1" in d["metrics"]:
            values = d["metrics"]["weighted_f1"].get("values", [])
            if values:
                f1_data.append(values)
                f1_labels.append(d["model"])

    if f1_data:
        bp = ax3.boxplot(f1_data, labels=f1_labels, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("lightblue")
            patch.set_alpha(0.7)
        ax3.set_xlabel("Model", fontsize=12)
        ax3.set_ylabel("Weighted F1 Score", fontsize=12)
        ax3.set_title(
            "F1 Score Distribution Across Runs", fontsize=14, fontweight="bold"
        )
        ax3.tick_params(axis="x", rotation=45)
        ax3.grid(True, alpha=0.3)
    ax4 = axes[1, 1]
    std_data = []
    std_labels = []
    for d in valid_data:
        if "weighted_f1" in d["metrics"]:
            std_val = d["metrics"]["weighted_f1"].get("std", 0)
            std_data.append(std_val)
            std_labels.append(d["model"])

    if std_data:
        y_pos = np.arange(len(std_labels))
        colors = plt.cm.RdYlGn_r(
            np.array(std_data) / max(std_data) if max(std_data) > 0 else std_data
        )
        ax4.barh(y_pos, std_data, color=colors, alpha=0.7, edgecolor="black")
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels(std_labels)
        ax4.set_xlabel("Standard Deviation", fontsize=12)
        ax4.set_title(
            "Model Stability (Lower Std = More Stable)", fontsize=14, fontweight="bold"
        )
        ax4.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    plot_path = MULTIPLE_RUNS_DIR / "multiple_runs_plots.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved plots to: {plot_path}", flush=True)


def main():
    """Main function."""
    print("\n" + "=" * 60, flush=True)
    print("Multiple Runs Analysis", flush=True)
    print("=" * 60, flush=True)
    progress = load_checkpoint()
    num_runs = progress.get("num_runs", DEFAULT_NUM_RUNS)
    seeds = progress.get("seeds", DEFAULT_SEEDS[:num_runs])

    print(f"\nConfiguration:", flush=True)
    print(f"  Number of runs: {num_runs}", flush=True)
    print(f"  Seeds: {seeds}", flush=True)
    print(f"  Models: {len(TRAINING_SCRIPTS)}", flush=True)
    print("\nChecking for existing training results...", flush=True)
    has_training = False
    for script_name in TRAINING_SCRIPTS.keys():
        exp_name = TRAINING_SCRIPTS[script_name]
        exp_dir = OUTPUT_BASE_DIR / exp_name
        if exp_dir.exists() and (exp_dir / "summary_report.json").exists():
            has_training = True
            break

    if not has_training:
        print("\nWARNING: No training results found!", flush=True)
        print("Please run training stages first (Stage 1, 2, or 3)", flush=True)
        confirm = input("Continue anyway? (yes/no): ").strip().lower()
        if confirm != "yes":
            return

    all_aggregated_data = []
    all_raw_data = []
    for script_name, exp_name in TRAINING_SCRIPTS.items():
        model_name = get_model_name_from_script(script_name)
        model_output_dir = OUTPUT_BASE_DIR / exp_name
        if script_name not in progress["models"]:
            progress["models"][script_name] = {"runs": {}, "status": "pending"}

        model_progress = progress["models"][script_name]
        completed_runs = check_existing_runs(model_output_dir, num_runs)
        print(f"\n{model_name}: Found {len(completed_runs)} completed runs", flush=True)
        for run_id in range(1, num_runs + 1):
            if run_id in completed_runs:
                print(f"  Skipping run {run_id} (already completed)", flush=True)
                continue

            run_key = f"run_{run_id}"
            if run_key not in model_progress["runs"]:
                model_progress["runs"][run_key] = {
                    "seed": seeds[run_id - 1],
                    "status": "pending",
                }

            run_info = model_progress["runs"][run_key]
            run_dir = model_output_dir / f"run_{run_id}"
            run_dir.mkdir(parents=True, exist_ok=True)
            run_info["status"] = "running"
            run_info["start_time"] = datetime.now().isoformat()
            save_checkpoint(progress)
            success = run_training_with_seed(
                script_name, seeds[run_id - 1], run_id, str(run_dir)
            )

            if success:
                run_info["status"] = "completed"
                run_info["end_time"] = datetime.now().isoformat()
            else:
                run_info["status"] = "failed"
                run_info["end_time"] = datetime.now().isoformat()

            save_checkpoint(progress)
        aggregated, raw_df = aggregate_multiple_runs(
            model_name, script_name, num_runs, seeds
        )
        if aggregated:
            all_aggregated_data.append(aggregated)
            if raw_df is not None:
                raw_df["model"] = model_name
                all_raw_data.append(raw_df)
    if all_aggregated_data:
        print("\n" + "=" * 60, flush=True)
        print("Generating Summary Tables and Visualizations", flush=True)
        print("=" * 60, flush=True)
        if all_raw_data:
            raw_df_all = pd.concat(all_raw_data, ignore_index=True)
            raw_path = MULTIPLE_RUNS_DIR / "all_runs_raw.csv"
            raw_df_all.to_csv(raw_path, index=False)
            print(f"Saved raw data to: {raw_path}", flush=True)
        stats_table = generate_statistics_table(all_aggregated_data)
        stats_path = MULTIPLE_RUNS_DIR / "mean_std_results.csv"
        stats_table.to_csv(stats_path, index=False)
        print(f"Saved statistics table to: {stats_path}", flush=True)
        print("Generating LaTeX tables...", flush=True)
        latex_mean_std = generate_latex_table_mean_std(all_aggregated_data)
        latex_path_mean_std = MULTIPLE_RUNS_DIR / "mean_std_results.tex"
        with open(latex_path_mean_std, "w", encoding="utf-8") as f:
            f.write(latex_mean_std)
        print(f"Saved LaTeX table (mean ± std) to: {latex_path_mean_std}", flush=True)

        latex_per_entity = generate_latex_table_per_entity(all_aggregated_data)
        if latex_per_entity:
            latex_path_per_entity = MULTIPLE_RUNS_DIR / "per_entity_results.tex"
            with open(latex_path_per_entity, "w", encoding="utf-8") as f:
                f.write(latex_per_entity)
            print(
                f"Saved LaTeX table (per-entity) to: {latex_path_per_entity}",
                flush=True,
            )
        plot_results_with_error_bars(all_aggregated_data)
        summary_path = MULTIPLE_RUNS_DIR / "all_runs_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(all_aggregated_data, f, indent=2)
        print(f"Saved aggregated summary to: {summary_path}", flush=True)

        print("\n" + "=" * 60, flush=True)
        print("Multiple Runs Analysis Completed!", flush=True)
        print("=" * 60, flush=True)
        print(f"\nResults saved to: {MULTIPLE_RUNS_DIR}", flush=True)
    else:
        print("\nNo data to aggregate. Please complete at least one run.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Progress saved. You can resume later.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
