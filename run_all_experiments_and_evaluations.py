"""
Run all training, evaluation, and analysis from reproducibility_package root.
Scripts are executed from code/ with current working directory = package root.
"""

import subprocess
import sys
import time
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
CODE_DIR = PACKAGE_ROOT / "code"

TRAINING_SCRIPTS = [
    "train_bilstm.py",
    "train_bilstm_w2v.py",
    "train_bilstm_w2v_cnn.py",
    "train_indobert.py",
    "train_indobert_bilstm.py",
    "train_mbert_bilstm.py",
    "train_xlm_roberta_bilstm.py",
    "train_attention_fusion.py",
]

EVALUATION_SCRIPTS = [
    "evaluate_bilstm_on_nerui.py",
    "evaluate_bilstm_w2v_on_nerui.py",
    "evaluate_bilstm_w2v_cnn_on_nerui.py",
    "evaluate_indobert_only_on_nerui.py",
    "evaluate_indobert_on_nerui.py",
    "evaluate_mbert_on_nerui.py",
    "evaluate_xlm_roberta_on_nerui.py",
    "evaluate_attention_on_nerui.py",
]

ANALYSIS_SCRIPTS = [
    "analysis.py",
    "cross_dataset_evaluation.py",
    "per_entity_analysis.py",
    "ablation_study.py",
    "linguistic_error_analysis.py",
    "efficiency_analysis.py",
    "benchmark_comparison.py",
    "dataset_documentation.py",
    "literature_review_generator.py",
    "visualize_attention.py",
    "generate_paper_figures.py",
]


def check_script_exists(script_name):
    """Check if script exists in code/."""
    return (CODE_DIR / script_name).exists()


def run_script(script_name, script_type="script", index=None, total=None):
    """Run one Python script from code/ with cwd = package root."""
    if index is not None and total is not None:
        print("\n" + "=" * 80)
        print(f"[{index}/{total}] Running {script_type}: {script_name}")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print(f"Running {script_type}: {script_name}")
        print("=" * 80)

    if not check_script_exists(script_name):
        print(f"[ERROR] Script not found: {CODE_DIR / script_name}")
        return False

    script_path = CODE_DIR / script_name
    start_time = time.time()

    try:
        subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(PACKAGE_ROOT),
            check=True,
        )
        duration = time.time() - start_time
        hours = int(duration // 3600)
        minutes = int((duration % 3600) // 60)
        seconds = int(duration % 60)
        duration_str = (
            f"{hours}h {minutes}m {seconds}s"
            if hours
            else (f"{minutes}m {seconds}s" if minutes else f"{seconds}s")
        )
        print(f"\n[SUCCESS] {script_name} completed in {duration_str}")
        return True
    except subprocess.CalledProcessError as e:
        duration = time.time() - start_time
        hours = int(duration // 3600)
        minutes = int((duration % 3600) // 60)
        seconds = int(duration % 60)
        duration_str = (
            f"{hours}h {minutes}m {seconds}s"
            if hours
            else (f"{minutes}m {seconds}s" if minutes else f"{seconds}s")
        )
        print(
            f"\n[FAILED] {script_name} failed with exit code {e.returncode} (duration: {duration_str})"
        )
        return False
    except Exception as e:
        print(f"\n[ERROR] Unexpected error running {script_name}: {e}")
        return False


def run_all_training():
    print("\n" + "#" * 80)
    print("# PHASE 1: TRAINING ALL MODELS")
    print("#" * 80)
    total_scripts = len(TRAINING_SCRIPTS)
    successful = 0
    failed = []
    for i, script in enumerate(TRAINING_SCRIPTS, 1):
        if run_script(script, script_type="TRAINING", index=i, total=total_scripts):
            successful += 1
        else:
            failed.append(script)
            print(f"\n[WARNING] Training {script} failed. Continuing...")
    print("\n" + "-" * 80)
    print(f"Training Summary: {successful}/{total_scripts} successful")
    if failed:
        print(f"Failed: {', '.join(failed)}")
    print("-" * 80)
    return successful, failed


def run_all_evaluations():
    print("\n" + "#" * 80)
    print("# PHASE 2: EVALUATE ALL MODELS ON NER-UI DATASET")
    print("#" * 80)
    total_scripts = len(EVALUATION_SCRIPTS)
    successful = 0
    failed = []
    for i, script in enumerate(EVALUATION_SCRIPTS, 1):
        if run_script(script, script_type="EVALUATION", index=i, total=total_scripts):
            successful += 1
        else:
            failed.append(script)
            print(f"\n[WARNING] Evaluation {script} failed. Continuing...")
    print("\n" + "-" * 80)
    print(f"Evaluation Summary: {successful}/{total_scripts} successful")
    if failed:
        print(f"Failed: {', '.join(failed)}")
    print("-" * 80)
    return successful, failed


def run_all_analysis():
    print("\n" + "#" * 80)
    print("# PHASE 3: COMPREHENSIVE ANALYSIS AND FIGURE GENERATION")
    print("#" * 80)
    total_scripts = len(ANALYSIS_SCRIPTS)
    successful = 0
    failed = []
    for i, script in enumerate(ANALYSIS_SCRIPTS, 1):
        if not check_script_exists(script):
            print(f"[INFO] Skipping {script} (not in package)")
            continue
        if run_script(script, script_type="ANALYSIS", index=i, total=total_scripts):
            successful += 1
        else:
            failed.append(script)
            print(f"\n[WARNING] Analysis {script} failed. Continuing...")
    print("\n" + "-" * 80)
    print(f"Analysis Summary: {successful} completed")
    if failed:
        print(f"Failed: {', '.join(failed)}")
    print("-" * 80)
    return successful, failed


def main():
    overall_start_time = time.time()
    print("\n" + "=" * 80)
    print("  REPRODUCIBILITY PACKAGE: TRAINING AND EVALUATION OF ALL NER MODELS")
    print("  Run from package root; scripts in code/")
    print("=" * 80)
    print(f"\nTotal Training: {len(TRAINING_SCRIPTS)}")
    print(f"Total Evaluation: {len(EVALUATION_SCRIPTS)}")
    print(f"Total Analysis: {len(ANALYSIS_SCRIPTS)}")
    print("\n[INFO] Starting in 3 seconds... (Ctrl+C to cancel)")
    try:
        for i in range(3, 0, -1):
            print(f"      Starting in {i}...", end="\r")
            time.sleep(1)
        print("      Starting now!                    ")
    except KeyboardInterrupt:
        print("\n\n[INFO] Cancelled.")
        return

    training_successful, training_failed = run_all_training()
    evaluation_successful, evaluation_failed = run_all_evaluations()
    analysis_successful, analysis_failed = run_all_analysis()

    overall_duration = time.time() - overall_start_time
    hours = int(overall_duration // 3600)
    minutes = int((overall_duration % 3600) // 60)
    seconds = int(overall_duration % 60)
    time_str = (
        f"{hours}h {minutes}m {seconds}s"
        if hours
        else (f"{minutes}m {seconds}s" if minutes else f"{seconds}s")
    )

    print("\n" + "=" * 80)
    print("  FINAL SUMMARY")
    print("=" * 80)
    print(f"\nTraining: {training_successful}/{len(TRAINING_SCRIPTS)}")
    if training_failed:
        print(f"  Failed: {', '.join(training_failed)}")
    print(f"\nEvaluation: {evaluation_successful}/{len(EVALUATION_SCRIPTS)}")
    if evaluation_failed:
        print(f"  Failed: {', '.join(evaluation_failed)}")
    print(f"\nAnalysis: {analysis_successful} completed")
    if analysis_failed:
        print(f"  Failed: {', '.join(analysis_failed)}")
    print(f"\nTotal time: {time_str}")
    print("\n[RESULT LOCATIONS]")
    print("  - outputs/experiment_*/")
    print("  - outputs/evaluation_nerui_*/")
    if analysis_successful > 0:
        print("  - comparison_results/")
        print("  - outputs/epoch_analysis/, outputs/attention_visualization/, etc.")
    print()


if __name__ == "__main__":
    main()
