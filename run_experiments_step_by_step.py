"""
Interactive step-by-step experiment runner for reproducibility_package.
Run from package root; scripts in code/. Missing scripts are skipped.
"""
import sys
import time
import subprocess
import json
from datetime import datetime
from pathlib import Path

# When run from reproducibility_package root
PACKAGE_ROOT = Path(__file__).resolve().parent
CODE_DIR = PACKAGE_ROOT / "code"

CHECKPOINT_FILE = PACKAGE_ROOT / "experiment_progress.json"
LOG_FILE = PACKAGE_ROOT / "experiment_log.txt"

TRAINING_STAGES = [
    "stage_1_baseline",
    "stage_2_transformer",
    "stage_3_advanced",
]

SKIP_IN_FULL_PIPELINE = [
    "stage_6_epoch_analysis",
    "stage_7_multiple_runs",
    "stage_8_statistical_tests",
]

EXPERIMENT_STAGES = {
    "stage_1_baseline": {
        "name": "Baseline Models",
        "scripts": [
            ("train_bilstm.py", "BiLSTM-CRF Baseline"),
            ("train_bilstm_w2v.py", "BiLSTM + Word2Vec"),
            ("train_bilstm_w2v_cnn.py", "BiLSTM + Word2Vec + CharCNN"),
        ],
    },
    "stage_2_transformer": {
        "name": "Transformer Models",
        "scripts": [
            ("train_indobert.py", "IndoBERT-CRF"),
            ("train_indobert_bilstm.py", "IndoBERT + BiLSTM-CRF"),
            ("train_mbert_bilstm.py", "mBERT + BiLSTM-CRF"),
            ("train_xlm_roberta_bilstm.py", "XLM-RoBERTa + BiLSTM-CRF"),
        ],
    },
    "stage_3_advanced": {
        "name": "Advanced Model",
        "scripts": [("train_attention_fusion.py", "TAMSA (MultiSource Attention CRF)")],
    },
    "stage_4_evaluation": {
        "name": "Cross-Dataset Evaluation",
        "scripts": [
            ("evaluate_bilstm_on_nerui.py", "Evaluate BiLSTM on NER-UI"),
            ("evaluate_bilstm_w2v_on_nerui.py", "Evaluate BiLSTM-W2V on NER-UI"),
            ("evaluate_bilstm_w2v_cnn_on_nerui.py", "Evaluate BiLSTM-W2V-CNN on NER-UI"),
            ("evaluate_indobert_only_on_nerui.py", "Evaluate IndoBERT-CRF on NER-UI"),
            ("evaluate_indobert_on_nerui.py", "Evaluate IndoBERT+BiLSTM on NER-UI"),
            ("evaluate_mbert_on_nerui.py", "Evaluate mBERT on NER-UI"),
            ("evaluate_xlm_roberta_on_nerui.py", "Evaluate XLM-R on NER-UI"),
            ("evaluate_attention_on_nerui.py", "Evaluate Attention on NER-UI"),
        ],
    },
    "stage_5_analysis": {
        "name": "Comprehensive Analysis",
        "scripts": [
            ("analysis.py", "Basic Comparison Analysis"),
            ("cross_dataset_evaluation.py", "Cross-Dataset Evaluation"),
            ("per_entity_analysis.py", "Per-Entity Type Analysis"),
            ("ablation_study.py", "Ablation Study"),
            ("linguistic_error_analysis.py", "Linguistic Error Analysis"),
            ("efficiency_analysis.py", "Efficiency Analysis"),
            ("benchmark_comparison.py", "Benchmark Comparison"),
            ("dataset_documentation.py", "Dataset Documentation"),
            ("literature_review_generator.py", "Literature Review"),
            ("visualize_attention.py", "Visualize Attention Weights"),
        ],
    },
    "stage_6_epoch_analysis": {
        "name": "Epoch Analysis",
        "scripts": [
            ("epoch_analysis.py", "Analyze Epoch Configuration and Training Patterns")
        ],
    },
    "stage_7_multiple_runs": {
        "name": "Multiple Runs Analysis",
        "scripts": [
            ("multiple_runs_analysis.py", "Run Multiple Experiments with Different Seeds")
        ],
    },
    "stage_8_statistical_tests": {
        "name": "Statistical Tests",
        "scripts": [
            ("statistical_tests_comprehensive.py", "Comprehensive Statistical Analysis")
        ],
    },
    "stage_9_figures": {
        "name": "Final Figure Generation",
        "scripts": [("generate_paper_figures.py", "Generate All Figures for Paper")],
    },
}


class ExperimentRunner:
    def __init__(self):
        self.progress = self.load_progress()
        self.start_time = time.time()

    def load_progress(self):
        if CHECKPOINT_FILE.exists():
            with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"completed": [], "current_stage": None, "stage_progress": {}}

    def save_progress(self):
        with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
            json.dump(self.progress, f, indent=2)

    def log(self, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(log_message + "\n")

    def has_training_completed(self):
        return any(
            len(self.progress["stage_progress"].get(ts, [])) > 0
            for ts in TRAINING_STAGES
        )

    def has_multiple_runs_completed(self):
        multiple_runs_dir = PACKAGE_ROOT / "outputs" / "multiple_runs"
        return (multiple_runs_dir / "all_runs_summary.json").exists()

    def format_duration(self, seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"

    def _script_path(self, script_name):
        return CODE_DIR / script_name

    def run_script(self, script_name, description):
        script_path = self._script_path(script_name)
        if not script_path.exists():
            self.log(f"SKIP (not in package): {script_name}")
            return True

        self.log(f"Starting: {description}")
        start_time = time.time()
        process = None

        try:
            process = subprocess.Popen(
                [sys.executable, str(script_path)],
                cwd=str(PACKAGE_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
            )

            for line in process.stdout:
                print(line, end="")

            process.wait()

            if process.returncode == 0:
                duration = time.time() - start_time
                self.log(
                    f"SUCCESS: {description} completed in {self.format_duration(duration)}"
                )
                return True
            else:
                self.log(f"FAILED: {description} exited with code {process.returncode}")
                return False

        except KeyboardInterrupt:
            self.log(f"INTERRUPTED: {description} cancelled by user")
            if process:
                process.terminate()
            raise
        except Exception as e:
            self.log(f"ERROR: {description} - {str(e)}")
            if process:
                process.terminate()
            return False

    def run_stage(self, stage_key, stage_info):
        self.log(f"\n{'='*60}")
        self.log(f"STAGE: {stage_info['name']}")
        self.log(f"{'='*60}")

        if stage_key not in self.progress["stage_progress"]:
            self.progress["stage_progress"][stage_key] = []

        completed_in_stage = set(self.progress["stage_progress"][stage_key])

        if stage_key == "stage_6_epoch_analysis":
            if not self.has_training_completed():
                self.log("WARNING: Epoch analysis requires at least one completed training!")
                confirm = input("Continue anyway? (yes/no): ").strip().lower()
                if confirm != "yes":
                    return False

        elif stage_key == "stage_7_multiple_runs":
            if not self.has_training_completed():
                self.log("WARNING: Multiple runs analysis requires completed training!")
                confirm = input("Continue anyway? (yes/no): ").strip().lower()
                if confirm != "yes":
                    return False

        elif stage_key == "stage_8_statistical_tests":
            if not self.has_multiple_runs_completed():
                self.log("WARNING: Statistical tests require multiple runs analysis!")
                confirm = input("Continue anyway? (yes/no): ").strip().lower()
                if confirm != "yes":
                    return False

        elif stage_key == "stage_5_analysis":
            evaluation_stage = "stage_4_evaluation"
            has_evaluation = (
                len(self.progress["stage_progress"].get(evaluation_stage, [])) > 0
            )
            if not has_evaluation:
                self.log("WARNING: Comprehensive analysis requires completed evaluation!")
                confirm = input("Continue anyway? (yes/no): ").strip().lower()
                if confirm != "yes":
                    return False

        for script_name, description in stage_info["scripts"]:
            if script_name in completed_in_stage:
                self.log(f"NOTE: {description} (already completed)")
                confirm = input(f"Run again? (yes/no, default=no): ").strip().lower()
                if confirm != "yes":
                    self.log(f"SKIPPING: {description}")
                    continue

            success = self.run_script(script_name, description)

            if success:
                if script_name not in self.progress["completed"]:
                    self.progress["completed"].append(script_name)
                if script_name not in self.progress["stage_progress"][stage_key]:
                    self.progress["stage_progress"][stage_key].append(script_name)
                completed_in_stage.add(script_name)
                self.save_progress()
            else:
                self.log(f"Stage {stage_key} halted due to error")
                return False

        return True

    def show_menu(self):
        print("\n" + "=" * 60)
        print("  Indonesian NER Experiment Runner")
        print("=" * 60)
        print("\n  PIPELINE")
        print("  " + "-" * 8)
        print("  1. Main pipeline")
        print("  2. Run specific stage")
        print("\n  ADVANCED ANALYSIS (run after training)")
        print("  " + "-" * 17)
        print("  3. Epoch analysis")
        print("  4. Multiple runs (5 seeds)")
        print("  5. Statistical tests")
        print("\n  UTILITIES")
        print("  " + "-" * 9)
        print("  6. Show progress")
        print("  7. Reset progress")
        print("  8. Exit")

        choice = input(f"\n  Select (1-8): ").strip()
        return choice

    def _get_stage_dependency_note(self, stage_key):
        """Return dependency note for a stage, or empty string if none."""
        if stage_key in ("stage_6_epoch_analysis", "stage_7_multiple_runs"):
            if not self.has_training_completed():
                return " [requires training]"
        elif stage_key == "stage_8_statistical_tests":
            if not self.has_multiple_runs_completed():
                return " [requires multiple runs]"
        elif stage_key == "stage_5_analysis":
            has_eval = len(self.progress["stage_progress"].get("stage_4_evaluation", [])) > 0
            if not has_eval:
                return " [requires evaluation]"
        return ""

    def show_stage_menu(self):
        print("\n" + "=" * 60)
        print("  Run Specific Stage")
        print("=" * 60)

        print("\n  TRAINING")
        print("  " + "-" * 8)
        for i in range(1, 4):
            key = list(EXPERIMENT_STAGES.keys())[i - 1]
            info = EXPERIMENT_STAGES[key]
            done = len(self.progress["stage_progress"].get(key, []))
            total = len(info["scripts"])
            dep = self._get_stage_dependency_note(key)
            print(f"  {i}. {info['name']} ({done}/{total}){dep}")

        print("\n  EVALUATION")
        print("  " + "-" * 10)
        key = "stage_4_evaluation"
        info = EXPERIMENT_STAGES[key]
        done = len(self.progress["stage_progress"].get(key, []))
        total = len(info["scripts"])
        dep = self._get_stage_dependency_note(key)
        print(f"  4. {info['name']} ({done}/{total}){dep}")

        print("\n  ANALYSIS")
        print("  " + "-" * 8)
        for i in range(5, 9):
            key = list(EXPERIMENT_STAGES.keys())[i - 1]
            info = EXPERIMENT_STAGES[key]
            done = len(self.progress["stage_progress"].get(key, []))
            total = len(info["scripts"])
            dep = self._get_stage_dependency_note(key)
            print(f"  {i}. {info['name']} ({done}/{total}){dep}")

        print("\n  OUTPUT")
        print("  " + "-" * 6)
        key = "stage_9_figures"
        info = EXPERIMENT_STAGES[key]
        done = len(self.progress["stage_progress"].get(key, []))
        total = len(info["scripts"])
        dep = self._get_stage_dependency_note(key)
        print(f"  9. {info['name']} ({done}/{total}){dep}")

        print("\n  0. Back to main menu")

        choice = input(f"\n  Select (0-9): ").strip()
        return choice

    def show_progress(self):
        print("\n" + "=" * 60)
        print("Current Progress")
        print("=" * 60)

        total_scripts = sum(
            len(stage["scripts"]) for stage in EXPERIMENT_STAGES.values()
        )
        completed = len(self.progress["completed"])
        pct = (completed / total_scripts * 100) if total_scripts else 0
        print(f"Overall: {completed}/{total_scripts} scripts completed ({pct:.1f}%)")

        for stage_key, stage_info in EXPERIMENT_STAGES.items():
            completed_in_stage = len(self.progress["stage_progress"].get(stage_key, []))
            total_in_stage = len(stage_info["scripts"])
            status_note = ""
            if stage_key == "stage_6_epoch_analysis" or stage_key == "stage_7_multiple_runs":
                if not self.has_training_completed():
                    status_note = " (requires training)"
            elif stage_key == "stage_8_statistical_tests":
                if not self.has_multiple_runs_completed():
                    status_note = " (requires multiple runs)"
            elif stage_key == "stage_5_analysis":
                has_eval = len(self.progress["stage_progress"].get("stage_4_evaluation", [])) > 0
                if not has_eval:
                    status_note = " (requires evaluation)"

            print(f"\n{stage_info['name']}{status_note}:")
            print(f"  Progress: {completed_in_stage}/{total_in_stage}")
            for script, desc in stage_info["scripts"]:
                status = (
                    "[OK]"
                    if script in self.progress["stage_progress"].get(stage_key, [])
                    else "[ ]"
                )
                print(f"    {status} {desc}")

    def reset_progress(self):
        confirm = (
            input("Are you sure you want to reset all progress? (yes/no): ")
            .strip()
            .lower()
        )
        if confirm == "yes":
            self.progress = {
                "completed": [],
                "current_stage": None,
                "stage_progress": {},
            }
            self.save_progress()
            if LOG_FILE.exists():
                LOG_FILE.unlink()
            print("Progress reset successfully!")

    def run_all(self):
        self.log("Starting complete experiment pipeline")

        for stage_key, stage_info in EXPERIMENT_STAGES.items():
            if stage_key in SKIP_IN_FULL_PIPELINE:
                self.log(
                    f"Skipping {stage_info['name']} (run separately after training)"
                )
                continue

            if not self.run_stage(stage_key, stage_info):
                self.log("Pipeline halted due to error")
                return False

        self.log("All experiments completed successfully!")
        return True

    def run(self):
        while True:
            choice = self.show_menu()

            if choice == "1":
                self.run_all()
            elif choice == "2":
                stage_choice = self.show_stage_menu()
                if stage_choice.isdigit() and 1 <= int(stage_choice) <= len(EXPERIMENT_STAGES):
                    stage_key = list(EXPERIMENT_STAGES.keys())[int(stage_choice) - 1]
                    self.run_stage(stage_key, EXPERIMENT_STAGES[stage_key])
            elif choice == "3":
                self.log("Running epoch analysis...")
                if self._script_path("epoch_analysis.py").exists():
                    success = self.run_script("epoch_analysis.py", "Epoch Analysis")
                    if success:
                        self.log("Epoch analysis completed!")
                        print("\nCheck outputs/epoch_analysis/ for results")
                else:
                    print("Error: epoch_analysis.py not in package.")
            elif choice == "4":
                self.log("Running multiple runs analysis...")
                if not self.has_training_completed():
                    confirm = input("Continue anyway? (yes/no): ").strip().lower()
                    if confirm != "yes":
                        continue
                if self._script_path("multiple_runs_analysis.py").exists():
                    success = self.run_script(
                        "multiple_runs_analysis.py", "Multiple Runs Analysis"
                    )
                    if success:
                        self.log("Multiple runs analysis completed!")
                        print("\nCheck outputs/multiple_runs/ for results")
                else:
                    print("Error: multiple_runs_analysis.py not in package.")
            elif choice == "5":
                self.log("Running statistical tests...")
                if not self.has_multiple_runs_completed():
                    confirm = input("Continue anyway? (yes/no): ").strip().lower()
                    if confirm != "yes":
                        continue
                if self._script_path("statistical_tests_comprehensive.py").exists():
                    success = self.run_script(
                        "statistical_tests_comprehensive.py", "Statistical Tests"
                    )
                    if success:
                        self.log("Statistical tests completed!")
                        print("\nCheck outputs/statistical_tests/ for results")
                else:
                    print("Error: statistical_tests_comprehensive.py not in package.")
            elif choice == "6":
                self.show_progress()
            elif choice == "7":
                self.reset_progress()
            elif choice == "8":
                print("\nExiting...")
                break
            else:
                print("Invalid option!")

        total_duration = time.time() - self.start_time
        self.log(f"Total session time: {self.format_duration(total_duration)}")


def main():
    runner = ExperimentRunner()

    try:
        runner.run()
    except KeyboardInterrupt:
        print("\n\nExperiment interrupted by user!")
        runner.save_progress()
        print("Progress saved. You can resume later.")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        runner.save_progress()


if __name__ == "__main__":
    main()
