import os
import subprocess
import sys
import time
from pathlib import Path

# ============================================================================
# CONFIGURATION: List of Training and Evaluation Scripts
# ============================================================================

# List of training scripts (ordered from MOST BASIC to MOST COMPLEX)
# Order: Baseline -> Non-Transformer -> Transformer -> Advanced Fusion
TRAINING_SCRIPTS = [
    # 1. Baseline
    "train_bilstm.py",                    # BiLSTM-CRF
    # 2. Non-Transformer: additional embeddings
    "train_bilstm_w2v.py",                # BiLSTM + Word2Vec
    "train_bilstm_w2v_cnn.py",            # BiLSTM + Word2Vec + CharCNN
    # 3. Transformer-based
    "train_indobert_bilstm.py",           # IndoBERT + BiLSTM-CRF
    "train_mbert_bilstm.py",              # mBERT + BiLSTM-CRF
    "train_xlm_roberta_bilstm.py",        # XLM-RoBERTa + BiLSTM-CRF
    # 4. Advanced: attention fusion
    "train_attention_fusion.py",          # TAMSA (MultiSource Attention CRF)
]

# List of evaluation scripts (must match order with training scripts)
EVALUATION_SCRIPTS = [
    "evaluate_bilstm_on_nerui.py",
    "evaluate_bilstm_w2v_on_nerui.py",
    "evaluate_bilstm_w2v_cnn_on_nerui.py",
    "evaluate_indobert_on_nerui.py",
    "evaluate_mbert_on_nerui.py",
    "evaluate_xlm_roberta_on_nerui.py",
    "evaluate_attention_on_nerui.py",
]

# Analysis script (optional, will be run at the end if available)
ANALYSIS_SCRIPT = "analysis.py"

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def check_script_exists(script_name):
    """Check if script exists in current directory."""
    script_path = Path(script_name)
    return script_path.exists()

def run_script(script_name, script_type="script", index=None, total=None):
    """
    Run one Python script and handle output/error.
    Output is displayed in real-time for progress monitoring.
    
    Args:
        script_name: Name of script file to run
        script_type: Type of script ("training" or "evaluation")
        index: Script sequence number (for progress tracking)
        total: Total number of scripts (for progress tracking)
    """
    # Header with progress information
    if index is not None and total is not None:
        print("\n" + "=" * 80)
        print(f"[{index}/{total}] Running {script_type}: {script_name}")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print(f"Running {script_type}: {script_name}")
        print("=" * 80)
    
    # Check if file exists
    if not check_script_exists(script_name):
        print(f"[ERROR] Script not found: {script_name}")
        return False
    
    # Record start time
    start_time = time.time()
    
    try:
        # Run script with real-time output (not captured)
        # This allows user to see training progress directly
        process = subprocess.run(
            [sys.executable, script_name],
            check=True
            # Not using capture_output=True so output is immediately visible
        )
        
        # Calculate duration
        duration = time.time() - start_time
        hours = int(duration // 3600)
        minutes = int((duration % 3600) // 60)
        seconds = int(duration % 60)
        
        if hours > 0:
            duration_str = f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            duration_str = f"{minutes}m {seconds}s"
        else:
            duration_str = f"{seconds}s"
        
        print(f"\n[SUCCESS] {script_name} completed in {duration_str}")
        return True
        
    except subprocess.CalledProcessError as e:
        duration = time.time() - start_time
        hours = int(duration // 3600)
        minutes = int((duration % 3600) // 60)
        seconds = int(duration % 60)
        
        if hours > 0:
            duration_str = f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            duration_str = f"{minutes}m {seconds}s"
        else:
            duration_str = f"{seconds}s"
        
        print(f"\n[FAILED] {script_name} failed with exit code {e.returncode} (duration: {duration_str})")
        return False
        
    except Exception as e:
        duration = time.time() - start_time
        print(f"\n[ERROR] Unexpected error occurred while running {script_name}: {e}")
        print(f"(duration: {duration:.2f} seconds)")
        return False

# ============================================================================
# MAIN FUNCTIONS
# ============================================================================

def run_all_training():
    """Run all training scripts automatically."""
    print("\n" + "#" * 80)
    print("# PHASE 1: TRAINING ALL MODELS")
    print("# Order: Baseline -> Non-Transformer -> Transformer -> Advanced")
    print("#" * 80)
    
    total_scripts = len(TRAINING_SCRIPTS)
    successful = 0
    failed = []
    
    for i, script in enumerate(TRAINING_SCRIPTS, 1):
        success = run_script(script, script_type="TRAINING", index=i, total=total_scripts)
        if success:
            successful += 1
        else:
            failed.append(script)
            # Automatically continue to next training even if one fails
            print(f"\n[WARNING] Training {script} failed. Continuing to next training...")
    
    print("\n" + "-" * 80)
    print(f"Training Summary: {successful}/{total_scripts} successful")
    if failed:
        print(f"Failed scripts: {', '.join(failed)}")
        print("[INFO] Evaluation will still be run for successfully trained models.")
    print("-" * 80)
    
    return successful, failed

def run_all_evaluations():
    """Run all evaluation scripts automatically."""
    print("\n" + "#" * 80)
    print("# PHASE 2: EVALUATE ALL MODELS ON NER-UI DATASET")
    print("# Evaluation is performed for all trained models")
    print("#" * 80)
    
    total_scripts = len(EVALUATION_SCRIPTS)
    successful = 0
    failed = []
    
    for i, script in enumerate(EVALUATION_SCRIPTS, 1):
        success = run_script(script, script_type="EVALUATION", index=i, total=total_scripts)
        if success:
            successful += 1
        else:
            failed.append(script)
            # Automatically continue to next evaluation even if one fails
            print(f"\n[WARNING] Evaluation {script} failed. Continuing to next evaluation...")
    
    print("\n" + "-" * 80)
    print(f"Evaluation Summary: {successful}/{total_scripts} successful")
    if failed:
        print(f"Failed scripts: {', '.join(failed)}")
    print("-" * 80)
    
    return successful, failed

def run_analysis():
    """Run analysis script if available."""
    print("\n" + "#" * 80)
    print("# PHASE 3: EXPERIMENT RESULTS ANALYSIS")
    print("#" * 80)
    
    if not check_script_exists(ANALYSIS_SCRIPT):
        print(f"[INFO] Analysis script '{ANALYSIS_SCRIPT}' not found. Skipped.")
        return False
    
    success = run_script(ANALYSIS_SCRIPT, script_type="ANALYSIS")
    
    if success:
        print("\n[SUCCESS] Analysis completed. Results saved in 'comparison_results' directory")
    else:
        print("\n[WARNING] Analysis failed, but evaluation results are still saved.")
    
    return success

def main():
    """Main function to run all training and evaluation automatically."""
    # Record overall start time
    overall_start_time = time.time()
    
    print("\n" + "=" * 80)
    print("=" * 80)
    print("  AUTOMATED SYSTEM: TRAINING AND EVALUATION OF ALL NER MODELS")
    print("  Process will run fully automatically from start to finish")
    print("=" * 80)
    print("=" * 80)
    print(f"\nTotal Training Scripts: {len(TRAINING_SCRIPTS)}")
    print(f"Total Evaluation Scripts: {len(EVALUATION_SCRIPTS)}")
    print(f"Analysis Script: {ANALYSIS_SCRIPT}")
    print("\nExperiment Order:")
    print("  1. Baseline: BiLSTM-CRF")
    print("  2. Non-Transformer: +Word2Vec, +CharCNN")
    print("  3. Transformer: IndoBERT, mBERT, XLM-RoBERTa")
    print("  4. Advanced: TAMSA (MultiSource Attention CRF)")
    print("\n" + "=" * 80)
    print("[INFO] Starting automated process in 3 seconds...")
    print("      (Press Ctrl+C to cancel)")
    print("=" * 80)
    
    # Short countdown before starting
    try:
        for i in range(3, 0, -1):
            print(f"      Starting in {i}...", end='\r')
            time.sleep(1)
        print("      Starting now!                    ")
    except KeyboardInterrupt:
        print("\n\n[INFO] Process cancelled by user.")
        return
    
    # Phase 1: Training (from most basic to most complex)
    training_successful, training_failed = run_all_training()
    
    # Phase 2: Evaluation (for all successfully trained models)
    evaluation_successful, evaluation_failed = run_all_evaluations()
    
    # Phase 3: Analysis (optional, to compare all results)
    analysis_success = run_analysis()
    
    # Final summary
    overall_duration = time.time() - overall_start_time
    hours = int(overall_duration // 3600)
    minutes = int((overall_duration % 3600) // 60)
    seconds = int(overall_duration % 60)
    
    print("\n" + "=" * 80)
    print("=" * 80)
    print("  FINAL SUMMARY - ALL PROCESSES COMPLETED")
    print("=" * 80)
    print("=" * 80)
    print(f"\nTraining: {training_successful}/{len(TRAINING_SCRIPTS)} successful")
    if training_failed:
        print(f"  - Failed: {', '.join(training_failed)}")
    
    print(f"\nEvaluation: {evaluation_successful}/{len(EVALUATION_SCRIPTS)} successful")
    if evaluation_failed:
        print(f"  - Failed: {', '.join(evaluation_failed)}")
    
    if analysis_success:
        print(f"\nAnalysis: Successful")
    else:
        print(f"\nAnalysis: Failed or skipped")
    
    if hours > 0:
        time_str = f"{hours} hours {minutes} minutes {seconds} seconds"
    elif minutes > 0:
        time_str = f"{minutes} minutes {seconds} seconds"
    else:
        time_str = f"{seconds} seconds"
    
    print(f"\nTotal Overall Time: {time_str}")
    print("\n" + "=" * 80)
    print("=" * 80)
    
    # Result location information
    print("\n[RESULT LOCATIONS]")
    print("  - Models and training results: ./outputs/experiment_*/")
    print("  - Evaluation results: ./outputs/evaluation_nerui_*/")
    if analysis_success:
        print("  - Comparison analysis: ./comparison_results/")
        print("  - Analysis report: ./comparison_results/full_analysis_report.md")
    print("\n[INFO] All processes completed!")
    print("\n")

if __name__ == "__main__":
    main()

