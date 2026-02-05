#!/usr/bin/env python
"""Reproduce all experiments: train 8 models, run cross-dataset evaluations, then analyses."""

import subprocess
import sys

def run_command(cmd):
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"Command failed: {cmd}")
        sys.exit(1)

def main():
    print("=== Reproducing Indonesian NER Experiments ===")
    print("\n1. Installing dependencies...")
    run_command("pip install -r requirements.txt")
    print("\n2. Training models...")
    models = [
        'train_bilstm.py',
        'train_bilstm_w2v.py',
        'train_bilstm_w2v_cnn.py',
        'train_indobert.py',           # IndoBERT-CRF
        'train_indobert_bilstm.py',
        'train_mbert_bilstm.py',
        'train_xlm_roberta_bilstm.py',
        'train_attention_fusion.py',   # TAMSA
    ]
    
    for model in models:
        run_command(f"python code/{model}")
    print("\n3. Running cross-dataset evaluations...")
    eval_scripts = [
        'evaluate_bilstm_on_nerui.py',
        'evaluate_bilstm_w2v_on_nerui.py',
        'evaluate_bilstm_w2v_cnn_on_nerui.py',
        'evaluate_indobert_only_on_nerui.py',
        'evaluate_indobert_on_nerui.py',
        'evaluate_mbert_on_nerui.py',
        'evaluate_xlm_roberta_on_nerui.py',
        'evaluate_attention_on_nerui.py',
    ]
    for script in eval_scripts:
        run_command(f"python code/{script}")
    run_command("python code/cross_dataset_evaluation.py")
    print("\n4. Running analyses...")
    analyses = [
        'ablation_study.py',
        'linguistic_error_analysis.py',
        'efficiency_analysis.py',
        'benchmark_comparison.py',
        'dataset_documentation.py'
    ]
    
    for analysis in analyses:
        run_command(f"python code/{analysis}")
    
    print("\n=== All experiments completed! ===")
    print("Check the outputs/ directory for results.")

if __name__ == '__main__':
    main()
