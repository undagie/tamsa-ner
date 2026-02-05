#!/usr/bin/env python
'''
Quick evaluation script for a single model on a dataset.
Usage: python quick_evaluate.py <model_name> <dataset_name>
'''

import sys
import subprocess

def main():
    if len(sys.argv) != 3:
        print("Usage: python quick_evaluate.py <model_name> <dataset_name>")
        print("Models: bilstm, bilstm_w2v, bilstm_w2v_cnn, indobert_bilstm, mbert_bilstm, xlm_roberta_bilstm, attention_fusion")
        print("Datasets: idner2k, nerugm, nerui")
        sys.exit(1)
    
    model = sys.argv[1]
    dataset = sys.argv[2]
    eval_script = f"evaluate_{model}_on_{dataset}.py"
    
    print(f"Evaluating {model} on {dataset}...")
    subprocess.run([sys.executable, f"code/{eval_script}"])

if __name__ == '__main__':
    main()
