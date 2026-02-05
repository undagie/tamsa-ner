# Results Interpretation Guide

## Understanding Output Files

### 1. Classification Reports
Location: `outputs/experiment_*/classification_report.txt`

```
              precision    recall  f1-score   support

         PER     0.9234    0.9156    0.9195      1234
         LOC     0.8876    0.8745    0.8810       987
         ORG     0.8234    0.8456    0.8343       654
        MISC     0.7654    0.7234    0.7438       432

    accuracy                         0.8734      3307
   macro avg     0.8500    0.8398    0.8447      3307
weighted avg     0.8745    0.8734    0.8738      3307
```

Key Metrics:
- **Precision**: Of all entities predicted as X, how many are actually X?
- **Recall**: Of all actual X entities, how many were found?
- **F1-score**: Harmonic mean of precision and recall
- **Support**: Number of instances of each class

### 2. Training History
Location: `outputs/experiment_*/training_history.csv`

Columns:
- `epoch`: Training epoch number
- `train_loss`: Average loss on training data
- `dev_f1`: F1 score on development set
- `dev_loss`: Loss on development set

### 3. Cross-Dataset Results
Location: `outputs/cross_dataset_evaluation/cross_dataset_results.csv`

Interpretation:
- Diagonal values: In-domain performance
- Off-diagonal: Cross-domain generalization
- High variance indicates poor generalization

### 4. Ablation Results
Location: `outputs/ablation_studies/ablation_results.csv`

Shows impact of removing components:
- Larger drop = more important component
- Statistical significance included

### 5. Efficiency Metrics

**Training Efficiency**
- `training_time_hours`: Total training time
- `epochs_to_95_percent`: Convergence speed

**Inference Efficiency**
- `samples_per_second`: Throughput
- `avg_inference_time`: Latency per sample
- `memory_mb`: Peak memory usage

## Comparing Models

### Performance vs Efficiency Trade-off
Good models balance:
1. High F1 score (performance)
2. Low inference time (speed)
3. Reasonable model size (deployment)

### Statistical Significance
McNemar's test results:
- p < 0.05: Significant difference
- p < 0.01: Highly significant

### Choosing Best Model

For **accuracy-critical** applications:
- Use attention fusion model
- Highest F1 scores across datasets

For **real-time** applications:
- Use BiLSTM-W2V-CNN
- Good balance of speed and accuracy

For **resource-constrained** deployment:
- Use basic BiLSTM
- Smallest model size

## Visualization Guide

### Learning Curves
- Smooth curves: stable training
- Early plateau: consider early stopping
- Oscillation: reduce learning rate

### Confusion Matrices
- Diagonal: correct predictions
- Common confusions reveal systematic errors
- Use for error analysis

### Performance Heatmaps
- Red/high values: better performance
- Blue/low values: poor performance
- Patterns show model strengths/weaknesses
