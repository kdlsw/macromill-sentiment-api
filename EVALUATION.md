# Model Evaluation Report

## Overview

This document summarizes the evaluation results for all trained models in the macromill sentiment classification project.

## Models Evaluated

| Model | Type | Description |
|-------|------|-------------|
| TF-IDF + Logistic Regression | Traditional ML | TF-IDF vectorization with Logistic Regression |
| TF-IDF + Linear SVM | Traditional ML | Support Vector Machine with TF-IDF features |
| RoBERTa-base | Transformer | Fine-tuned transformer model |

## Evaluation Configuration

### Dataset
- **Source**: IMDB Dataset of 50K Movie Reviews
- **Split**: 80% training / 20% testing (stratified)
- **Preprocessing**: HTML stripping, lowercasing
- **Test Set**: 2,000 samples (held out, never seen during training)

### Metrics

#### Quality Metrics
- **Accuracy**: Overall classification accuracy
- **F1 Score**: Weighted F1 score (harmonic mean of precision and recall)
- **ROC-AUC**: Area under the Receiver Operating Characteristic curve

#### Performance Metrics
- **Latency**: Average inference time per sample (milliseconds)
- **Throughput**: Samples processed per second
- **Artifact Size**: Model file size on disk (MB)

## Results Summary

### Quality Metrics

| Model | Accuracy | F1 Score | ROC-AUC |
|-------|----------|----------|---------|
| TF-IDF + LR | 86.20% | 0.8664 | 0.9379 |
| TF-IDF + Linear SVM | 86.50% | 0.8689 | N/A |
| **RoBERTa-base** | **92.15%** | **0.9215** | **0.9786** |

### Performance Metrics

| Model | Latency (ms/sample) | Throughput (samples/sec) | Artifact Size (MB) |
|-------|---------------------|--------------------------|--------------------|
| TF-IDF + LR | 0.70 | 5,618 | 1.36 |
| TF-IDF + Linear SVM | 0.74 | 5,393 | 1.36 |
| RoBERTa-base | 12.73 | 4,053 | 478.72 |

### Confusion Matrices

**TF-IDF + LR:**
```
              Predicted
             Neg    Pos
Actual Neg    829    163
       Pos    113    895
```

**TF-IDF + Linear SVM:**
```
              Predicted
             Neg    Pos
Actual Neg    835    157
       Pos    113    895
```

**RoBERTa-base:**
```
              Predicted
             Neg    Pos
Actual Neg    904     88
       Pos     69    939
```

## Interpretation

### Key Observations

1. **Accuracy Comparison**: RoBERTa achieves the highest accuracy (92.15%), approximately 6% better than TF-IDF models.

2. **F1 Score**: Same pattern - RoBERTa leads with 0.9215 vs ~0.867 for TF-IDF models.

3. **ROC-AUC**: Available for LR (0.9379) and RoBERTa (0.9786). RoBERTa shows better probability calibration.

4. **Speed vs Accuracy Trade-off**:
   - TF-IDF models are ~18x faster per sample (0.7ms vs 12.7ms)
   - TF-IDF throughput is slightly higher (~5,400-5,600 samples/sec vs ~4,000)
   - RoBERTa is 352x larger on disk (478MB vs 1.36MB)

5. **Confusion Analysis**:
   - RoBERTa has fewer false negatives (69 vs 113) and false positives (88 vs 157)
   - This means RoBERTa is significantly better at identifying both positive and negative sentiments

## Running Evaluation

To reproduce these results:

```bash
# Activate environment
cd /home/ubun/macromill
source .venv/bin/activate

# Run model comparison (generates summary.json and metrics_bar.png)
python work/run_eda.py compare

# Or evaluate individual models
python work/run_cli.py eval \
    --data-path "work/IMDB Dataset.csv" \
    --artifact-path work/artifacts/tfidf_lr_v3/model.joblib \
    --output-json work/artifacts/tfidf_lr_v3/eval.json

python work/run_cli.py eval \
    --data-path "work/IMDB Dataset.csv" \
    --artifact-path work/artifacts/tfidf_linearsvm_v3/model.joblib \
    --output-json work/artifacts/tfidf_linearsvm_v3/eval.json

python work/run_cli.py eval \
    --data-path "work/IMDB Dataset.csv" \
    --artifact-path work/artifacts/roberta_v3/model.safetensors \
    --output-json work/artifacts/roberta_v3/eval.json
```

## Data Split Details

### TF-IDF Models
- **Training**: 36,000 samples (80%)
- **Testing**: 2,000 samples (20%)
- **Split**: Random seed (42) used for train/test split

### RoBERTa
- **Training**: 32,000 samples (72%)
- **Validation**: 4,000 samples (8%) - used during training
- **Testing**: 2,000 samples (20%) - held out, never seen during training

The test set is completely separate from training, ensuring no data leakage in evaluation results.

## Visualization

Generated evaluation files in `work/eda/`:
- `metrics_bar.png` - Bar chart comparing all metrics across models
- `summary.json` - Machine-readable summary of all results
- `model_comparison.txt` - Text-based comparison table

## Notes

- All evaluations were run on the same hardware configuration
- Latency measurements include model loading and preprocessing overhead
- RoBERTa evaluations use GPU acceleration (CUDA)
- TF-IDF models use CPU for inference
