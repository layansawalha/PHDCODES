# Study 2: Multimodal Breast Lesion Ultrasound Classification

This directory contains the code to test whether a mid-level fusion of BERT, GPT-2 and ResNet-18 representations outperforms single-modality baselines on binary malignancy classification using the BrEaST-Lesions USG dataset.

## Overview

The `Code.py` script evaluates four model architectures (BERT-only, GPT2-only, ResNet18-only, and the proposed Multimodal fusion) across multiple random seeds, reporting calibration-aware metrics and statistical significance testing, with a five-fold cross-validation robustness check on the proposed model.

## Key Features

- **Data Handling**: Loads clinical/BI-RADS descriptor text (symptoms, which include family history, breast composition, BI-RADS category, lesion shape/margin/echogenicity/posterior features, calcifications) and paired ultrasound images from the BrEaST-Lesions USG Excel sheet and image directory. The "Classification", "Diagnosis", "Interpretation" and "Verification" fields are deliberately excluded from the text input to avoid target leakage.
- **Patient-Level Splitting**: Splits are stratified over unique patients, not rows, so no patient's records appear in both the training and test partitions of any split.
- **Multiple Baselines**: Implements BERT-only, GPT2-only, and ResNet18-only single-modality baselines alongside the proposed multimodal model, enabling controlled ablation.
- **Metrics**: Records accuracy, precision, recall, F1-score, ROC-AUC, Brier score, and Expected Calibration Error (ECE) for comprehensive and calibration-aware evaluation.
- **Checkpointing**: Automatically saves results to a CSV after every `(seed, model)` combination so interrupted Kaggle sessions can resume seamlessly.
- **Statistical Testing**: Runs Wilcoxon signed-rank tests across 7 seeds to measure statistical significance of the proposed model over each baseline.
- **Five-Fold Cross-Validation**: Runs a patient-stratified five-fold cross-validation on the proposed Multimodal model, reporting per-fold accuracy, validation loss, false positives/negatives, an aggregated confusion matrix, and sensitivity/specificity/PPV/NPV.
- **GPU Support**: Automatic detection and acceleration on T4 GPU.

## Usage

This script is configured for a Kaggle notebook environment with a T4 GPU. Update the `EXCEL_PATH` and `IMG_DIR` variables in the configuration section to point to the correct Kaggle input directories.

```python
# Run directly in Kaggle notebook
exec(open("Code.py").read())
```

Or run locally:

```bash
python Code.py
```

## Outputs

The script generates six CSV files in the `/kaggle/working/` directory:

- **`study2_patient_level_no_diagnosis_results.csv`**: Raw results for each `(seed, model)` combination with accuracy, precision, recall, F1-score, ROC-AUC, Brier score, and ECE.
- **`study2_summary.csv`**: Mean and standard deviation of accuracy, ROC-AUC, Brier score, and ECE across all 7 seeds, sorted by accuracy.
- **`study2_wilcoxon.csv`**: Wilcoxon signed-rank test results comparing the proposed Multimodal model against each baseline (`alternative='greater'`).
- **`table5_5_cv_results.csv`**: Per-fold accuracy, validation loss, false positives and false negatives from the five-fold cross-validation of the Multimodal model.
- **`figure5_10_confusion_matrix.csv`**: Confusion matrix aggregated across all five cross-validation folds.
- **`section5_7_clinical_metrics.csv`**: Sensitivity, specificity, PPV and NPV computed from the aggregated cross-validation confusion matrix.

## Architecture

### Proposed Multimodal Fusion

```
BrEaST-Lesions USG Record
    ├── Clinical/BI-RADS Text (BERT tokenization)
    │   └── BERT pooler output → 768 dims → Project to 128 dims
    │
    ├── Clinical/BI-RADS Text (GPT-2 tokenization)
    │   └── GPT-2 last real (non-padding) token → 768 dims → Project to 128 dims
    │
    └── Ultrasound Image
        └── ResNet-18 (ImageNet weights) → 512 dims → Project to 128 dims

                    ↓
        Mid-level Concatenation Fusion: 384 dims
                    ↓
    Classification Head (dropout + linear) → 2 classes (benign / malignant)
```

### Baselines

| Model | Input | Fusion |
|-------|-------|--------|
| BERT-only | Clinical/BI-RADS text | BERT pooler output → classifier |
| GPT2-only | Clinical/BI-RADS text | GPT-2 last real (non-padding) token hidden state → classifier |
| ResNet18-only | Ultrasound image | ResNet-18 (ImageNet weights) → 512 dims → classifier |
| **Multimodal** *(proposed)* | Text + image | Projected BERT + GPT-2 + ResNet-18 concatenated (384 dims) → classifier |

## Configuration

Key hyperparameters in `Code.py`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `SEEDS` | (42, 7, 123, 999, 2023, 8888, 7777) | 7 random seeds for reproducibility |
| `EPOCHS` | 5 | Training epochs per seed |
| `BATCH_SIZE` | 16 | Batch size |
| `MAX_LENGTH` | 128 | Max token length for text |
| `LR` | 2e-5 | AdamW learning rate |
| `WEIGHT_DECAY` | 0.01 | AdamW weight decay |
| `PATIENT_ID_COLUMN` | Patient_ID | Column identifying each patient for patient-level splitting |
| `CV_FOLDS` | 5 | Number of folds for the cross-validation robustness check |
| `CV_SEED` | 42 | Base seed for the cross-validation folds |

## Requirements

See `../Requirements.txt` for full dependency list. Key packages:
- `torch`, `torchvision`
- `transformers` (BERT, GPT-2)
- `openpyxl` (Excel data loading)
- `scikit-learn`
- `scipy` (Wilcoxon test)
- `pandas`, `numpy`, `Pillow`

## Citation

Part of PhD research on hybrid multimodal and quantum machine learning architectures.
