# Study 3: Multimodal Scientific PDF Classification

This directory contains the code to test whether a mid-level fusion of BERT, GPT-2 and ResNet-18 representations outperforms single-modality baselines on scientific PDF classification using extracted text and images.

## Overview

The `Code.py` script evaluates the proposed multimodal architecture, three single-modality ablation baselines, and a five-fold cross-validation robustness check, across multiple random seeds, to validate performance on the NUS Keyphrase Extraction Corpus.

## Key Features

- **Data Handling**: Extracts text from scientific PDFs using PyMuPDF, with PyTesseract OCR fallback for scanned pages. Up to four visual inputs are used per document, prioritising embedded figures and tables and rendering the first pages when insufficient embedded images are available.
- **Proposed Method and Ablations**: Implements BERT + GPT-2 + ResNet-18 multimodal fusion through a text fusion gate and a modality gate, alongside three single-modality ablation baselines (BERT-only, GPT-2-only, ResNet-18-only) for controlled comparison.
- **Metrics**: Records accuracy, weighted F1-score, ROC-AUC, Brier score, and Expected Calibration Error (ECE) for comprehensive classification evaluation.
- **Statistical Testing**: Runs Wilcoxon signed-rank tests across 7 seeds to measure statistical significance of the proposed model over each ablation baseline.
- **Five-Fold Cross-Validation**: Runs a five-fold cross-validation on the proposed multimodal model as a robustness check, reporting per-fold metrics and an aggregated confusion matrix.
- **GPU Support**: Automatic detection and acceleration on T4 GPU.

## Usage

This script is configured for a Kaggle notebook environment with a T4 GPU. It automatically locates the PDF corpus by searching `/kaggle/input` for an `extracted_pdfs/data` directory; adjust `IMG_CACHE` and `OUT_DIR` in the configuration section if you need different output locations.

```python
# Run directly in Kaggle notebook
exec(open("Code.py").read())
```

Or run locally:

```bash
python Code.py
```

## Outputs

The script generates six CSV files in the `thesis_outputs` directory:

- **`study3_cluster_labels.csv`**: The Sentence-BERT + K-Means cluster label assigned to each PDF, used as the classification target.
- **`study3_multimodal_per_seed.csv`**: Raw results for the proposed multimodal model on each of the 7 seeds (accuracy, F1, ROC-AUC, Brier score, ECE).
- **`study3_ablation_summary.csv`**: Mean and standard deviation of metrics across all 7 seeds for the proposed model and each ablation baseline.
- **`study3_wilcoxon.csv`**: Wilcoxon signed-rank test results comparing the proposed multimodal model against each single-modality ablation baseline.
- **`study3_5fold_cv.csv`**: Per-fold accuracy, F1, ROC-AUC, Brier score, and ECE from the five-fold cross-validation of the proposed model.
- **`study3_cv_confusion_matrix.csv`**: Confusion matrix aggregated across all five cross-validation folds.

## Architecture

### Proposed Multimodal Fusion

```
Scientific PDF
    ├── Text Extraction (BERT tokenization)
    │   └── BERT pooler output → 768 dims → Project to fusion_dim
    │
    ├── Text Extraction (GPT-2 tokenization)
    │   └── GPT-2 last real (non-padding) token → 768 dims → Project to fusion_dim
    │
    └── Image Extraction & Processing
        ├── Extract up to 4 embedded images (figures, tables)
        ├── Fallback: render first pages if insufficient embedded images
        ├── ResNet-18 (frozen) → 512 dims per image
        └── Visual attention pooling over up to 4 images → Project to fusion_dim

                    ↓
        Text Fusion Gate: h_text = z_bert + sigmoid(gate) * z_gpt
                    ↓
        Modality Gate: h_fusion = h_text + sigmoid(gate) * z_img
        (gate bias initialised so the visual stream starts closed)
                    ↓
    Classification Head (dropout + linear) → 5 thematic classes
```

### Ablation Baselines

| Model | Input | Representation |
|-------|-------|----------------|
| BERT-only | Text | BERT pooled output → 256-dim projection → classifier |
| GPT-2-only | Text | GPT-2 last non-padding token → 256-dim projection → classifier |
| ResNet-18-only | Images | Frozen ResNet-18 → visual attention pooling → 256-dim projection → classifier |
| **Multimodal** *(proposed)* | Text + images | Gated BERT/GPT-2 text fusion + gated visual representation → classifier |

## Configuration

Key hyperparameters in `Code.py`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `SEEDS` | (42, 7, 123, 999, 2023, 8888, 7777) | 7 random seeds for reproducibility |
| `N_CLUSTERS` | 5 | Number of Sentence-BERT + K-Means thematic clusters (classification classes) |
| `MM_EPOCHS` | 5 | Training epochs per seed |
| `BATCH_SIZE` | 16 | Batch size |
| `MAX_LEN` | 512 | Max token length for text |
| `LR` | 5e-5 | AdamW learning rate |
| `WEIGHT_DECAY` | 0.01 | AdamW weight decay |
| `FUSION_DIM` | 256 | Shared fusion dimension |
| `N_IMAGES` | 4 | Max images to extract per PDF |
| `DROPOUT` | 0.5 | Classification head dropout |

## Requirements

See `../Requirements.txt` for full dependency list. Key packages:
- `torch`, `torchvision`
- `transformers` (BERT, GPT-2)
- `pymupdf` (PDF text and image extraction)
- `pytesseract` (OCR fallback for scanned PDFs)
- `sentence-transformers` (Sentence-BERT thematic label generation)
- `scikit-learn`
- `scipy` (Wilcoxon test)
- `pandas`, `numpy`, `Pillow`

## Citation

Part of PhD research on hybrid multimodal and quantum machine learning architectures.
