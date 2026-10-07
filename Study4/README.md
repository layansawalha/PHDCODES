# Study 4: Hybrid Architectures for UK Construction Cost Prediction

This directory contains code to evaluate multiple machine learning, LLM, quantum-classical, and ensemble regression approaches for predicting construction costs on the BCIS dataset.

## Overview

The two scripts in this directory cover different parts of Study 4. `augmented_dataset.py` implements a single proposed model (the Hybrid MLP + XGBoost + CatBoost neural meta-learner) on the augmented dataset variant. `original_dataset.py` is a sequential collection of Study 4 experiments covering classical regressors, transformer-based hybrids, LLM fine-tuning, and the quantum-classical configuration across the BCIS dataset variants.

## Key Features

### `augmented_dataset.py`
- **Train-Test Splitting**: Splits by `record_id`, requiring `is_augmented` and `source_record_id` columns so augmented rows derived from test-set records are detected and rejected rather than silently included in training.
- **Base Learners**: XGBoost and CatBoost are trained separately on the unscaled training partition. Their predictions are concatenated with the MLP latent representation and passed to the neural synthesiser.
- **Leakage-Sensitive Columns Excluded**: `contract_contract_sum` and `cost_increment` are removed before training.
- **Reproducibility**: Seeds `random`, `numpy`, and TensorFlow, and enables TensorFlow op determinism.
- **Metrics**: Reports R², RMSE, MAE, MAPE, CV(RMSE) and NRMSE on the held-out original test set.

### `original_dataset.py`
- **Classical ML baselines**: Linear, Ridge, Lasso, SVR, Decision Tree, Random Forest, Gradient Boosting, with degree-2 polynomial features and `GridSearchCV` tuning for Random Forest and Gradient Boosting.
- **Gradient boosting pipelines**: XGBoost and LightGBM via a `ColumnTransformer`/`OneHotEncoder` pipeline.
- **Transformer-embedding hybrid**: GPT-2 mean-pooled text embeddings concatenated with scaled numeric features, fed to an XGBoost regressor.
- **LLM fine-tuning**: Mistral-7B-Instruct and DeepSeek-Coder-7B-Instruct, both 4-bit quantised with LoRA adapters, fine-tuned to generate the cost as text and parsed back out with a regex.
- **Quantum-classical regressor**: A PennyLane variational circuit (4 qubits, `StronglyEntanglingLayers`) embedded in a small feedforward network, trained with early stopping (the Hybrid Quantum SVR of Chapter 7).
- **Additional experiments**: a Keras "engression" FCNN, a LOWESS-based "LASER" demo, an LSTM applied to non-temporal tabular data via artificial windowing, a custom `LASERRegressor`, a `RandomForestRegressor` used as an approximation of an Engression baseline, an autoencoder + learned gating network blending Huber/Gaussian Process/XGBoost predictions, and correlation/feature-importance plots.

## Known Limitations

`original_dataset.py` reads from several different CSV filenames across its cells (`Processed_Datass.csv`, `Processed_Data.csv`, `Finaldataset9.csv`), which may or may not represent the same underlying dataset — this needs verifying, since results from different cells are only directly comparable if the underlying data is identical. The two GPT-2+XGBoost cells and the two DeepSeek fine-tuning cells are near-duplicates and could be consolidated.

## Usage

`augmented_dataset.py` expects `augmented_dataset_final.csv`, which already contains the Gaussian-noise copies of the training records (the augmentation is applied when the CSV is created, not in this script), with `record_id`, `is_augmented`, and `source_record_id` columns present.

```bash
python augmented_dataset.py
```

`original_dataset.py` is organised as a sequence of largely independent cells; run the whole file or individual sections as needed.

```bash
python original_dataset.py
```

## Outputs

`augmented_dataset.py` prints R², RMSE, MAE, MAPE, CV(RMSE) and NRMSE for the neural meta-learner on the held-out test set; it does not currently write these to a CSV file.

`original_dataset.py` prints RMSE/MAE/R² per model to stdout as each cell runs; it does not write results to CSV either, and also produces three matplotlib/seaborn plots (correlation heatmap, Random Forest feature importance, permutation importance).

## Configuration

Key parameters, by script:

| Script | Parameter | Value |
|--------|-----------|-------|
| `augmented_dataset.py` | XGBoost | 300 trees, depth 6, lr 0.05 |
| `augmented_dataset.py` | CatBoost | 300 iterations, depth 6, lr 0.05 |
| `augmented_dataset.py` | MLP encoder | 128-64-32, dropout 0.3/0.2 |
| `augmented_dataset.py` | Synthesiser | 64-32, dropout 0.3 after the first layer |
| `augmented_dataset.py` | Training | 100 epochs, batch 32, EarlyStopping patience 10, 10% validation split |
| `original_dataset.py` | GPT-2+XGBoost | XGBoost: 150 trees, depth 6, lr 0.05 |
| `original_dataset.py` | Mistral / DeepSeek LoRA | r=16, alpha=32, dropout=0.05, 3 epochs, lr=2e-5, 4-bit quantised |
| `original_dataset.py` | Quantum-classical regressor | 4 qubits, 3 entangling layers, Adam lr=5e-4, up to 400 epochs, patience 10 |
| `original_dataset.py` | Random Forest / Gradient Boosting tuning | `GridSearchCV`, `cv=5` |

## Datasets

- **`augmented_dataset_final.csv`**: used by `augmented_dataset.py`. Must include `record_id`, `is_augmented`, `source_record_id` for the train-test split, plus `contract_contract_sum` and `cost_increment` if present (these are excluded).
- **`Processed_Datass.csv`**, **`Processed_Data.csv`**, **`Finaldataset9.csv`**: three differently-named files read by different cells of `original_dataset.py`.

### Features

Feature sets differ across cells within `original_dataset.py` (some use 5 columns, some use 7, some add `original_location_factor` or `location`); there is no single fixed feature set across the whole file.

**Target (all scripts):**
- `cost_rebased` (numerical)

## Requirements

See `../Requirements.txt` for full dependency list. Key packages actually used across these two scripts:
- `pandas`, `numpy`
- `scikit-learn` (ML models, preprocessing, ensembles, `GridSearchCV`)
- `xgboost`, `lightgbm`, `catboost` (gradient boosting)
- `tensorflow`, `keras` (deep learning, neural meta-learner)
- `torch`, `transformers`, `peft`, `accelerate`, `datasets`, `bitsandbytes` (GPT-2 embeddings, Mistral/DeepSeek LoRA fine-tuning)
- `pennylane` (quantum-classical regressor)
- `statsmodels` (LOWESS)
- `matplotlib`, `seaborn` (plots)
- `tqdm`

## Files

- **`augmented_dataset.py`**: Hybrid MLP + XGBoost + CatBoost neural meta-learner on the augmented dataset.
- **`original_dataset.py`**: Collection of classical, transformer-hybrid, LLM fine-tuning, and quantum-classical regression experiments.
- **`Dataset/`**: Directory for storing dataset files.

## Key Findings

The scripts investigate:
1. **Does training-only Gaussian-noise augmentation** improve the Hybrid MLP + XGBoost + CatBoost neural meta-learner while preserving an untouched original test partition?
2. **Can LLM fine-tuning** (Mistral, DeepSeek) predict construction cost as generated text?
3. **Does quantum-classical fusion** via a variational circuit improve on classical regressors?
4. **How do classical and gradient-boosting baselines**, with and without hyperparameter tuning, compare as a reference point for the above?

## Citation

Part of PhD research on hybrid multimodal and quantum machine learning architectures.
