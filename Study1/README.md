# Study 1: Quantum and Classical SVM Fusion for Breast Cancer Classification

This directory contains code to evaluate whether fusing a Quantum Support Vector Machine (QSVM) with a classical SVM through ensemble methods improves breast cancer classification on the Wisconsin Breast Cancer Dataset (WBCD).

## Overview

The `Code.py` script runs a single-seed evaluation of the proposed Hybrid Model. The Hybrid Model is built from a classical SVM, a QSVM and three ensembles (Bagging, Stacking and Voting). Only the Hybrid Model is trained and evaluated by the script. The other models are used only as its components.

## Key Features

- **Data processing:** Loads the Wisconsin Breast Cancer Dataset (WBCD, 569 samples: 357 benign, 212 malignant). Labels are recoded so that malignant = 1 and benign = 0.
- **Two input branches:**
  - The classical SVM uses all 30 features, standardised with `StandardScaler` fitted on the training split only.
  - The quantum branch scales the 30 features to [0, 1] with `MinMaxScaler` and reduces them to 4 dimensions with a Restricted Boltzmann Machine (RBM), matching the 4 qubits of the circuit. Both the scaler and the RBM are fitted on the training split only.
- **Components of the Hybrid Model:**
  - Classical SVM: RBF kernel (C=10, gamma=0.01, class_weight=balanced), all 30 features.
  - QSVM with RBM: QSVC with a ZZ feature map (reps=2, linear entanglement), C=1.0, class_weight=balanced.
  - Ensemble with Bagging: 5 QSVM estimators.
  - Ensemble with Stacking: QSVM and SVM base learners, Random Forest meta-learner (50 trees, max_depth=3), 5-fold internal cross-validation.
  - Ensemble with Voting: soft voting of QSVM and SVM, with weights proportional to each model's five-fold cross-validation accuracy on the training split (Equation 3.15 in the thesis).
- **Hybrid Model (proposed):** master soft vote, with equal weights, over the Bagging, Stacking and Voting ensembles. This is the only model evaluated.
- **Hyperparameters:** The values in the code are the converged values from the Bayesian optimisation reported in Section 4.2.1 of the thesis. The optimisation itself is not part of this script.
- **Metrics:** Accuracy, precision, recall, F1-score (positive class = malignant), ROC-AUC, Brier score and Expected Calibration Error (ECE).
- **Reproducibility:** A fixed seed (42) is used for the train-test split and the model initialisations.

## Usage

Run in a Kaggle notebook (recommended):
```python
# Ensure qiskit packages are installed
!pip install qiskit qiskit-machine-learning qiskit-algorithms

# Run the script
exec(open("Code.py").read())
```

Or run locally:
```bash
pip install qiskit qiskit-machine-learning qiskit-algorithms scikit-learn numpy pandas scipy
python Code.py
```

## Outputs

The script writes three CSV files to `/kaggle/working/`:

- `study1_results.csv`: results of the Hybrid Model on the evaluation seed (1 row). Columns: seed, model, accuracy, precision, recall, f1, roc_auc, brier_score, ece, fit_eval_time_s.
- `study1_validation_weights.csv`: the cross-validation accuracies and the resulting QSVM and SVM voting weights.
- `study1_warnings.csv`: any warnings raised during fitting and evaluation.

If `RUN_FIVE_FOLD_CV = True`, it also writes `study1_hybrid_5fold_cv.csv`.

## Architecture
```text
Wisconsin Breast Cancer Dataset (569 samples, 30 features)
    |
Train-test split (80:20, stratified, seed 42)
    |
    +-- Classical SVM branch
    |     StandardScaler (fitted on train) -> RBF SVM on all 30 features
    |
    +-- Quantum branch
          MinMaxScaler [0, 1] (fitted on train)
            -> RBM, 30 -> 4 dimensions (fitted on train)
            -> ZZ feature map (4 qubits, reps=2, linear)
            -> fidelity quantum kernel (StatevectorSampler, 1024 shots)
            -> QSVC

Components of the Hybrid Model:
  Bagging                       5 x quantum branch
  Stacking                      quantum branch + SVM branch, Random Forest meta-learner
  Voting                        quantum branch + SVM branch, soft vote, CV-accuracy weights

Hybrid Model (proposed)         soft vote (equal weights) of Bagging, Stacking and Voting
                                Evaluated on the 20% test split

Metrics: Accuracy, Precision, Recall, F1, ROC-AUC, Brier, ECE
```

## Probability extraction

The Hybrid Model and all its components expose `predict_proba`. The QSVC and classical SVM are both configured with `probability=True`. Probability-based metrics use the resulting positive-class probabilities, where malignant is coded as class 1. The Hybrid Model probability is obtained through the soft-voting aggregation of its component ensembles.

## Configuration

| Parameter             | Value                                 | Description                                                   |
| --------------------- | ------------------------------------- | ------------------------------------------------------------- |
| SEED                  | 42                                    | Fixed seed for reproducibility                                |
| N_QUBITS              | 4                                     | RBM output size and number of qubits                          |
| RBM n_components      | 4                                     | Output dimensions                                             |
| RBM n_iter            | 20                                    | Passes over the training data                                 |
| RBM learning_rate     | 0.01                                  | RBM learning rate                                             |
| SVM kernel            | rbf                                   | Radial basis function                                         |
| SVM C                 | 10                                    | Regularisation parameter                                      |
| SVM gamma             | 0.01                                  | Kernel coefficient                                            |
| SVM class_weight      | balanced                              | Class weighting                                               |
| Bagging n_estimators  | 5                                     | Number of QSVM base estimators                                |
| Stacking cv           | 5                                     | Internal cross-validation folds                               |
| Stacking meta-learner | Random Forest (50 trees, max_depth=3) | Final combiner                                                |
| Feature map           | ZZFeatureMap                          | Quantum encoding                                              |
| Feature map reps      | 2                                     | Repetitions                                                   |
| Entanglement          | linear                                | Qubit connectivity                                            |
| QSVC C                | 1.0                                   | Regularisation parameter                                      |
| QSVC class_weight     | balanced                              | Class weighting                                               |
| Backend               | StatevectorSampler                    | Noiseless statevector simulation with 1024-shot sampling      |

## Dataset

Wisconsin Breast Cancer Dataset (WBCD)

- Samples: 569 instances (357 benign, 212 malignant)
- Features: 30 input features. The SVM uses all 30. The quantum branch reduces them to 4 with the RBM.
- Task: binary classification (malignant vs benign)
- Train-test split: 80:20 stratified (seed 42)

## Metrics explained

| Metric      | Range  | Interpretation                                                               |
| ----------- | ------ | ---------------------------------------------------------------------------- |
| Accuracy    | [0, 1] | Proportion of correct predictions (higher is better)                         |
| Precision   | [0, 1] | Proportion of predicted malignant cases that are correct (higher is better)  |
| Recall      | [0, 1] | Proportion of actual malignant cases correctly identified (higher is better) |
| F1-score    | [0, 1] | Harmonic mean of precision and recall (higher is better)                     |
| ROC-AUC     | [0, 1] | Area under the ROC curve (higher is better)                                  |
| Brier score | [0, 1] | Mean squared error of probabilities (lower is better)                        |
| ECE         | [0, 1] | Expected Calibration Error (lower is better; 0 = perfect calibration)        |

Precision, recall and F1 are computed for the malignant class (`pos_label=1`).

## Requirements

See `../Requirements.txt` for the full dependency list. Key packages:

- qiskit (quantum circuits)
- qiskit-machine-learning (QSVC and fidelity quantum kernel)
- qiskit-algorithms (state fidelity estimation)
- scikit-learn (SVM, ensemble methods, RBM)
- pandas, numpy, scipy (data handling and numerical computation)

## Key hypotheses

This script evaluates the proposed Hybrid Model, a master ensemble of Bagging, Stacking and Voting that fuses a QSVM with a classical SVM, on the WBCD.

## Citation

Part of PhD research on hybrid multimodal and quantum machine learning architectures.
