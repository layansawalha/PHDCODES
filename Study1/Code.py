import os
import time
import warnings

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import (
    BaggingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.neural_network import BernoulliRBM
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

from qiskit.circuit.library import ZZFeatureMap
from qiskit.primitives import StatevectorSampler
from qiskit_algorithms.state_fidelities import ComputeUncompute
from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.kernels import FidelityQuantumKernel


SEED = 42
N_QUBITS = 4
RBM_N_ITER = 20
KERNEL_SHOTS = 1024
OUTPUT_DIR = "/kaggle/working"
RUN_FIVE_FOLD_CV = False
os.makedirs(OUTPUT_DIR, exist_ok=True)


def expected_calibration_error(y_true, y_prob, n_bins=10):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            in_bin = (y_prob >= lo) & (y_prob <= hi)
        else:
            in_bin = (y_prob >= lo) & (y_prob < hi)

        n_in_bin = int(in_bin.sum())
        if n_in_bin == 0:
            continue

        observed_event_rate = y_true[in_bin].mean()
        mean_confidence = y_prob[in_bin].mean()

        ece += (n_in_bin / len(y_prob)) * abs(observed_event_rate - mean_confidence)

    return float(ece)


class ClassicalSVMBranch(ClassifierMixin, BaseEstimator):
    def __init__(self, C=10.0, gamma=0.01, class_weight="balanced", random_state=42):
        self.C = C
        self.gamma = gamma
        self.class_weight = class_weight
        self.random_state = random_state

    def fit(self, X, y):
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)

        self.model_ = SVC(
            kernel="rbf",
            C=self.C,
            gamma=self.gamma,
            class_weight=self.class_weight,
            probability=True,
            random_state=self.random_state,
        )
        self.model_.fit(X_scaled, y)
        self.classes_ = self.model_.classes_
        return self

    def predict(self, X):
        return self.model_.predict(self.scaler_.transform(X))

    def predict_proba(self, X):
        return self.model_.predict_proba(self.scaler_.transform(X))

    def decision_function(self, X):
        return self.model_.decision_function(self.scaler_.transform(X))


class QuantumSVMBranch(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        n_components=4,
        rbm_learning_rate=0.01,
        rbm_n_iter=20,
        qsvc_C=1.0,
        class_weight="balanced",
        reps=2,
        entanglement="linear",
        random_state=42,
    ):
        self.n_components = n_components
        self.rbm_learning_rate = rbm_learning_rate
        self.rbm_n_iter = rbm_n_iter
        self.qsvc_C = qsvc_C
        self.class_weight = class_weight
        self.reps = reps
        self.entanglement = entanglement
        self.random_state = random_state

    def _build_qsvc(self):
        feature_map = ZZFeatureMap(
            feature_dimension=self.n_components,
            reps=self.reps,
            entanglement=self.entanglement,
        )

        sampler = StatevectorSampler(default_shots=KERNEL_SHOTS, seed=self.random_state)
        fidelity = ComputeUncompute(sampler=sampler)
        qkernel = FidelityQuantumKernel(fidelity=fidelity, feature_map=feature_map)

        return QSVC(
            quantum_kernel=qkernel,
            C=self.qsvc_C,
            class_weight=self.class_weight,
            probability=True,
        )

    def fit(self, X, y):
        self.scaler_ = MinMaxScaler()
        X_scaled = self.scaler_.fit_transform(X)

        self.rbm_ = BernoulliRBM(
            n_components=self.n_components,
            learning_rate=self.rbm_learning_rate,
            n_iter=self.rbm_n_iter,
            random_state=self.random_state,
        )
        X_reduced = self.rbm_.fit_transform(X_scaled)

        self.model_ = self._build_qsvc()
        self.model_.fit(X_reduced, y)
        self.classes_ = self.model_.classes_
        return self

    def _transform(self, X):
        X_scaled = self.scaler_.transform(X)
        return self.rbm_.transform(X_scaled)

    def predict(self, X):
        return self.model_.predict(self._transform(X))

    def predict_proba(self, X):
        return self.model_.predict_proba(self._transform(X))

    def decision_function(self, X):
        return self.model_.decision_function(self._transform(X))


def validation_derived_voting_weights(X_train, y_train, seed):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    q_branch = QuantumSVMBranch(
        n_components=N_QUBITS,
        rbm_learning_rate=0.01,
        rbm_n_iter=RBM_N_ITER,
        qsvc_C=1.0,
        random_state=seed,
    )
    c_branch = ClassicalSVMBranch(random_state=seed)

    q_score = cross_val_score(q_branch, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=1).mean()
    c_score = cross_val_score(c_branch, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=1).mean()

    raw = np.array([q_score, c_score], dtype=float)
    if raw.sum() <= 0:
        weights = np.array([0.5, 0.5])
    else:
        weights = raw / raw.sum()

    return weights.tolist(), {"QSVM_cv_accuracy": q_score, "SVM_cv_accuracy": c_score}


def build_models(seed, X_train, y_train):
    qsvc = QuantumSVMBranch(
        n_components=N_QUBITS,
        rbm_learning_rate=0.01,
        rbm_n_iter=RBM_N_ITER,
        qsvc_C=1.0,
        random_state=seed,
    )
    svm = ClassicalSVMBranch(C=10.0, gamma=0.01, random_state=seed)

    voting_weights, weight_info = validation_derived_voting_weights(X_train, y_train, seed)

    bagging = BaggingClassifier(estimator=qsvc, n_estimators=5, random_state=seed, n_jobs=1)

    stacking = StackingClassifier(
        estimators=[("qsvm", qsvc), ("svm", svm)],
        final_estimator=RandomForestClassifier(n_estimators=50, max_depth=3, random_state=seed),
        stack_method="predict_proba",
        cv=5,
        n_jobs=1,
    )

    voting = VotingClassifier(
        estimators=[("qsvm", qsvc), ("svm", svm)],
        voting="soft",
        weights=voting_weights,
        n_jobs=1,
    )

    hybrid = VotingClassifier(
        estimators=[
            ("bagging_model", bagging),
            ("stacking_model", stacking),
            ("voting_model", voting),
        ],
        voting="soft",
        weights=None,
        n_jobs=1,
    )

    return hybrid, weight_info, voting_weights


def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    caught_rows = []
    t0 = time.time()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if not hasattr(model, "predict_proba"):
            raise RuntimeError(f"{name} does not expose predict_proba.")

        y_prob = model.predict_proba(X_test)[:, 1]

        result = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, pos_label=1, zero_division=0),
            "recall": recall_score(y_test, y_pred, pos_label=1, zero_division=0),
            "f1": f1_score(y_test, y_pred, pos_label=1, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_prob),
            "brier_score": brier_score_loss(y_test, y_prob),
            "ece": expected_calibration_error(y_test, y_prob, n_bins=10),
            "fit_eval_time_s": time.time() - t0,
        }

        for w in caught:
            caught_rows.append(
                {
                    "model": name,
                    "category": w.category.__name__,
                    "message": str(w.message),
                    "filename": os.path.basename(w.filename),
                    "lineno": int(w.lineno),
                }
            )

    return result, caught_rows


def run_one_seed(seed=SEED):
    data = load_breast_cancer()
    X = data.data
    y = 1 - data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=seed, stratify=y,
    )

    print(f"Train class counts [benign=0, malignant=1]: {np.bincount(y_train)}")
    print(f"Test class counts  [benign=0, malignant=1]: {np.bincount(y_test)}")

    hybrid, weight_info, voting_weights = build_models(seed, X_train, y_train)

    print("\nTraining-only validation scores used for SVM-QSVM voting weights:")
    print(weight_info)
    print(f"Normalised voting weights [QSVM, SVM] = {np.round(voting_weights, 4).tolist()}\n")

    metrics = {}
    warning_rows = []

    for name, model in {"Hybrid Model (proposed)": hybrid}.items():
        result, model_warnings = evaluate_model(name, model, X_train, y_train, X_test, y_test)
        metrics[name] = result
        warning_rows.extend(model_warnings)

        print(
            f"{name:<24} "
            f"acc={result['accuracy']:.4f} "
            f"prec={result['precision']:.4f} "
            f"rec={result['recall']:.4f} "
            f"f1={result['f1']:.4f} "
            f"auc={result['roc_auc']:.4f} "
            f"brier={result['brier_score']:.4f} "
            f"ece={result['ece']:.4f} "
            f"({result['fit_eval_time_s']:.1f}s)"
        )

    return X, y, metrics, warning_rows, weight_info, voting_weights


def run_five_fold_hybrid_cv(X, y, seed=SEED):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    fold_rows = []

    for fold_no, (tr_idx, te_idx) in enumerate(cv.split(X, y), start=1):
        X_train, X_test = X[tr_idx], X[te_idx]
        y_train, y_test = y[tr_idx], y[te_idx]

        hybrid, _, _ = build_models(seed + fold_no, X_train, y_train)

        result, fold_warnings = evaluate_model(
            f"Hybrid_CV_fold_{fold_no}", hybrid, X_train, y_train, X_test, y_test,
        )

        fold_rows.append(
            {"fold": fold_no, "accuracy": result["accuracy"], "n_warnings": len(fold_warnings)}
        )
        print(f"CV fold {fold_no}: accuracy={result['accuracy']:.4f}, warnings={len(fold_warnings)}")

    df_cv = pd.DataFrame(fold_rows)
    print(
        "\nHybrid 5-fold CV: "
        f"mean={df_cv['accuracy'].mean():.4f}, "
        f"SD={df_cv['accuracy'].std(ddof=1):.4f}, "
        f"min={df_cv['accuracy'].min():.4f}, "
        f"max={df_cv['accuracy'].max():.4f}"
    )
    return df_cv


def main():
    print(f"Study 1 WBCD evaluation - seed={SEED}, QSVM={N_QUBITS} RBM components/qubits")

    X, y, metrics, warning_rows, weight_info, voting_weights = run_one_seed(SEED)

    rows = [{"seed": SEED, "model": name, **values} for name, values in metrics.items()]
    df_results = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
    df_results.to_csv(f"{OUTPUT_DIR}/study1_results.csv", index=False)

    df_weights = pd.DataFrame(
        [{**weight_info, "QSVM_vote_weight": voting_weights[0], "SVM_vote_weight": voting_weights[1]}]
    )
    df_weights.to_csv(f"{OUTPUT_DIR}/study1_validation_weights.csv", index=False)

    df_warnings = pd.DataFrame(
        warning_rows, columns=["model", "category", "message", "filename", "lineno"],
    )
    df_warnings.to_csv(f"{OUTPUT_DIR}/study1_warnings.csv", index=False)

    print("\nFinal results:")
    with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 220):
        print(df_results.to_string(index=False))

    print("\nWarning report:")
    if df_warnings.empty:
        print("No warnings were emitted during fit/evaluation.")
    else:
        print(df_warnings.to_string(index=False))

    print(
        f"\nSaved:\n"
        f"  {OUTPUT_DIR}/study1_results.csv\n"
        f"  {OUTPUT_DIR}/study1_validation_weights.csv\n"
        f"  {OUTPUT_DIR}/study1_warnings.csv"
    )

    if RUN_FIVE_FOLD_CV:
        df_cv = run_five_fold_hybrid_cv(X, y, SEED)
        df_cv.to_csv(f"{OUTPUT_DIR}/study1_hybrid_5fold_cv.csv", index=False)
        print(f"  {OUTPUT_DIR}/study1_hybrid_5fold_cv.csv")


if __name__ == "__main__":
    main()
