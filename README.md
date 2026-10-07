Hybrid Multimodal and Quantum Machine Learning Architectures

This repository contains the official code, models and experimental setups for my PhD research. The methodology investigates multimodal and hybrid fusion architectures and their effect on classification and prediction performance across medical, scientific and construction datasets.

Overview of Studies

The research is hypothesis-driven and divided into four core studies:

Study 1: Quantum & Classical SVM Fusion (Medical Tabular Data)

It is hypothesised that the fusion of a Quantum Support Vector Machine with a classical Support Vector Machine through ensemble methods will improve breast cancer classification accuracy beyond that of either paradigm in isolation. Evaluated on the WBCD and BCCD datasets.

Study 2: Mid-Level Multimodal Fusion (Medical Imaging & Text)

It is hypothesised that the mid-level fusion of BERT, GPT-2 and ResNet18 representations will outperform single-modality baselines on combined clinical text and ultrasound image data. Evaluated on the Breast Lesions Ultrasound (USG) Collection.

Study 3: Multimodal Scientific PDF Classification

It is hypothesised that the multimodal fusion of BERT, GPT-2 and a ResNet-18 vision encoder over document-extracted images will outperform classical, ensemble and single-transformer baselines on full-text scientific PDF classification. Evaluated on the NUS Keyphrase Extraction Corpus. This repository currently includes the single-transformer ablations and the proposed multimodal model; the classical ML and ensemble baseline layers are not yet included.

Study 4: Advanced Regression Fusion Strategies (Construction Costs)

It is hypothesised that three distinct fusion strategies (quantum-enhanced regression, transformer-boosted regression and neural-meta-learner ensemble fusion) will outperform traditional regressors on UK construction cost prediction. Evaluated on the BCIS Construction Cost Dataset.

Experimental Design & Validation

This codebase adopts a post-positivist quantitative experimental design. To ensure rigorous validation and prevent data leakage, the following protocols are enforced across the repository:

Strict Partitioning: Preprocessing transformations (e.g. Z-score standardisation, TF-IDF vectorisation and PCA) are fitted exclusively on training partitions. Splits are performed at the patient/record level rather than the row level for datasets where this applies, so no patient or augmented-row-derived sample appears in both partitions.
Evaluation Metrics: Models are evaluated using accuracy, precision, recall and F1-score for classification tasks. Regression tasks are evaluated using MAE, RMSE and R².
Reproducibility: A random seed of 42 is fixed for Study 1's primary split and for Study 4; Studies 2 and 3 use a seven-seed protocol (42, 7, 123, 999, 2023, 8888, 7777) with Wilcoxon significance testing across seeds. Cross-validation splits, model initialisations and training loops are seeded throughout to ensure reproducibility.
Installation and Requirements

This project requires Python 3.10 and relies on specific library versions to ensure deterministic behaviour across PyTorch, TensorFlow and Qiskit environments.

To replicate the environment, run:

pip install -r requirements.txt
