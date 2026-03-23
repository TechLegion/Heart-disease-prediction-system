# Proposed Methodology — Explanation for Proposal Defence

This document explains the proposed methodology and how it is carried out in the Heart Disease Prediction System (GA + Swarm-Optimized ANN).

---

## (i) Data Collected from UCI Heart Disease Dataset

**What is proposed**  
The study uses a well-known, publicly available dataset so that results are reproducible and comparable with other research. The UCI Machine Learning Repository Heart Disease dataset is widely used in medical prediction studies.

**How it is carried out**

- **Source:** The dataset is the **Heart Disease dataset** (Cleveland subset) from the UCI ML Repository. It can be obtained in three ways in this project:
  1. **Local file:** From the project folder (e.g. `heart disease/processed.cleveland.data` or `data/processed.cleveland.data`).
  2. **Programmatic fetch:** Using the `ucimlrepo` library (`fetch_ucirepo(id=45)`) when the package is installed, so the dataset is downloaded automatically if no local file is found.
  3. **Direct URL:** As a fallback, the raw CSV is downloaded from the UCI archive URL.

- **Content:** The dataset contains **303 patient records** (after loading), with **14 attributes**: age, sex, chest pain type (cp), resting blood pressure (trestbps), serum cholesterol (chol), fasting blood sugar (fbs), resting ECG (restecg), maximum heart rate (thalach), exercise-induced angina (exang), ST depression (oldpeak), slope of ST segment (slope), number of major vessels (ca), thalassemia (thal), and the **target** (presence or severity of heart disease). This aligns with the proposal’s use of “reliable sources such as the UCI Machine Learning Repository.”

---

## (ii) Data Preprocessing: Cleaning, Normalization, Encoding, and Splitting

**What is proposed**  
Before building the model, the data must be cleaned, normalized, encoded, and split into training and testing (and optionally validation) sets so that the model is trained and evaluated on consistent, comparable scales and without missing or invalid values.

**How it is carried out**

- **Cleaning**
  - **Missing values:** The UCI dataset uses `?` for missing values. These are replaced with `NaN`, then filled: **numeric** columns (age, trestbps, chol, thalach, oldpeak, ca) with the **median** of the column; **categorical** columns with the **mode** (or 0 if no mode). This follows the proposal’s “replace with statistical measures (mean/median).”
  - **Duplicates and inconsistencies:** Duplicate rows are removed with `drop_duplicates()` to avoid bias and over-representation of the same patient.

- **Encoding**
  - Categorical variables (sex, cp, fbs, restecg, exang, slope, ca, thal) are converted to numbers using **label encoding** (e.g. `sklearn.preprocessing.LabelEncoder`) so the ANN can use them as inputs. The proposal’s “encoding categorical attributes” is implemented in the preprocessing module.

- **Normalization**
  - Numerical features are put on a common scale using **Z-score standardization** (zero mean, unit variance) via `StandardScaler`. This avoids features with larger ranges (e.g. cholesterol) dominating the network and improves training. The proposal mentions “Min-Max Scaling or Z-score Standardization”; this implementation uses Z-score.

- **Splitting**
  - The data is split into **training** and **testing** sets (e.g. 80% train, 20% test) with **stratified sampling** so that the proportion of “disease” vs “no disease” is preserved in both sets. For the hybrid and optimization phases, the training portion is split again to obtain a **validation** set (e.g. 80% of the 80% for train, 20% of the 80% for validation). The proposal’s “splitting the dataset into training and testing sets” is thus carried out with a clear, reproducible split strategy.

- **Target preparation**
  - The UCI target has values 0–4 (0 = no disease, 1–4 = disease). The study uses **binary classification**: target is converted to 0 (no disease) and 1 (disease) by mapping any value > 0 to 1.

All of the above is implemented in a single **preprocessing pipeline** in `data_preprocessing.py`, which runs in the order: load → clean (missing, duplicates) → encode → prepare target → normalize → split.

---

## (iii) Development of ANN Prediction Model

**What is proposed**  
An Artificial Neural Network (ANN) is developed as the core prediction model. The proposal states that ANNs can learn complex, non-linear relationships in medical data and are suitable for heart disease prediction.

**How it is carried out**

- **Architecture:** A **Multi-Layer Perceptron (MLP)** with:
  - **Input layer:** Number of units equals the number of features (13 after excluding the target).
  - **Hidden layers:** Two hidden layers (e.g. 100 and 50 neurons) to capture non-linear patterns without excessive complexity.
  - **Output layer:** One unit for binary classification (heart disease present or not).
  - **Activation:** ReLU in hidden layers; output uses the appropriate activation for classification (e.g. softmax/logistic in the library).

- **Training mechanism:** The ANN is trained using **backpropagation** with an **Adam** optimizer (adaptive learning rate). Options such as **early stopping** (using a small validation fraction) help prevent overfitting when enough data is available.

- **Implementation:** The ANN is implemented using **scikit-learn’s MLPClassifier** in `baseline_ann.py`. This provides a standard, reproducible ANN that serves as:
  - The **baseline model** for comparison, and
  - The **underlying model** whose parameters (weights, biases) and hyperparameters (e.g. learning rate) are later optimized by the Genetic Algorithm and Swarm Intelligence.

So “development of ANN prediction model” is carried out by defining this architecture and training procedure, and by exposing methods to get/set weights so that GA and PSO can optimize them.

---

## (iv) Optimization Using Genetic Algorithm and Swarm Intelligence

**What is proposed**  
The proposal states that ANN performance depends on weights, learning rate, and structure, and that optimization techniques can improve accuracy. It proposes using a **Genetic Algorithm (GA)** and **Swarm Intelligence (e.g. Particle Swarm Optimization, PSO)** to optimize the ANN.

**How it is carried out**

- **Genetic Algorithm (GA)**
  - **Role:** GA is used to search for better ANN configurations. In this implementation, each **individual** in the population encodes ANN-related parameters (e.g. a learning rate and, in extended form, weights). The **fitness** of an individual is the **validation accuracy** of the ANN when trained with that configuration.
  - **Process:** A **population** of individuals is created; each is **evaluated** by building/training the ANN and measuring validation accuracy. The best individuals are **selected** (e.g. tournament selection), **crossover** (e.g. blend crossover) combines two parents to produce offspring, and **mutation** (e.g. small random changes) adds diversity. This repeats for a fixed number of **generations**.
  - **Result:** The best individual after the last generation gives an **optimized** ANN (e.g. best learning rate and, if implemented, initial weights). This is the “GA-optimized” model and is also used as the starting point for PSO in the hybrid.
  - **Implementation:** Carried out in `ga_optimizer.py` using the DEAP library (selection, crossover, mutation, fitness evaluation).

- **Swarm Intelligence (Particle Swarm Optimization – PSO)**
  - **Role:** PSO fine-tunes the ANN’s **weights and biases** by treating them as a single long vector. Each **particle** in the swarm is one such vector (one full set of ANN parameters). The **fitness** of a particle is again the **validation accuracy** after setting the ANN to those weights and training (or fine-tuning).
  - **Process:** A **swarm** of particles moves through the space of possible weight vectors. Each particle updates its position using its own best position and the swarm’s global best, with parameters such as inertia (w), cognitive (c1), and social (c2) coefficients. The process runs for a fixed number of **iterations**.
  - **Result:** The **best position** (best weight vector) found by the swarm is used to set the ANN’s weights and biases, giving the “PSO-optimized” model.
  - **Implementation:** Carried out in `pso_optimizer.py` using the PySwarms library (GlobalBestPSO). The same ANN can be converted to a vector and back using a reference architecture so that PSO optimizes the full parameter set.

- **Hybrid (GA + PSO)**  
  The **Hybrid GA-PSO-ANN** model implements the proposal’s idea of combining both methods: first GA is run to get an optimized ANN; then PSO is run **starting from that GA-optimized ANN** to further refine weights and biases. The final model is this PSO-refined ANN (possibly with one more training pass). This is implemented in `hybrid_model.py`.

So “optimization using Genetic Algorithm and Swarm Intelligence” is carried out by (a) GA optimizing ANN configuration/parameters with validation accuracy as fitness, and (b) PSO optimizing the full weight vector of the ANN (optionally from the GA solution), again using validation accuracy.

---

## (v) Model Training and Testing

**What is proposed**  
Models are trained on the training set and tested on a separate testing set to ensure that reported performance reflects generalization and not just memorization of the training data.

**How it is carried out**

- **Training**
  - **Baseline ANN:** Trained on the **training set** (e.g. 80% of the data after the first split, or the 64% “train” portion when a validation set is used).
  - **GA-optimized ANN:** During GA, each candidate is evaluated by training the ANN on the training set and measuring validation accuracy. After GA, the **best** ANN is trained again on the full training set (or the same train split used for comparison) before being evaluated on the test set.
  - **PSO-optimized ANN:** During PSO, each particle’s ANN is trained (or fine-tuned) on the training set and evaluated on the validation set. The **best** particle’s ANN is then trained on the full training set before testing.
  - **Hybrid GA-PSO-ANN:** GA is run on train/validation; the GA-optimized ANN is trained on the training set; PSO starts from that ANN and again uses train/validation; the PSO-optimized ANN is set as the final model and trained one more time on the full training set.

- **Testing**
  - All four models (Baseline, GA-optimized, PSO-optimized, Hybrid) are evaluated on the **same held-out test set** (e.g. 20% of the data) that is not used for training or for selecting GA/PSO parameters. This gives a fair comparison and an estimate of real-world performance.

So “model training and testing” is carried out by training each model (with the described use of train/validation where applicable) and then testing once on the fixed test set.

---

## (vi) Performance Evaluation Using Accuracy, Precision, Recall, and F1-Score

**What is proposed**  
The proposal states that the performance of the hybrid GA–PSO–ANN model is compared with existing models using standard evaluation metrics. The metrics mentioned are accuracy, precision, recall, and F1-score.

**How it is carried out**

- **Metrics (binary classification)**  
  For “heart disease present” as the positive class (1):
  - **Accuracy:** (TP + TN) / (TP + TN + FP + FN) — proportion of all predictions that are correct.
  - **Precision:** TP / (TP + FP) — among predicted positives, proportion that are actually positive (relevant when false positives are costly).
  - **Recall (Sensitivity):** TP / (TP + FN) — among actual positives, proportion that are correctly predicted (relevant for not missing diseased patients).
  - **F1-Score:** Harmonic mean of precision and recall: 2 × (Precision × Recall) / (Precision + Recall) — balances precision and recall in one number.

- **Implementation**  
  These are computed using **scikit-learn** functions (`accuracy_score`, `precision_score`, `recall_score`, `f1_score`) with `average='binary'` and `zero_division=0` where applicable. A **confusion matrix** (TP, FN, FP, TN) is also produced for each model.

- **Comparison**  
  All four models (Baseline ANN, GA-optimized ANN, PSO-optimized ANN, Hybrid GA-PSO-ANN) are evaluated on the **same test set**, and their Accuracy, Precision, Recall, and F1-Score are tabulated (e.g. in `results/comparison_results.csv`) and visualized (e.g. bar chart in `results/comparison.png`). The “best” model can be chosen (e.g. by accuracy or F1) for reporting.

So “performance evaluation using Accuracy, Precision, Recall, and F1-Score” is carried out by computing these four metrics (plus confusion matrix) for each model on the test set and comparing them in a table and a plot.

---

## Summary Table for Defence

| Step | Proposed | Carried out in the project |
|------|----------|----------------------------|
| (i)  | Data from UCI Heart Disease Dataset | Local file / ucimlrepo(id=45) / UCI URL; Cleveland data, 14 attributes, 303 records. |
| (ii) | Preprocessing: cleaning, normalization, encoding, splitting | Clean: missing (median/mode), duplicates removed. Encode: label encoding for categoricals. Normalize: Z-score. Split: 80/20 stratified; train further split for validation. |
| (iii)| Development of ANN | MLP: input → (100, 50) hidden → output; ReLU, Adam; implemented with MLPClassifier in baseline_ann.py. |
| (iv) | Optimization: GA and Swarm Intelligence | GA: DEAP, fitness = validation accuracy, optimizes ANN config (e.g. learning rate). PSO: PySwarms, fitness = validation accuracy, optimizes full weight vector. Hybrid: GA then PSO from GA solution. |
| (v)  | Model training and testing | Train on training (and validation for GA/PSO); test on same held-out test set for all four models. |
| (vi) | Evaluation: Accuracy, Precision, Recall, F1-Score | sklearn metrics on test set; comparison table (CSV) and bar chart (PNG) for Baseline, GA-ANN, PSO-ANN, Hybrid. |

---

Using this document, you can walk the panel through each point of the methodology and state clearly what was proposed and how it is carried out in your implementation.
