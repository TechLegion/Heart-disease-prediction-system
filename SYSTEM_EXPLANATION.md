# Heart Disease Prediction System — Full Technical Explanation

This document explains the entire system from high-level architecture down to implementation details.

---

## 1. High-Level Architecture

The system implements the proposal: **“An Improved Heart Disease Prediction System Using Genetic Algorithm and Swarm Artificial Neural Network.”**

### 1.1 End-to-End Flow

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           DATA LAYER                                              │
│  UCI Heart Disease (local file / ucimlrepo / URL)  →  Raw DataFrame               │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      PREPROCESSING PIPELINE                                       │
│  Load → Missing values → Duplicates → Encode categorical → Normalize → Split      │
│  Output: X_train, X_test, y_train, y_test (+ optional X_val, y_val)              │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
          ┌─────────────────────────────┼─────────────────────────────┐
          ▼                             ▼                             ▼
┌──────────────────┐         ┌──────────────────┐         ┌──────────────────┐
│  BASELINE ANN    │         │  GA-OPTIMIZED    │         │  PSO-OPTIMIZED   │
│  (scikit-learn   │         │  ANN              │         │  ANN             │
│   MLPClassifier) │         │  (GA tunes ANN)   │         │  (PSO tunes ANN) │
└──────────────────┘         └──────────────────┘         └──────────────────┘
          │                             │                             │
          │                             └─────────────┬───────────────┘
          │                                           ▼
          │                             ┌──────────────────────────────┐
          │                             │  HYBRID GA-PSO-ANN           │
          │                             │  GA first → then PSO → ANN   │
          │                             └──────────────────────────────┘
          │                                           │
          └───────────────────────────┬───────────────┘
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  COMPARISON: Accuracy, Precision, Recall, F1, Confusion Matrix → CSV + Plot      │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 File and Module Roles

| Path | Role |
|------|------|
| `main.py` | Entry point: finds dataset, runs preprocessing, trains all four models, compares, saves CSV and plot. |
| `src/data_preprocessing.py` | Loads data (local / ucimlrepo / URL), cleans, encodes, normalizes, splits. |
| `src/baseline_ann.py` | Wrapper around scikit-learn `MLPClassifier`: train, predict, evaluate, get/set weights. |
| `src/ga_optimizer.py` | Genetic Algorithm (DEAP): population of “individuals” (ANN-related params), fitness = validation accuracy. |
| `src/pso_optimizer.py` | Particle Swarm Optimization (PySwarms): swarm of weight vectors, fitness = validation accuracy. |
| `src/hybrid_model.py` | Orchestrates GA → PSO → final ANN; uses same interfaces as baseline for predict/evaluate. |

---

## 2. Data Pipeline (Last-Minute Detail)

### 2.1 Where Data Comes From

**Priority order:**

1. **Local file**  
   Paths checked in order: `heart disease/`, `heart+disease/`, `data/`.  
   Filenames: `processed.cleveland.data`, then `cleveland.data`.  
   First existing path is used.

2. **ucimlrepo**  
   If no local file: `fetch_ucirepo(id=45)` (Heart Disease).  
   Returns `features` and `targets`; code concatenates them into one DataFrame and renames the target column to `target` if needed.

3. **Direct URL**  
   If ucimlrepo fails (ImportError or other): CSV from  
   `https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data`  
   with `header=None` and column names set to the standard 14 names.

### 2.2 Column Names (UCI Standard)

Exactly 14 names used everywhere:

- `age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalach`, `exang`, `oldpeak`, `slope`, `ca`, `thal`, `target`.

If a local file has more than 14 columns (e.g. extra “name”), only the first 14 are kept and assigned these names.

### 2.3 Preprocessing Steps (Order and Detail)

1. **Load**  
   As above; result is a single DataFrame with 14 columns.

2. **Missing values**  
   - Replace literal `'?'` with `np.nan`.  
   - Numeric columns `age`, `trestbps`, `chol`, `thalach`, `oldpeak`, `ca` are coerced with `pd.to_numeric(..., errors='coerce')`.  
   - For each column with NaNs: numeric → `fillna(median)`; otherwise → `fillna(mode()[0])` or 0.

3. **Duplicates**  
   `df.drop_duplicates()`; duplicate rows are removed.

4. **Categorical encoding**  
   Columns `sex`, `cp`, `fbs`, `restecg`, `exang`, `slope`, `ca`, `thal` are label-encoded with `sklearn.preprocessing.LabelEncoder` (fit on first use, stored in `self.label_encoders`).  
   Values are converted to string before encoding so mixed types don’t break.

5. **Target**  
   `target` is made binary: `(df['target'] > 0).astype(int)` so 0 = no disease, 1 = disease (UCI original 1–4 are all “disease”).

6. **Normalization**  
   All columns except `target` are scaled with `sklearn.preprocessing.StandardScaler` (zero mean, unit variance).  
   Feature list is stored in `self.feature_names`.

7. **Split**  
   `train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)` so 80% train, 20% test, class proportions preserved.

**Pipeline output:**  
`(X_train, X_test, y_train, y_test, feature_names)`.

In `main.py`, the 80% training block is split again with the same pattern (80/20, stratified) to get **train** and **validation** for GA/PSO/Hybrid. So effectively: 64% train, 16% validation, 20% test of the original data.

---

## 3. Baseline ANN (Last-Minute Detail)

### 3.1 What It Is

A thin wrapper around `sklearn.neural_network.MLPClassifier`.

- **Architecture:** One input layer (size = number of features), two hidden layers `(100, 50)`, one output layer (binary).
- **Default settings:**  
  `max_iter=1000`, `learning_rate_init=0.001`, `activation='relu'`, `solver='adam'`, `random_state=42`.  
  **Early stopping:** `early_stopping=True` → `validation_fraction=0.1`, `n_iter_no_change=10`.  
  If `early_stopping=False` (used inside GA/PSO to avoid tiny validation sets), `validation_fraction=0.0`.

### 3.2 Important Methods

- **`train(X_train, y_train)`**  
  Calls `self.model.fit(X_train, y_train)`. No extra logic.

- **`predict(X)` / `predict_proba(X)`**  
  Forward to `self.model.predict` and `predict_proba`.

- **`evaluate(X_test, y_test, verbose=True)`**  
  Computes accuracy, precision, recall, F1 (binary), confusion matrix; optionally prints them.

- **`get_weights()`**  
  Returns `{'weights': self.model.coefs_, 'biases': self.model.intercepts_}`.  
  `coefs_` is a list of weight matrices (one per layer), `intercepts_` a list of bias vectors.

- **`set_weights(weights, biases)`**  
  Assigns `self.model.coefs_` and `self.model.intercepts_` so that GA/PSO can install a candidate solution without changing architecture.

### 3.3 Why Early Stopping Can Be Turned Off

When the training set is very small (e.g. 10 samples in GA’s dummy fit), a 10% validation split gives 1 sample → stratified split fails (“test_size = 1 should be greater or equal to the number of classes = 2”). So inside GA and PSO the code uses `early_stopping=False` for any ANN that is fit on small or repeatedly-evaluated data.

---

## 4. Genetic Algorithm Optimizer (Last-Minute Detail)

### 4.1 Idea

- **Population:** A set of “individuals.” Each individual is a **list of floats** (a chromosome).
- **Fitness:** For each individual, build an ANN (with that individual’s parameters), train it on `X_train, y_train`, predict on `X_val`; fitness = **validation accuracy**.
- **Goal:** Maximize fitness over generations using selection, crossover, and mutation (DEAP).

### 4.2 What an “Individual” Encodes

- **Creation (`_create_individual`):**  
  A small ANN `(100, 50)` is created with `early_stopping=False`, and fit on `X_train[:10], y_train[:10]` only to **initialize structure**.  
  Then all weight and bias values are flattened in order (layer by layer) into a list, and one extra float is appended (e.g. 0.001) to represent learning rate.  
  So the chromosome = `[w1, w2, ..., wn, learning_rate]`.

- **Interpretation (`_individual_to_ann`):**  
  In the current implementation, the chromosome is **not** fully decoded back into weights. Only the last element is used:  
  `learning_rate = max(0.0001, min(0.01, individual[-1] * 0.01))`, and the ANN’s `learning_rate_init` is set to that.  
  So effectively the GA is optimizing **learning rate** (and the rest of the chromosome participates in crossover/mutation but is not yet mapped back to weights). The ANN is then trained from scratch (with that learning rate) for 100 iterations when evaluating an individual.

### 4.3 DEAP Setup

- **Fitness:** `creator.FitnessMax` with `weights=(1.0,)` → single objective, maximize.
- **Individual:** `creator.Individual` = list with a `fitness` attribute.
- **Operators:**  
  - **Evaluate:** `_evaluate_individual(ind)` → train ANN, return `(validation_accuracy,)`.  
  - **Mate:** `tools.cxBlend`, alpha=0.5 (blend crossover).  
  - **Mutate:** `_mutate_individual(ind, indpb=0.1)` → each gene with probability 0.1 gets Gaussian noise (0, 0.1), then clipped to [-10, 10].  
  - **Select:** `tools.selTournament`, tournsize=3.

### 4.4 Evolution Loop

1. Build initial population: `n_population` individuals via `_create_individual`.
2. Evaluate all, assign `ind.fitness.values`.
3. For each generation:
   - Select `len(population)` individuals (tournament).
   - Clone them into `offspring`.
   - For each pair of offspring, with probability `crossover_prob` apply mate; invalidate fitness.
   - For each offspring, with probability `mutation_prob` apply mutate; invalidate fitness.
   - Evaluate only those with invalid fitness.
   - Replace population with offspring.
4. Best individual = `tools.selBest(population, 1)[0]`.

### 4.5 Output

- `get_optimized_ann()` returns the ANN that corresponds to the best individual (same `_individual_to_ann` logic: mainly “best learning rate” and a freshly trained ANN with that setting).

---

## 5. Particle Swarm Optimization (Last-Minute Detail)

### 5.1 Idea

- **Swarm:** `n_particles` particles; each particle is a **vector of length D**, where D = total number of ANN parameters (all weights and biases flattened).
- **Position:** A point in R^D = one full set of ANN weights and biases.
- **Fitness:** For each particle, set the ANN’s weights from the vector, train the ANN on `X_train`, predict on `X_val`; fitness = validation accuracy. PySwarms minimizes, so the code passes **negative accuracy** as cost.

### 5.2 ANN ↔ Vector Mapping

- **Reference ANN:**  
  Either provided as `initial_ann` or created by a dummy fit on `X_train[:10], y_train[:10]` with `early_stopping=False`. This fixes the **architecture** and the **shapes** of all weight and bias matrices.

- **`_ann_to_vector(ann)`:**  
  Flatten `ann.get_weights()['weights']` and `['biases']` in fixed order into one numpy vector.

- **`_vector_to_ann(vector)`:**  
  Build a new ANN with the same architecture; slice the vector according to the reference’s weight/biases shapes and fill `coefs_` and `intercepts_`; no random init. Then `set_weights(...)`.

So PSO **actually optimizes the full weight and bias vector** (unlike the current GA, which only uses the last gene as learning rate).

### 5.3 PSO Algorithm (PySwarms)

- **GlobalBestPSO:** One swarm, global best only.
- **Parameters:** `c1` (cognitive), `c2` (social), `w` (inertia); default `c1=0.5, c2=0.3, w=0.9`.
- **Bounds:** Default each dimension in `[-5, 5]`.
- **Initialization:**  
  If `initial_ann` is given, all particles are initialized as the same vector (from that ANN) plus small Gaussian noise (0, 0.1).  
  Otherwise PySwarms uses its default init within bounds.

### 5.4 Fitness and Optimization

- **Cost function:** For a swarm matrix of shape `(n_particles, D)`, for each row call `_vector_to_ann`, train that ANN, compute validation accuracy, return **-accuracy** (so lower cost = better).
- After optimization, `best_position` is the best vector found; `best_fitness` is stored as the positive accuracy (negate the returned cost).

### 5.5 Output

- `get_optimized_ann()` returns `_vector_to_ann(self.best_position)`.

---

## 6. Hybrid GA-PSO-ANN (Last-Minute Detail)

### 6.1 Role

Implements the proposal’s pipeline: **first GA, then PSO, then a final ANN** as the model the user calls `predict` / `evaluate` on.

### 6.2 Parameter Merging

- **GA params:** Defaults `n_population=30`, `n_generations=20`, `crossover_prob=0.7`, `mutation_prob=0.2`.  
  User can pass e.g. `ga_params={'n_population': 30, 'n_generations': 20}`; missing keys are filled from defaults (`{**_default_ga, **(ga_params or {})}`).
- **PSO params:** Defaults `n_particles=20`, `iterations=30`, `options={'c1': 0.5, 'c2': 0.3, 'w': 0.9}`. Same merge logic.

### 6.3 Training Steps

1. **GA step**  
   Build `GAOptimizer(X_train, y_train, X_val, y_val, ...)` with the merged GA params.  
   Run `optimize()`.  
   Get `ga_optimized_ann = get_optimized_ann()`, then train it again on full `X_train, y_train` (to align with how we use it as initial point for PSO).

2. **PSO step**  
   Build `PSOOptimizer(X_train, y_train, X_val, y_val, ...)` with merged PSO params.  
   Run `optimize(initial_ann=ga_optimized_ann)` so the swarm starts near the GA solution.  
   Get `pso_optimized_ann = get_optimized_ann()`.

3. **Final model**  
   Create a new `BaselineANN` (e.g. 100, 50, max_iter=500, default early_stopping).  
   Copy weights and biases from `pso_optimized_ann` via `set_weights(...)`.  
   Train this final model on `X_train, y_train` again.  
   This trained ANN is stored as `self.final_model` and used for all subsequent `predict` and `evaluate` calls.

So the “hybrid” is: **GA-optimized ANN → PSO fine-tuning of its weights → final ANN with one more training pass.**

### 6.4 Predict and Evaluate

- `predict(X)` and `predict_proba(X)` delegate to `self.final_model`.
- `evaluate(X_test, y_test)` uses the same metrics as the baseline (accuracy, precision, recall, F1, confusion matrix).

---

## 7. Main Script and Comparison (Last-Minute Detail)

### 7.1 Dataset Resolution

- Loop over `('heart disease', 'heart+disease', 'data')` and `('processed.cleveland.data', 'cleveland.data')`; take the first existing path.
- If found: `preprocess_pipeline(file_path=dataset_path)`.
- If not: `preprocess_pipeline()` with no path (ucimlrepo then URL fallback).  
  If that fails, print instructions and exit.

### 7.2 Data Splits Used

- Preprocessing returns 80% train, 20% test.
- `main` then does a stratified 80/20 split of the 80% train to get **train** and **validation**.  
  So: Train ≈ 64%, Val ≈ 16%, Test = 20% of original rows.

### 7.3 compare_models(…)

Runs four pipelines in order:

1. **Baseline ANN**  
   `BaselineANN(100, 50, max_iter=500).train(X_train, y_train).evaluate(X_test, y_test)`.

2. **GA-optimized ANN**  
   `GAOptimizer(...).optimize()` → `get_optimized_ann().train(X_train, y_train)` (on full train again) → `evaluate(X_test, y_test)`.  
   On exception, append a row of zeros for GA.

3. **PSO-optimized ANN**  
   Uses the **same** `baseline_ann` instance as initial ANN:  
   `PSOOptimizer(...).optimize(initial_ann=baseline_ann)` → get optimized ANN, train on full train, evaluate.  
   On exception, append zeros for PSO.

4. **Hybrid GA-PSO-ANN**  
   `HybridGAPSOANN(...).train(X_train, y_train, X_val, y_val)` then `evaluate(X_test, y_test)`.  
   On exception, append zeros for Hybrid.

Each model’s Accuracy, Precision, Recall, F1-Score are stored in a list of dicts, then turned into a pandas DataFrame.

### 7.4 Outputs

- **Console:** Per-model evaluation summaries and a final table of the four models; “Best Model” and “Best Accuracy” from the DataFrame.
- **Files:**  
  - `results/comparison_results.csv`: same DataFrame.  
  - `results/comparison.png`: bar chart of Accuracy, Precision, Recall, F1-Score for each of the four models (y-axis 0–1).

---

## 8. Summary Table: Who Uses What

| Component        | Train data     | Validation data | Test data   | Output / Role                          |
|-----------------|----------------|-----------------|------------|----------------------------------------|
| Preprocessing   | —              | —               | —          | Produces train/test (and main splits to train/val). |
| Baseline ANN    | X_train, y_train | (internal 10% if early_stopping) | X_test, y_test | Single ANN; baseline metrics.          |
| GA              | X_train, y_train | X_val, y_val  | —          | Best individual → “GA-optimized” ANN.  |
| PSO             | X_train, y_train | X_val, y_val  | —          | Best position → “PSO-optimized” ANN.  |
| Hybrid          | X_train, y_train | X_val, y_val  | —          | GA then PSO then final ANN.            |
| main compare_models | Same train/val for all; each model evaluated on same X_test, y_test. | | | CSV + plot. |

---

## 9. Design Choices (Why Things Are This Way)

- **scikit-learn MLPClassifier:** Keeps the project free of a heavy deep-learning stack; sufficient for the UCI Heart Disease size; has `coefs_`/`intercepts_` for GA/PSO weight injection.
- **Early stopping off in GA/PSO:** Avoids stratified split errors when fitting on very small data or many times; validation is already done explicitly on `X_val, y_val`.
- **GA currently only learning rate:** The chromosome carries full weight vectors from the dummy ANN, but `_individual_to_ann` only uses the last gene. Extending to full weight encoding would require matching shapes and careful crossover/mutation (as in PSO).
- **PSO on full weights:** PSO works in the same dimension as the full parameter vector and uses a reference ANN only for shape; so it truly fine-tunes every weight and bias.
- **Hybrid order (GA → PSO):** Matches the proposal: GA for broader search / hyperparameter, PSO for local refinement of the ANN parameters (here, from the GA-produced ANN as starting point).

This is the full picture from architecture down to the last detail in the current codebase.
