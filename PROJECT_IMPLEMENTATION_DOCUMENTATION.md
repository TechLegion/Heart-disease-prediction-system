# Final Year Project — Complete Implementation Documentation

**Title (working):** An Improved Heart Disease Prediction System Using Genetic Algorithm and Swarm-Optimized Artificial Neural Network  

**Institution:** Redeemer's University, Ede, Osun State, Nigeria  
**Academic session:** 2025/2026  
**Project type:** Final-year undergraduate project (implementation + evaluation + deployment interfaces)

**Team**

| Name | Matric (as in README) |
|------|------------------------|
| Ogundana Moyinolawa | RUN/IFT/22/13194 |
| Adebayo Oluwatimilehin | RUN/IFT/22/13157 |
| Okorie Samuel | RUN/CMP/22/12967 |
| Adewale Olukolade | RUN/CMP/22/13162 |

**Supervisor:** Dr. T. O. Ojewumi  

**Repository (referenced in deployment config):** `https://github.com/TechLegion/Heart-disease-prediction-system/` (see `railway.toml`).

**Document purpose:** This file is the **single source of truth** for what has been built, how it works, how to run it, and what artifacts it produces. A teammate can add their own sections (e.g. literature review, user study, viva Q&A) and **paste this into Microsoft Word** for the formal project report (headings map cleanly to report chapters).  

**Code version note:** The codebase is referred to in comments as **v2** (see `main.py` docstring). An older technical narrative in `SYSTEM_EXPLANATION.md` describes an earlier design (e.g. weight-vector PSO). **Trust this document and the source files in `src/`** for the current behavior.

---

## Table of contents

1. [Executive summary](#1-executive-summary)  
2. [Problem and objectives](#2-problem-and-objectives)  
3. [Dataset and data sources](#3-dataset-and-data-sources)  
4. [High-level system architecture](#4-high-level-system-architecture)  
5. [Methodology (as implemented)](#5-methodology-as-implemented)  
6. [Project structure and key files](#6-project-structure-and-key-files)  
7. [Environment setup and dependencies](#7-environment-setup-and-dependencies)  
8. [How to run the full pipeline](#8-how-to-run-the-full-pipeline)  
9. [Outputs, models, and results files](#9-outputs-models-and-results-files)  
10. [Web user interface (Streamlit)](#10-web-user-interface-streamlit)  
11. [REST API (FastAPI)](#11-rest-api-fastapi)  
12. [Cloud deployment (Railway)](#12-cloud-deployment-railway)  
13. [Evaluation strategy](#13-evaluation-strategy)  
14. [Version control history (summary)](#14-version-control-history-summary)  
15. [Disclaimers and limitations](#15-disclaimers-and-limitations)  
16. [For your partner: extending the work and building the Word report](#16-for-your-partner-extending-the-work-and-building-the-word-report)  
17. [References and further reading](#17-references-and-further-reading)

---

## 1. Executive summary

This project implements a **binary heart disease risk classifier** from standard UCI-style clinical features. The core model is a **scikit-learn Multi-Layer Perceptron (MLP)**. Performance is improved using **bio-inspired optimisers**:

- A **Genetic Algorithm (GA)**, implemented with **DEAP**, searches over **network architecture and training hyperparameters** (learning rate, hidden sizes, L2 regularisation, activation). Fitness is **k-fold cross-validation accuracy** on the training data.
- **Particle Swarm Optimization (PSO)**, implemented with **PySwarms**, refines **learning rate, L2, and momentum** for a **fixed** architecture, again using **CV fitness**.
- A **hybrid** pipeline runs GA first, keeps the **top-K** distinct candidates, runs PSO for **each** candidate, and selects the best CV score before a final retrain.
- A **seed-averaging ensemble** retrains the winning hybrid configuration with multiple random seeds and **averages predicted probabilities** at inference time.

The system includes:

- A batch **experiment runner** (`main.py`) that trains all models, saves metrics, plots, cross-validation summaries, and serialised models.
- A **Streamlit** app (`app.py`) for interactive prediction and result visualisation.
- A **FastAPI** backend (`api.py`) that exposes the same logic as JSON for a **React** or any HTTP client, with OpenAPI/Swagger at `/docs`.

**Important:** The tool is for **research and education**; it is **not** a medical device and must not replace professional diagnosis.

---

## 2. Problem and objectives

**Problem:** Heart disease remains a major cause of morbidity and mortality. Machine learning can assist risk stratification from routine clinical and exercise-test variables, but model quality depends on data preparation and hyperparameter choice.

**Objectives (as reflected in the implementation):**

1. Ingest and preprocess the **UCI Heart Disease** data (optionally **combining** multiple site files for more samples).  
2. Establish a **reproducible baseline ANN** and compare it to **GA-optimised**, **PSO-optimised**, and **hybrid GA–PSO** variants.  
3. Optionally compare against **XGBoost** as a strong non-neural **reference** (if the package is installed).  
4. Report **standard classification metrics** (accuracy, precision, recall, F1, ROC AUC) on a held-out test set and via **5-fold stratified cross-validation**.  
5. Deliver **usable interfaces**: Streamlit for demos, FastAPI for integration, and optional **cloud deployment** via Railway.

---

## 3. Dataset and data sources

### 3.1 Origin

Data follow the **UCI Heart Disease** schema: 14 attributes (13 inputs + target), including demographics, blood pressure, cholesterol, ECG-related measures, exercise stress test results, and angiographic findings.

### 3.2 How the project loads data

`HeartDiseasePreprocessor` in `src/data_preprocessing.py` supports:

1. **Local files** — The experiment runner in `main.py` searches directories `heart disease/`, `heart+disease/`, and `data/` for:

   - `processed.cleveland.data`  
   - `processed.hungarian.data`  
   - `processed.switzerland.data`  
   - `processed.va.data`  

   If **two or more** of these are found, **all are combined** into one dataset (larger *N*, more realistic for ML).

2. **Single file** — If only one matching file exists, that file is used alone.

3. **Automatic fetch** — If no local file is found, the pipeline can fetch via `ucimlrepo` (dataset id **45**), with failure surfaced as a clear error.

### 3.3 Target definition

The raw UCI target is multi-class (0 = no disease, 1–4 = disease presence/severity). The pipeline converts this to **binary** classification: **0 = no disease**, **1 = any disease** (any original label **greater than 0**).

### 3.4 Preprocessing highlights (v2)

The module docstring in `data_preprocessing.py` lists the main design decisions:

- **Consistent raw value normalisation** (e.g. UCI 1-based vs 0-based codes for `cp` and `slope`) so **training data and the Streamlit/API forms stay aligned**.
- **One-hot encoding** for genuinely nominal fields: `cp`, `restecg`, `slope`, `thal` (instead of label encoding that imposes a false order).
- **KNN imputation** for missing values (important for the sparser Hungarian / Switzerland / VA subsets).
- **StandardScaler** (Z-score) on continuous features after the feature matrix is built.
- **Stratified train/test split** (80% / 20%, `random_state=42` in the main pipeline) to preserve class balance.

A dedicated method **`transform_patient(raw_dict)`** encodes a **single** patient dict from the UI/API through the **same** logic as batch training, avoiding train/inference skew.

---

## 4. High-level system architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Data: UCI files (1–4 sites) or ucimlrepo / fallback           │
└───────────────────────────────┬─────────────────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  HeartDiseasePreprocessor: clean → impute → encode → scale     │
│  → X_train, X_test, y_train, y_test (+ feature name list)      │
└───────────────────────────────┬─────────────────────────────────┘
                                ▼
         Train further split: 80/20 of the 80% → train + validation
         (used conceptually; optimisers use CV on training matrix)
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  Models: Baseline ANN | GA-ANN | PSO-ANN | Hybrid | (XGBoost)  │
│          Seed ensemble (Hybrid params)                          │
└───────────────────────────────┬─────────────────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  Evaluation: test-set metrics, ROC, CV; save CSV, plots, joblib│
└───────────────────────────────┬─────────────────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  Consumption: Streamlit (app.py) | FastAPI (api.py) | Railway  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. Methodology (as implemented)

### 5.1 Baseline ANN

Implemented in `src/baseline_ann.py` as a thin wrapper over **`sklearn.neural_network.MLPClassifier`**. Default structure used in `main.py` is **hidden layers (100, 50)**, ReLU-style activations (configurable), **Adam**-style training via scikit-learn’s solver, with **early stopping** where appropriate for standalone training.

The baseline is trained on the **training** portion and evaluated on the **held-out test** set like all other models.

### 5.2 Genetic Algorithm (GA)

Implemented in `src/ga_optimizer.py`.

- **Chromosome:** **5** genes in **[0, 1]** decoded to:
  - learning rate (log scale),
  - first hidden size,
  - second hidden size,
  - L2 `alpha` (log scale),
  - activation index (`relu` / `tanh` / `logistic`).

- **Fitness:** **mean k-fold cross-validation accuracy** (default **3** folds) on the training data, not a single validation split. This reduces variance in hyperparameter search.

- **DEAP** handles selection, crossover, mutation; **elitism** and **top-K** extraction support the hybrid stage (`get_topk_params`).

- After optimisation, the best ANN is built via `build_ann_from_params` and retrained on the full training set for test evaluation in `main.py`.

### 5.3 Particle Swarm Optimization (PSO)

Implemented in `src/pso_optimizer.py`.

- **Particle position:** **3** dimensions in **[0, 1]**, decoded to:
  - learning rate (log scale),
  - L2 `alpha` (log scale),
  - **momentum** (for the solver’s `momentum` parameter where used).

- **Architecture** (hidden sizes and activation) is **fixed** via `base_params` so the hybrid can run PSO **per GA candidate architecture**.

- **Fitness:** same **CV** philosophy as GA (minimise **negative** mean CV accuracy in PySwarms).

- In `main.py`, standalone PSO uses the **initial baseline ANN**’s implied architecture as the `base_params` default path (see `PSOOptimizer.optimize(initial_ann=baseline)` in the call pattern — the optimiser is constructed to inherit the baseline structure).

### 5.4 Hybrid GA–PSO–ANN

Implemented in `src/hybrid_model.py` as `HybridGAPSOANN`.

1. **Stage 1 — GA:** full GA run; retrieve **top-K** parameter sets (distinct architectures / settings).  
2. **Stage 2 — PSO:** for **each** of the K candidates, run PSO with `base_params` set to that candidate’s architecture and activation.  
3. **Stage 3:** pick the configuration with the **best CV** score, then **train the final** `BaselineANN` on the full training data for production evaluation.

The class exposes **`best_params`** for the downstream **seed ensemble**.

### 5.5 Seed-averaging ensemble

Implemented in `src/ensemble_model.py` as `SeedEnsembleANN`.

- Takes the **hybrid’s `best_params`** (learning rate, alpha, momentum, hidden sizes, activation).  
- Trains **N** (default **5**) `BaselineANN` models with **seeds** `base_seed`, `base_seed+1`, …  
- **Inference:** `predict_proba` is the **mean** of all models’ probabilities; `predict` is the argmax of that mean.

This reduces **seed sensitivity** and often stabilises metrics on small tabular data.

### 5.6 XGBoost (optional)

`src/xgboost_baseline.py` provides a tree-based **ceiling** reference. It is only included in runs if `import` succeeds; `requirements.txt` lists it as a **commented optional** dependency (large wheel on some Windows networks). If missing, the pipeline **skips** XGBoost and prints a warning from `main.py`.

### 5.7 Evaluation

On the **same** test set for all models:

- **Accuracy, Precision, Recall, F1-Score, ROC AUC** (see `main.py` and `baseline_ann.evaluate` / ensemble `evaluate`).  
- **Confusion matrix** and **ROC curve** data stored for plotting.  
- **5-fold stratified cross-validation** over the **full** feature matrix (train+test rows concatenated) in `cross_validate_models` — note CV uses **lighter** GA/PSO settings than the main run for **runtime** reasons.

---

## 6. Project structure and key files

| Path | Role |
|------|------|
| `main.py` | End-to-end experiment: preprocess → compare models → plots → CV → save `models/` and `results/`. |
| `app.py` | Streamlit multi-page UI: Predict, Model Comparison, About. |
| `api.py` | FastAPI REST API + OpenAPI docs. |
| `src/data_preprocessing.py` | `HeartDiseasePreprocessor`, loading, encoding, imputation, scaling, `transform_patient`. |
| `src/baseline_ann.py` | `MLPClassifier` wrapper: train, evaluate, metrics, ROC helpers. |
| `src/ga_optimizer.py` | DEAP-based GA, CV fitness, top-K for hybrid. |
| `src/pso_optimizer.py` | PySwarms PSO, CV fitness, `base_params` for fixed architecture. |
| `src/hybrid_model.py` | Top-K GA → per-candidate PSO → final model. |
| `src/ensemble_model.py` | Seed-averaging ensemble. |
| `src/xgboost_baseline.py` | Optional XGBoost baseline. |
| `src/save_model.py` | Standalone or auxiliary saving utility (if used). |
| `heart+disease/` | UCI-format `.data` files and documentation fragments. |
| `models/` | Saved `joblib` models and `preprocessor.joblib` (regenerate with `main.py`). |
| `results/` | `comparison_results.csv`, `cv_results.csv`, `roc_data.joblib`, plots (`comparison.png`, `roc_curves.png` when generated). |
| `requirements.txt` | Python dependencies. |
| `Procfile` | Heroku-style process definition (`uvicorn` on `$PORT`). |
| `railway.toml` | Railway: start command, health check on `/api/health`. |
| `README.md` | Quick-start and overview. |
| `PROPOSAL_DEFENCE_METHODOLOGY.md` | Defence-oriented methodology narrative (align wording with v2 where code evolved). |
| `SYSTEM_EXPLANATION.md` | Older deep-dive; cross-check against current `src/` if citing implementation details. |
| `test.py` | Minimal `ucimlrepo` fetch demo script. |

---

## 7. Environment setup and dependencies

**Prerequisites:** **Python 3.10+** (as stated in `README.md`).

```text
# From project root
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**Core packages** (from `requirements.txt`):

- `numpy`, `pandas`  
- `scikit-learn`  
- `deap` (GA)  
- `pyswarms` (PSO)  
- `ucimlrepo` (optional automatic dataset fetch)  
- `streamlit`, `plotly`  
- `joblib`  
- `matplotlib`, `seaborn`  
- `fastapi`, `uvicorn[standard]`  

**Optional:** `xgboost` (uncomment or install manually for the tree baseline).

---

## 8. How to run the full pipeline

### 8.1 Train everything and generate artifacts

```text
python main.py
```

**Expect long runtime** (tens of minutes to over an hour on a laptop) because GA, PSO, and hybrid each perform many **inner** ANN training runs under cross-validation.

**UTF-8:** `main.py` attempts to reconfigure **stdout/stderr to UTF-8** on Windows so special characters in logs do not break.

### 8.2 Launch the Streamlit demo

```text
streamlit run app.py
```

Requires `models/preprocessor.joblib` and the selected model `*.joblib` files from **Step 8.1**.

### 8.3 Launch the FastAPI server (local)

```text
uvicorn api:app --reload --port 8000
```

- Interactive API docs: `http://localhost:8000/docs`  
- Health check: `GET /api/health`  

### 8.4 Smoke test: UCI fetch

`test.py` is a **minimal** script showing `fetch_ucirepo(id=45)`; it is not part of the main training path.

---

## 9. Outputs, models, and results files

After a successful `main.py` run:

| Output | Description |
|--------|-------------|
| `results/comparison_results.csv` | Per-model test metrics (single split). |
| `results/cv_results.csv` | 5-fold CV means and std devs. |
| `results/roc_data.joblib` | Per-model FPR/TPR/AUC for ROC plots and API. |
| `results/comparison.png` | Bar chart of metrics. |
| `results/roc_curves.png` | ROC plot. |
| `models/baseline_ann.joblib` | Serialised `MLPClassifier` (baseline). |
| `models/ga_ann.joblib` | Best GA model (sklearn object). |
| `models/pso_ann.joblib` | Best PSO model. |
| `models/hybrid_ann.joblib` | Final hybrid MLP. |
| `models/xgboost.joblib` | If XGBoost ran. |
| `models/hybrid_ensemble.joblib` | Full `SeedEnsembleANN` object (for averaging). |
| `models/preprocessor.joblib` | Fitted `HeartDiseasePreprocessor`. |
| `models/feature_names.joblib` | Ordered feature names after encoding (API exposes `features_used`). |
| `models/ann_model.joblib` | Legacy alias copy of baseline (see `save_all_models` in `main.py`). |

**Note:** `.gitignore` may exclude some generated binaries; the **Railway** deployment and README assume committed models/ROC for a zero-setup API — your team’s policy on committing large `joblib` files should be **explicit in the report** (reproducibility vs. repository size).

---

## 10. Web user interface (Streamlit)

**File:** `app.py`  

**Pages**

1. **Predict Heart Disease** — Form for all clinical fields (with help text and sample patients). Runs `preprocessor.transform_patient` then `predict` / `predict_proba` on the chosen model. Includes simple **“risk factor”** flags when numeric values fall outside configured normal ranges.  
2. **Model Comparison** — Loads `comparison_results.csv`, bar chart, ROC from `roc_data.joblib`, CV table from `cv_results.csv`.  
3. **About / Methodology** — High-level description aligned with the v2 pipeline.

**Important:** The UI uses **human-readable** labels for categoricals (e.g. `"0 - Typical Angina"`). The preprocessor maps these to the same encoding as training.

---

## 11. REST API (FastAPI)

**File:** `api.py`  
**Version field in app:** 2.0.0  

**CORS** is set permissive (`allow_origins=["*"]`) for local development; **tighten** for production.

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/` | API metadata and endpoint list. |
| GET | `/api/health` | Status, which models exist, whether preprocessor/results are present. |
| GET | `/api/models` | List models + descriptions. |
| GET | `/api/features` | Field help, allowed option strings, normal ranges, slider defaults. |
| GET | `/api/samples` | Built-in example patients. |
| POST | `/api/predict` | JSON body: `model_name` + `patient` object → prediction + probabilities + risk flags + `features_used`. |
| GET | `/api/comparison` | Test-set metrics table + best model. |
| GET | `/api/roc` | Serialised ROC curve data. |
| GET | `/api/cv` | Cross-validation table. |
| GET | `/docs` | Swagger UI. |
| GET | `/redoc` | ReDoc. |
| GET | `/openapi.json` | OpenAPI schema. |

Use **`/api/predict`** for React or mobile clients: the request schema is defined by Pydantic models `PredictionRequest` and `PatientInput`.

---

## 12. Cloud deployment (Railway)

**Files:** `railway.toml`, `Procfile`

- **Start command:** `uvicorn api:app --host 0.0.0.0 --port $PORT`  
- **Health check path:** `/api/health`  
- **Timeout:** 300 s (as configured) for slow cold starts if applicable.

**Academic report tip:** Document the **public URL** (if assigned), **uptime** expectations, and that **sensitive patient data** must never be sent to a demo instance without governance approval.

---

## 13. Evaluation strategy

1. **Single hold-out (20%):** fair comparison because every model sees the same split.  
2. **Stratification:** reduces misleading accuracy when classes are imbalanced.  
3. **K-fold CV inside GA/PSO:** reduces overfitting to one validation draw.  
4. **5-fold CV report:** reports **mean ± std** for robustness; inner CV uses **reduced** population/generations for speed — **mention this** in the thesis if reviewers ask why CV block differs from the main run.  
5. **ROC AUC** complements threshold-based metrics when calibration matters.

---

## 14. Version control history (summary)

Recent commits in this repository (subject to `git log`):

| Commit (short) | Message (paraphrased) |
|----------------|------------------------|
| `eaf17dc` | Railway deployment (uvicorn, models, roc data for API) |
| `ad1f347` | API: explicit Swagger at `/docs` and OpenAPI |
| `e97ed8e` | v2: CV-based GA/PSO, top-K hybrid, one-hot features, ensemble |
| `b1143ae` | FastAPI backend and Streamlit enhancements |
| `65178db` | First commit |

Use this table in the “development timeline” section of the Word report if needed.

---

## 15. Disclaimers and limitations

1. **Not medical advice** — The Streamlit and API UIs state the system is for **educational/research** use. A positive or negative prediction must **not** be treated as a diagnosis.  
2. **Dataset shift** — Models are trained on **UCI** populations; real hospitals differ in **demographics, equipment, and lab assays**.  
3. **Class imbalance and small *N*** — Even with four sites combined, sample size is **modest** for deep search; metrics **vary** with seed and split. The ensemble and CV reporting partly mitigate this.  
4. **Fairness** — The dataset encodes **sex** as a feature; any deployment requires **ethical and fairness** review.  
5. **Documentation drift** — Always prefer **`src/*.py` and this file** over older markdown if there is a conflict.

---

## 16. For your partner: extending the work and building the Word report

### 16.1 Suggested Word report structure (map from this project)

1. **Cover, declaration, abstract** — You write.  
2. **Introduction & problem statement** — Use Sections **1–2** here; expand with **literature** (papers on heart disease ML, GA/PSO for NNs, UCI heart disease).  
3. **Related work** — Your own survey.  
4. **Methodology** — **Section 5** of this document + equations for GA/PSO and MLP (from textbooks or papers). Reconcile wording with `PROPOSAL_DEFENCE_METHODOLOGY.md` if the department expects proposal alignment.  
5. **System design** — **Section 4** + **Section 6**; add **UML** or data-flow diagrams in Word.  
6. **Implementation** — **Sections 6–12**; include screenshots of Streamlit, Swagger `/docs`, and (if live) the Railway health JSON.  
7. **Results & discussion** — Paste tables from `comparison_results.csv` and `cv_results.csv`, embed `comparison.png` and `roc_curves.png`, discuss **overfitting**, **variance**, and **why hybrid might beat/lose** baseline.  
8. **Conclusion & future work** — e.g. SHAP for explainability, calibration curves, external validation, mobile app, clinician UX study.  
9. **References** — See **Section 17** below.  
10. **Appendix** — API endpoint list, full hyperparameter tables, user manual.

### 16.2 Converting this Markdown to Word

- Open **Microsoft Word** → **Open** this `.md` file (Word 2016+ can import Markdown) **or** paste into Word and apply **Heading 1/2/3** styles to lines starting with `#`, `##`, `###`.  
- Fix **table** and **code block** formatting (Word: Insert → Table; Courier New for code).  
- For **Mermaid/ASCII** diagrams, redraw using Word **SmartArt** or **Insert Shapes** for a cleaner thesis look.  
- Update the **author block** and **matric numbers** if anything changed at the department.

### 16.3 If you add features (technical checklist)

- **Retrain** with `main.py` after any change to `HeartDiseasePreprocessor` so `preprocessor.joblib` matches.  
- **Add tests** (even simple `pytest` assertions) for `transform_patient` with edge cases.  
- **Log** `random_state` and `requirements.txt` versions in the report appendix for **reproducibility**.  
- If you add **new endpoints**, update **`api.py`** and mention them in the report; keep **`/api/health`** accurate.

---

## 17. References and further reading

1. UCI Machine Learning Repository — *Heart Disease* Data Set (document the exact URL and access date in the thesis).  
2. scikit-learn documentation — `MLPClassifier`, metrics, `StratifiedKFold`, pipelines.  
3. DEAP documentation — genetic algorithms in Python.  
4. PySwarms documentation — PSO variants.  
5. Original literature on **GA/NN** and **PSO/NN** hybridisation (add peer-reviewed papers as required by your department).

---

*End of **PROJECT_IMPLEMENTATION_DOCUMENTATION.md**. Update this file when the codebase changes so the team and supervisor always have a single up-to-date narrative.*
