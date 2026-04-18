# Heart Disease Prediction Using Hybrid GA-PSO Optimized ANN

An improved heart disease prediction system that combines **Genetic Algorithm (GA)** and **Particle Swarm Optimization (PSO)** to optimize an **Artificial Neural Network (ANN)**. Developed as a final year project at Redeemer's University, Ede.

## Overview

| Component | Detail |
|---|---|
| **Dataset** | UCI Heart Disease — Cleveland, Hungarian, Switzerland & VA Long Beach (combined ~920 samples) |
| **Baseline** | Multi-layer Perceptron (scikit-learn `MLPClassifier`) |
| **GA Optimization** | Evolves architecture, learning rate, L2 and activation; fitness = **3-fold CV accuracy** |
| **PSO Optimization** | Fine-tunes learning rate, L2 and **momentum**; fitness = **3-fold CV accuracy** |
| **Hybrid GA-PSO-ANN** | GA returns **top-K** architectures → PSO refines each → best CV wins → final retrain |
| **Seed Ensemble** | 5× retrain of the best Hybrid config with different seeds; **average probabilities** |
| **XGBoost (optional)** | Tree-based ceiling reference — `pip install xgboost` if your network allows the large wheel |
| **Evaluation** | Accuracy, Precision, Recall, F1-Score, ROC AUC, 5-fold Stratified CV |
| **Web UI** | Interactive Streamlit app for prediction, model comparison, and methodology |

## Project Structure

```
├── heart+disease/               # UCI heart disease dataset files
│   ├── processed.cleveland.data
│   ├── processed.hungarian.data
│   ├── processed.switzerland.data
│   └── processed.va.data
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py    # Loading, cleaning, encoding, scaling
│   ├── baseline_ann.py          # BaselineANN wrapper around MLPClassifier
│   ├── ga_optimizer.py          # GA hyperparameter optimizer (DEAP)
│   ├── pso_optimizer.py         # PSO hyperparameter optimizer (PySwarms)
│   ├── hybrid_model.py          # Hybrid GA → top-K → PSO → ANN pipeline
│   ├── ensemble_model.py        # Seed-averaging ensemble
│   ├── xgboost_baseline.py      # Optional XGBoost ceiling reference
│   └── save_model.py            # Standalone model saving utility
├── models/                      # Saved trained models (generated)
├── results/                     # Metrics CSVs, plots, ROC data (generated)
├── app.py                       # Streamlit web application
├── api.py                       # FastAPI JSON backend (for React)
├── main.py                      # Full experiment runner
├── requirements.txt             # Python dependencies
└── README.md
```

## Installation & Setup

**Prerequisites**: Python 3.10+

```bash
# 1. Clone the repository
git clone <repo-url>
cd "Final year Project"

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt
```

## Usage

### Train All Models & Generate Results

```bash
python main.py
```

This will:
1. Load and combine all four UCI heart disease datasets
2. Train and evaluate **Baseline ANN, GA-ANN, PSO-ANN, Hybrid GA-PSO-ANN, Hybrid Seed Ensemble** (and **XGBoost** if installed)
3. Run 5-fold stratified cross-validation
4. Save trained models + the fitted `preprocessor.joblib` to `models/` and metrics to `results/`

> **Note**: Full training can take **30–60+ minutes** on a laptop because every GA/PSO fitness evaluation retrains an ANN under 3-fold CV.

### Launch the Web App

```bash
streamlit run app.py
```

The app has three pages:
- **Predict Heart Disease** — enter patient data, choose a model, get a prediction with confidence
- **Model Comparison** — side-by-side metrics, interactive ROC curves, cross-validation summary
- **About / Methodology** — detailed explanation of the research approach

> You must run `python main.py` first to generate the trained models and results files.

### Launch the JSON API (React-ready)

```bash
uvicorn api:app --reload --port 8000
```

Open `http://localhost:8000/docs` for interactive Swagger documentation.

## Methodology

1. **Data Preprocessing**: KNN imputation, duplicate removal, **one-hot encoding of nominal columns** (`cp`, `restecg`, `slope`, `thal`), Z-score scaling of continuous features, target binarisation, 80/20 stratified train-test split.

2. **Baseline ANN**: 2-layer MLP (100, 50 neurons), ReLU activation, Adam optimizer, early stopping.

3. **GA Optimization (DEAP)**: Evolves `learning_rate`, hidden-layer sizes, L2 (`alpha`) and **activation**. Fitness = **mean 3-fold CV accuracy** on the training fold. Elitism + tournament selection.

4. **PSO Optimization (PySwarms)**: 3-D particle space (`learning_rate`, `alpha`, **momentum**). Fitness = **3-fold CV accuracy**.

5. **Hybrid GA-PSO-ANN**: GA produces **top-K distinct architectures** → each gets its own PSO run → configuration with the best CV score is selected for the final full training.

6. **Seed Ensemble**: The winning Hybrid hyperparameters are retrained 5× with seeds 42–46; inference averages `predict_proba`.

7. **Evaluation**: All models assessed on the same held-out test set using Accuracy, Precision, Recall, F1, and ROC AUC. Robustness confirmed with 5-fold stratified cross-validation.

## Dependencies

| Package | Purpose |
|---|---|
| `scikit-learn` | ANN (MLPClassifier), metrics, preprocessing |
| `deap` | Genetic Algorithm framework |
| `pyswarms` | Particle Swarm Optimization |
| `streamlit` | Interactive web UI |
| `plotly` | Interactive charts (ROC curves, comparison bars) |
| `joblib` | Model serialization |
| `pandas` / `numpy` | Data manipulation |
| `matplotlib` / `seaborn` | Static plots |
| `ucimlrepo` | UCI dataset fallback download |

## Authors

- Ogundana Moyinolawa — RUN/IFT/22/13194
- Adebayo Oluwatimilehin — RUN/IFT/22/13157
- Okorie Samuel — RUN/CMP/22/12967
- Adewale Olukolade — RUN/CMP/22/13162

**Supervisor**: Dr. T. O. Ojewumi

Redeemer's University, Ede, Osun State — 2025/2026
