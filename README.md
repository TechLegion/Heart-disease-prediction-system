# Heart Disease Prediction Using Hybrid GA-PSO Optimized ANN

An improved heart disease prediction system that combines **Genetic Algorithm (GA)** and **Particle Swarm Optimization (PSO)** to optimize an **Artificial Neural Network (ANN)**. Developed as a final year project at Redeemer's University, Ede.

## Overview

| Component | Detail |
|---|---|
| **Dataset** | UCI Heart Disease — Cleveland, Hungarian, Switzerland & VA Long Beach (combined ~920 samples) |
| **Baseline** | Multi-layer Perceptron (scikit-learn `MLPClassifier`) |
| **GA Optimization** | Evolves hidden-layer architecture and learning rate using DEAP |
| **PSO Optimization** | Fine-tunes learning rate and L2 regularization using PySwarms |
| **Hybrid GA-PSO-ANN** | Sequential pipeline — GA selects architecture, PSO refines training params |
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
│   ├── hybrid_model.py          # Hybrid GA → PSO → ANN pipeline
│   └── save_model.py            # Standalone model saving utility
├── models/                      # Saved trained models (generated)
├── results/                     # Metrics CSVs, plots, ROC data (generated)
├── app.py                       # Streamlit web application
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
2. Train and evaluate four models — Baseline ANN, GA-ANN, PSO-ANN, Hybrid GA-PSO-ANN
3. Run 5-fold stratified cross-validation
4. Save trained models to `models/` and results to `results/`

> **Note**: Training takes several minutes due to the optimization search.

### Launch the Web App

```bash
streamlit run app.py
```

The app has three pages:
- **Predict Heart Disease** — enter patient data, choose a model, get a prediction with confidence
- **Model Comparison** — side-by-side metrics, interactive ROC curves, cross-validation summary
- **About / Methodology** — detailed explanation of the research approach

> You must run `python main.py` first to generate the trained models and results files.

## Methodology

1. **Data Preprocessing**: Missing-value imputation, duplicate removal, label encoding, Z-score normalization, target binarization (heart disease present / absent), 80/20 train-test split.

2. **Baseline ANN**: 2-layer MLP (100, 50 neurons), ReLU activation, Adam optimizer, early stopping.

3. **GA Optimization (DEAP)**: Evolves a population of chromosomes encoding `learning_rate_init`, `hidden_layer_1` size, and `hidden_layer_2` size. Fitness = validation accuracy. Uses tournament selection, two-point crossover, and Gaussian mutation.

4. **PSO Optimization (PySwarms)**: Particles encode `learning_rate_init` and L2 regularization strength (`alpha`). Initialized near the GA-found learning rate. Minimizes negative validation accuracy.

5. **Hybrid GA-PSO-ANN**: GA finds the best architecture → PSO fine-tunes the training parameters for that architecture → final model is trained with all optimized hyperparameters.

6. **Evaluation**: All models assessed on the same held-out test set using Accuracy, Precision, Recall, F1, and ROC AUC. Robustness confirmed with 5-fold stratified cross-validation.

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
