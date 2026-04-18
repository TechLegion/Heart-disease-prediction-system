"""
Main Script for the Heart Disease Prediction System (v2).

Pipeline:
  1. Load + preprocess (one-hot encoded features, KNN imputation).
  2. Train & compare: Baseline ANN, GA-ANN, PSO-ANN, Hybrid GA-PSO-ANN,
     XGBoost baseline, and a Seed-Averaging Ensemble of the best Hybrid.
  3. Save single-split metrics + ROC data.
  4. Run 5-fold stratified cross-validation for every model.
  5. Save every trained model plus the fitted preprocessor for the UI / API.
"""

from __future__ import annotations

import os
import sys
import joblib

# Force UTF-8 on Windows consoles so Greek letters / arrows print cleanly.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
)

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from data_preprocessing import HeartDiseasePreprocessor
from baseline_ann import BaselineANN
from ga_optimizer import GAOptimizer, build_ann_from_params
from pso_optimizer import PSOOptimizer
from hybrid_model import HybridGAPSOANN
from ensemble_model import SeedEnsembleANN

try:
    from xgboost_baseline import XGBoostBaseline  # type: ignore
    _HAS_XGBOOST = True
except Exception as _e:
    print(f"[WARN] XGBoost baseline unavailable: {_e}")
    _HAS_XGBOOST = False
    XGBoostBaseline = None  # type: ignore


# ====================================================================
# Helpers
# ====================================================================
def _row(name: str, m: dict) -> dict:
    return {
        "Model": name,
        "Accuracy":  m["accuracy"],
        "Precision": m["precision"],
        "Recall":    m["recall"],
        "F1-Score":  m["f1_score"],
        "AUC":       m["auc"],
    }


def _zero_row(name: str) -> dict:
    return {"Model": name, "Accuracy": 0.0, "Precision": 0.0,
            "Recall": 0.0, "F1-Score": 0.0, "AUC": 0.0}


# ====================================================================
# Single-split model comparison
# ====================================================================
def compare_models(X_train, y_train, X_val, y_val, X_test, y_test,
                   random_state: int = 42, verbose: bool = True):
    """Train all models, evaluate on X_test, return dataframes + trained models."""
    results: list[dict] = []
    roc_data: dict = {}
    trained_models: dict = {}

    print("\n" + "=" * 80 + "\nMODEL COMPARISON EXPERIMENT\n" + "=" * 80)

    # 1. Baseline ANN -------------------------------------------------
    print("\n[1/6] Training Baseline ANN Model...")
    baseline = BaselineANN(hidden_layer_sizes=(100, 50), max_iter=500,
                           random_state=random_state)
    baseline.train(X_train, y_train, verbose=verbose)
    m = baseline.evaluate(X_test, y_test, verbose=verbose,
                          model_name="BASELINE ANN")
    results.append(_row("Baseline ANN", m))
    roc_data["Baseline ANN"] = (m["fpr"], m["tpr"], m["auc"])
    trained_models["Baseline ANN"] = baseline

    # 2. GA-Optimized ANN --------------------------------------------
    print("\n[2/6] Training GA-Optimised ANN Model...")
    try:
        ga = GAOptimizer(X_train, y_train,
                         n_population=30, n_generations=20, cv_folds=3,
                         eval_max_iter=200, random_state=random_state)
        ga.optimize(verbose=verbose)
        ga_ann = ga.get_optimized_ann()
        ga_ann.train(X_train, y_train, verbose=False)
        m = ga_ann.evaluate(X_test, y_test, verbose=verbose,
                            model_name="GA-OPTIMISED ANN")
        results.append(_row("GA-Optimized ANN", m))
        roc_data["GA-Optimized ANN"] = (m["fpr"], m["tpr"], m["auc"])
        trained_models["GA-Optimized ANN"] = ga_ann
    except Exception as e:
        print(f"GA optimisation failed: {e}")
        results.append(_zero_row("GA-Optimized ANN"))

    # 3. PSO-Optimized ANN -------------------------------------------
    print("\n[3/6] Training PSO-Optimised ANN Model...")
    try:
        pso = PSOOptimizer(X_train, y_train,
                           n_particles=20, iterations=25, cv_folds=3,
                           eval_max_iter=200, random_state=random_state)
        pso.optimize(initial_ann=baseline, verbose=verbose)
        pso_ann = pso.get_optimized_ann()
        pso_ann.train(X_train, y_train, verbose=False)
        m = pso_ann.evaluate(X_test, y_test, verbose=verbose,
                             model_name="PSO-OPTIMISED ANN")
        results.append(_row("PSO-Optimized ANN", m))
        roc_data["PSO-Optimized ANN"] = (m["fpr"], m["tpr"], m["auc"])
        trained_models["PSO-Optimized ANN"] = pso_ann
    except Exception as e:
        print(f"PSO optimisation failed: {e}")
        results.append(_zero_row("PSO-Optimized ANN"))

    # 4. Hybrid GA-PSO-ANN -------------------------------------------
    print("\n[4/6] Training Hybrid GA-PSO-ANN Model...")
    hybrid_params: dict | None = None
    try:
        hybrid = HybridGAPSOANN(
            top_k=3,
            ga_params={"n_population": 30, "n_generations": 20, "cv_folds": 3},
            pso_params={"n_particles": 15, "iterations": 20, "cv_folds": 3},
            random_state=random_state,
        )
        hybrid.train(X_train, y_train, verbose=verbose)
        m = hybrid.evaluate(X_test, y_test, verbose=verbose)
        results.append(_row("Hybrid GA-PSO-ANN", m))
        roc_data["Hybrid GA-PSO-ANN"] = (m["fpr"], m["tpr"], m["auc"])
        trained_models["Hybrid GA-PSO-ANN"] = hybrid
        hybrid_params = hybrid.best_params
    except Exception as e:
        print(f"Hybrid training failed: {e}")
        results.append(_zero_row("Hybrid GA-PSO-ANN"))

    # 5. XGBoost (ceiling reference) ---------------------------------
    if _HAS_XGBOOST:
        print("\n[5/6] Training XGBoost Baseline...")
        try:
            xgb = XGBoostBaseline(random_state=random_state)
            xgb.train(X_train, y_train, verbose=verbose)
            m = xgb.evaluate(X_test, y_test, verbose=verbose,
                             model_name="XGBOOST BASELINE")
            results.append(_row("XGBoost", m))
            roc_data["XGBoost"] = (m["fpr"], m["tpr"], m["auc"])
            trained_models["XGBoost"] = xgb
        except Exception as e:
            print(f"XGBoost failed: {e}")
    else:
        print("\n[5/6] Skipping XGBoost (package not installed).")

    # 6. Seed-averaging ensemble of the best Hybrid config -----------
    print("\n[6/6] Training Seed Ensemble (5× Hybrid config)...")
    if hybrid_params:
        try:
            ens = SeedEnsembleANN(params=hybrid_params, n_models=5,
                                  base_seed=random_state)
            ens.train(X_train, y_train, verbose=verbose)
            m = ens.evaluate(X_test, y_test, verbose=verbose,
                             model_name="HYBRID SEED ENSEMBLE")
            results.append(_row("Hybrid Ensemble", m))
            roc_data["Hybrid Ensemble"] = (m["fpr"], m["tpr"], m["auc"])
            trained_models["Hybrid Ensemble"] = ens
        except Exception as e:
            print(f"Ensemble training failed: {e}")
            results.append(_zero_row("Hybrid Ensemble"))
    else:
        print("Skipping ensemble (no hybrid params available).")
        results.append(_zero_row("Hybrid Ensemble"))

    df = pd.DataFrame(results)
    print("\n" + "=" * 80 + "\nFINAL COMPARISON RESULTS\n" + "=" * 80)
    print(df.to_string(index=False))
    best_idx = df["Accuracy"].idxmax()
    print(f"\nBest Model : {df.loc[best_idx, 'Model']}")
    print(f"Accuracy   : {df.loc[best_idx, 'Accuracy']:.4f}")
    print(f"AUC        : {df.loc[best_idx, 'AUC']:.4f}")
    print("=" * 80)
    return df, roc_data, trained_models


# ====================================================================
# Cross-validation
# ====================================================================
def cross_validate_models(X, y, n_folds: int = 5, random_state: int = 42):
    print("\n" + "=" * 80)
    print(f"{n_folds}-FOLD CROSS-VALIDATION")
    print("=" * 80)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True,
                          random_state=random_state)
    model_names = ["Baseline ANN", "GA-Optimized ANN", "PSO-Optimized ANN",
                   "Hybrid GA-PSO-ANN"]
    if _HAS_XGBOOST:
        model_names.append("XGBoost")
    cv_scores = {n: {k: [] for k in ["Accuracy", "Precision", "Recall",
                                      "F1-Score", "AUC"]}
                 for n in model_names}

    def _append(name, m):
        cv_scores[name]["Accuracy"].append(m["accuracy"])
        cv_scores[name]["Precision"].append(m["precision"])
        cv_scores[name]["Recall"].append(m["recall"])
        cv_scores[name]["F1-Score"].append(m["f1_score"])
        cv_scores[name]["AUC"].append(m["auc"])

    def _append_zero(name):
        for k in cv_scores[name]:
            cv_scores[name][k].append(0.0)

    X = X.reset_index(drop=True) if hasattr(X, "reset_index") else pd.DataFrame(X)
    y = y.reset_index(drop=True) if hasattr(y, "reset_index") else pd.Series(y)

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n--- Fold {fold}/{n_folds} ---")
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]

        try:
            ann = BaselineANN(hidden_layer_sizes=(100, 50), max_iter=400,
                              random_state=random_state)
            ann.train(X_tr, y_tr, verbose=False)
            _append("Baseline ANN", ann.evaluate(X_te, y_te, verbose=False))
        except Exception:
            _append_zero("Baseline ANN")

        try:
            ga = GAOptimizer(X_tr, y_tr, n_population=15, n_generations=10,
                             cv_folds=3, eval_max_iter=150,
                             random_state=random_state)
            ga.optimize(verbose=False)
            ga_ann = ga.get_optimized_ann()
            ga_ann.train(X_tr, y_tr, verbose=False)
            _append("GA-Optimized ANN", ga_ann.evaluate(X_te, y_te, verbose=False))
        except Exception:
            _append_zero("GA-Optimized ANN")

        try:
            pso = PSOOptimizer(X_tr, y_tr, n_particles=12, iterations=12,
                               cv_folds=3, eval_max_iter=150,
                               random_state=random_state)
            pso.optimize(initial_ann=ann, verbose=False)
            pso_ann = pso.get_optimized_ann()
            pso_ann.train(X_tr, y_tr, verbose=False)
            _append("PSO-Optimized ANN", pso_ann.evaluate(X_te, y_te, verbose=False))
        except Exception:
            _append_zero("PSO-Optimized ANN")

        try:
            hyb = HybridGAPSOANN(
                top_k=2,
                ga_params={"n_population": 15, "n_generations": 10, "cv_folds": 3},
                pso_params={"n_particles": 10, "iterations": 10, "cv_folds": 3},
                random_state=random_state)
            hyb.train(X_tr, y_tr, verbose=False)
            _append("Hybrid GA-PSO-ANN", hyb.evaluate(X_te, y_te, verbose=False))
        except Exception:
            _append_zero("Hybrid GA-PSO-ANN")

        if _HAS_XGBOOST:
            try:
                xgb = XGBoostBaseline(random_state=random_state)
                xgb.train(X_tr, y_tr, verbose=False)
                _append("XGBoost", xgb.evaluate(X_te, y_te, verbose=False))
            except Exception:
                _append_zero("XGBoost")

    rows = []
    for name in model_names:
        row = {"Model": name}
        for metric in ["Accuracy", "Precision", "Recall", "F1-Score", "AUC"]:
            vals = cv_scores[name][metric]
            row[f"{metric}_mean"] = float(np.mean(vals))
            row[f"{metric}_std"]  = float(np.std(vals))
        rows.append(row)
    cv_df = pd.DataFrame(rows)

    print("\n" + "=" * 80 + "\nCROSS-VALIDATION RESULTS (mean ± std)\n" + "=" * 80)
    for _, r in cv_df.iterrows():
        print(f"\n{r['Model']}:")
        for m in ["Accuracy", "Precision", "Recall", "F1-Score", "AUC"]:
            print(f"  {m:10s}: {r[f'{m}_mean']:.4f} ± {r[f'{m}_std']:.4f}")
    print("=" * 80)
    return cv_df


# ====================================================================
# Plotting
# ====================================================================
def plot_results(comparison_df: pd.DataFrame, roc_data: dict,
                 save_dir: str = "results"):
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(comparison_df))
    width = 0.15
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC"]
    for i, metric in enumerate(metrics):
        ax.bar(x + i * width, comparison_df[metric], width, label=metric)
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(comparison_df["Model"], rotation=25, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison")
    ax.legend(); ax.grid(axis="y", alpha=0.3); ax.set_ylim([0, 1])
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "comparison.png"), dpi=300)
    print(f"Bar chart saved to {save_dir}/comparison.png")

    fig, ax = plt.subplots(figsize=(8, 6))
    for name, (fpr, tpr, auc) in roc_data.items():
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves"); ax.legend(loc="lower right"); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "roc_curves.png"), dpi=300)
    print(f"ROC curves saved to {save_dir}/roc_curves.png")


# ====================================================================
# Model persistence
# ====================================================================
def save_all_models(trained_models: dict, preprocessor: HeartDiseasePreprocessor,
                    feature_names: list[str]):
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(models_dir, exist_ok=True)

    # Map display name → (filename, way to extract the raw predict model)
    mapping: list[tuple[str, str, str]] = [
        ("Baseline ANN",      "baseline_ann.joblib",     "sklearn"),
        ("GA-Optimized ANN",  "ga_ann.joblib",           "sklearn"),
        ("PSO-Optimized ANN", "pso_ann.joblib",          "sklearn"),
        ("Hybrid GA-PSO-ANN", "hybrid_ann.joblib",       "sklearn"),
        ("XGBoost",           "xgboost.joblib",          "sklearn"),
        ("Hybrid Ensemble",   "hybrid_ensemble.joblib",  "ensemble"),
    ]

    for display, fname, kind in mapping:
        model = trained_models.get(display)
        if model is None:
            continue
        path = os.path.join(models_dir, fname)
        if kind == "ensemble":
            joblib.dump(model, path)
        elif hasattr(model, "final_model"):
            joblib.dump(model.final_model.model, path)
        elif hasattr(model, "model"):
            joblib.dump(model.model, path)
        else:
            joblib.dump(model, path)
        print(f"  Saved {fname}")

    # Legacy alias used elsewhere.
    if "Baseline ANN" in trained_models:
        joblib.dump(trained_models["Baseline ANN"].model,
                    os.path.join(models_dir, "ann_model.joblib"))

    joblib.dump(preprocessor, os.path.join(models_dir, "preprocessor.joblib"))
    joblib.dump(feature_names, os.path.join(models_dir, "feature_names.joblib"))
    print("  Saved preprocessor + feature_names")


# ====================================================================
# Main
# ====================================================================
def main():
    print("=" * 80)
    print("HEART DISEASE PREDICTION SYSTEM — EXPERIMENT RUNNER (v2)")
    print("=" * 80)

    # --- Step 1: Load & preprocess ---
    print("\n[STEP 1] Loading and Preprocessing Data...")
    preprocessor = HeartDiseasePreprocessor()

    search_dirs = ["heart disease", "heart+disease", "data"]
    site_files = [
        "processed.cleveland.data", "processed.hungarian.data",
        "processed.switzerland.data", "processed.va.data",
    ]
    all_paths: list[str] = []
    for d in search_dirs:
        found = [os.path.join(d, f) for f in site_files
                 if os.path.exists(os.path.join(d, f))]
        if found:
            all_paths = found
            break

    if len(all_paths) >= 2:
        print(f"Found {len(all_paths)} dataset files — combining")
        X_tr_full, X_test, y_tr_full, y_test, feature_names = \
            preprocessor.preprocess_pipeline(file_paths=all_paths)
    elif len(all_paths) == 1:
        X_tr_full, X_test, y_tr_full, y_test, feature_names = \
            preprocessor.preprocess_pipeline(file_path=all_paths[0])
    else:
        print("Downloading from UCI...")
        X_tr_full, X_test, y_tr_full, y_test, feature_names = \
            preprocessor.preprocess_pipeline()

    X_train, X_val, y_train, y_val = train_test_split(
        X_tr_full, y_tr_full, test_size=0.2, random_state=42, stratify=y_tr_full)
    print(f"\nTrain: {X_train.shape[0]}  Val: {X_val.shape[0]}  "
          f"Test: {X_test.shape[0]}  Features: {len(feature_names)}")

    # --- Step 2: Single-split comparison ---
    print("\n[STEP 2] Running Model Comparison...")
    comparison_df, roc_data, trained_models = compare_models(
        X_train, y_train, X_val, y_val, X_test, y_test)
    os.makedirs("results", exist_ok=True)
    comparison_df.to_csv("results/comparison_results.csv", index=False)
    joblib.dump(roc_data, "results/roc_data.joblib")

    # --- Step 3: Plots ---
    print("\n[STEP 3] Generating Plots...")
    plot_results(comparison_df, roc_data)

    # --- Step 4: Cross-validation ---
    print("\n[STEP 4] Cross-Validation...")
    X_full = pd.concat([X_tr_full, X_test], ignore_index=True)
    y_full = pd.concat([y_tr_full, y_test], ignore_index=True)
    cv_df = cross_validate_models(X_full, y_full, n_folds=5)
    cv_df.to_csv("results/cv_results.csv", index=False)
    print("CV results saved to results/cv_results.csv")

    # --- Step 5: Save models ---
    print("\n[STEP 5] Saving Models for Web UI / API...")
    save_all_models(trained_models, preprocessor, feature_names)

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE. Next: streamlit run app.py  |  uvicorn api:app")
    print("=" * 80)


if __name__ == "__main__":
    main()
