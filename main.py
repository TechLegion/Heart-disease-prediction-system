"""
Main Script for Heart Disease Prediction System

Pipeline:
  1. Load and preprocess data
  2. Train & compare: Baseline ANN, GA-ANN, PSO-ANN, Hybrid GA-PSO-ANN
  3. Save results (CSV, plots, ROC data)
  4. Run 5-fold cross-validation
  5. Save all trained models for the Streamlit UI
"""

import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_preprocessing import HeartDiseasePreprocessor
from baseline_ann import BaselineANN
from ga_optimizer import GAOptimizer
from pso_optimizer import PSOOptimizer
from hybrid_model import HybridGAPSOANN


# ====================================================================
# Single-split model comparison
# ====================================================================
def compare_models(X_train, y_train, X_val, y_val, X_test, y_test,
                   verbose=True):
    """Train all four models, evaluate, and return (DataFrame, dict of models)."""
    results = []
    roc_data = {}
    trained_models = {}

    print("\n" + "=" * 80)
    print("MODEL COMPARISON EXPERIMENT")
    print("=" * 80)

    # 1. Baseline ANN ------------------------------------------------
    print("\n[1/4] Training Baseline ANN Model...")
    print("-" * 80)
    baseline = BaselineANN(hidden_layer_sizes=(100, 50), max_iter=500)
    baseline.train(X_train, y_train)
    m = baseline.evaluate(X_test, y_test, verbose=verbose,
                          model_name="BASELINE ANN")
    results.append(_row("Baseline ANN", m))
    roc_data["Baseline ANN"] = (m['fpr'], m['tpr'], m['auc'])
    trained_models["Baseline ANN"] = baseline

    # 2. GA-Optimised ANN -------------------------------------------
    print("\n[2/4] Training GA-Optimised ANN Model...")
    print("-" * 80)
    try:
        ga_opt = GAOptimizer(
            X_train, y_train, X_val, y_val,
            n_population=50, n_generations=40, random_state=42)
        ga_opt.optimize(verbose=verbose)
        ga_ann = ga_opt.get_optimized_ann()
        ga_ann.train(X_train, y_train)
        m = ga_ann.evaluate(X_test, y_test, verbose=verbose,
                            model_name="GA-OPTIMISED ANN")
        results.append(_row("GA-Optimized ANN", m))
        roc_data["GA-Optimized ANN"] = (m['fpr'], m['tpr'], m['auc'])
        trained_models["GA-Optimized ANN"] = ga_ann
    except Exception as e:
        print(f"GA optimization failed: {e}")
        results.append(_zero_row("GA-Optimized ANN"))

    # 3. PSO-Optimised ANN ------------------------------------------
    print("\n[3/4] Training PSO-Optimised ANN Model...")
    print("-" * 80)
    try:
        pso_opt = PSOOptimizer(
            X_train, y_train, X_val, y_val,
            n_particles=30, iterations=50, random_state=42)
        pso_opt.optimize(initial_ann=baseline, verbose=verbose)
        pso_ann = pso_opt.get_optimized_ann()
        pso_ann.train(X_train, y_train)
        m = pso_ann.evaluate(X_test, y_test, verbose=verbose,
                             model_name="PSO-OPTIMISED ANN")
        results.append(_row("PSO-Optimized ANN", m))
        roc_data["PSO-Optimized ANN"] = (m['fpr'], m['tpr'], m['auc'])
        trained_models["PSO-Optimized ANN"] = pso_ann
    except Exception as e:
        print(f"PSO optimization failed: {e}")
        results.append(_zero_row("PSO-Optimized ANN"))

    # 4. Hybrid GA-PSO-ANN -----------------------------------------
    print("\n[4/4] Training Hybrid GA-PSO-ANN Model...")
    print("-" * 80)
    try:
        hybrid = HybridGAPSOANN(
            hidden_layer_sizes=(100, 50),
            ga_params={'n_population': 50, 'n_generations': 40},
            pso_params={'n_particles': 30, 'iterations': 50},
            random_state=42,
        )
        hybrid.train(X_train, y_train, X_val, y_val, verbose=verbose)
        m = hybrid.evaluate(X_test, y_test, verbose=verbose)
        results.append(_row("Hybrid GA-PSO-ANN", m))
        roc_data["Hybrid GA-PSO-ANN"] = (m['fpr'], m['tpr'], m['auc'])
        trained_models["Hybrid GA-PSO-ANN"] = hybrid
    except Exception as e:
        print(f"Hybrid model training failed: {e}")
        results.append(_zero_row("Hybrid GA-PSO-ANN"))

    df = pd.DataFrame(results)

    print("\n" + "=" * 80)
    print("FINAL COMPARISON RESULTS")
    print("=" * 80)
    print(df.to_string(index=False))
    best_idx = df['Accuracy'].idxmax()
    print(f"\nBest Model : {df.loc[best_idx, 'Model']}")
    print(f"Accuracy   : {df.loc[best_idx, 'Accuracy']:.4f}")
    print(f"AUC        : {df.loc[best_idx, 'AUC']:.4f}")
    print("=" * 80)

    return df, roc_data, trained_models


def _row(name, m):
    return {
        'Model': name,
        'Accuracy': m['accuracy'], 'Precision': m['precision'],
        'Recall': m['recall'], 'F1-Score': m['f1_score'],
        'AUC': m['auc'],
    }


def _zero_row(name):
    return {
        'Model': name,
        'Accuracy': 0.0, 'Precision': 0.0,
        'Recall': 0.0, 'F1-Score': 0.0, 'AUC': 0.0,
    }


# ====================================================================
# Cross-validation
# ====================================================================
def cross_validate_models(X, y, n_folds=5, random_state=42):
    """Run stratified k-fold CV for all four model types."""
    print("\n" + "=" * 80)
    print(f"{n_folds}-FOLD CROSS-VALIDATION")
    print("=" * 80)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True,
                          random_state=random_state)

    model_names = [
        "Baseline ANN", "GA-Optimized ANN",
        "PSO-Optimized ANN", "Hybrid GA-PSO-ANN",
    ]
    cv_scores = {n: {'Accuracy': [], 'Precision': [], 'Recall': [],
                      'F1-Score': [], 'AUC': []}
                 for n in model_names}

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n--- Fold {fold}/{n_folds} ---")
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        X_train, X_val, y_train, y_val = train_test_split(
            X_tr, y_tr, test_size=0.2, random_state=42, stratify=y_tr)

        # Baseline ANN
        try:
            ann = BaselineANN(hidden_layer_sizes=(100, 50), max_iter=300)
            ann.train(X_train, y_train)
            m = ann.evaluate(X_te, y_te, verbose=False)
            _append_cv(cv_scores, "Baseline ANN", m)
        except Exception:
            _append_cv_zero(cv_scores, "Baseline ANN")

        # GA-ANN (reduced params for speed)
        try:
            ga = GAOptimizer(X_train, y_train, X_val, y_val,
                             n_population=25, n_generations=15,
                             random_state=42)
            ga.optimize(verbose=False)
            ga_ann = ga.get_optimized_ann()
            ga_ann.train(X_train, y_train)
            m = ga_ann.evaluate(X_te, y_te, verbose=False)
            _append_cv(cv_scores, "GA-Optimized ANN", m)
        except Exception:
            _append_cv_zero(cv_scores, "GA-Optimized ANN")

        # PSO-ANN (reduced params)
        try:
            pso = PSOOptimizer(X_train, y_train, X_val, y_val,
                               n_particles=15, iterations=20,
                               random_state=42)
            pso.optimize(initial_ann=ann, verbose=False)
            pso_ann = pso.get_optimized_ann()
            pso_ann.train(X_train, y_train)
            m = pso_ann.evaluate(X_te, y_te, verbose=False)
            _append_cv(cv_scores, "PSO-Optimized ANN", m)
        except Exception:
            _append_cv_zero(cv_scores, "PSO-Optimized ANN")

        # Hybrid (reduced params)
        try:
            hybrid = HybridGAPSOANN(
                ga_params={'n_population': 25, 'n_generations': 15},
                pso_params={'n_particles': 15, 'iterations': 20},
                random_state=42)
            hybrid.train(X_train, y_train, X_val, y_val, verbose=False)
            m = hybrid.evaluate(X_te, y_te, verbose=False)
            _append_cv(cv_scores, "Hybrid GA-PSO-ANN", m)
        except Exception:
            _append_cv_zero(cv_scores, "Hybrid GA-PSO-ANN")

    # Build summary DataFrame
    rows = []
    for name in model_names:
        row = {'Model': name}
        for metric in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']:
            vals = cv_scores[name][metric]
            row[f'{metric}_mean'] = np.mean(vals)
            row[f'{metric}_std']  = np.std(vals)
        rows.append(row)

    cv_df = pd.DataFrame(rows)

    print("\n" + "=" * 80)
    print("CROSS-VALIDATION RESULTS  (mean ± std)")
    print("=" * 80)
    for _, r in cv_df.iterrows():
        print(f"\n{r['Model']}:")
        for metric in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']:
            print(f"  {metric:10s}: {r[f'{metric}_mean']:.4f} ± {r[f'{metric}_std']:.4f}")
    print("=" * 80)

    return cv_df


def _append_cv(cv_scores, name, m):
    cv_scores[name]['Accuracy'].append(m['accuracy'])
    cv_scores[name]['Precision'].append(m['precision'])
    cv_scores[name]['Recall'].append(m['recall'])
    cv_scores[name]['F1-Score'].append(m['f1_score'])
    cv_scores[name]['AUC'].append(m['auc'])


def _append_cv_zero(cv_scores, name):
    for k in cv_scores[name]:
        cv_scores[name][k].append(0.0)


# ====================================================================
# Plotting
# ====================================================================
def plot_results(comparison_df, roc_data, save_dir='results'):
    os.makedirs(save_dir, exist_ok=True)

    # --- Bar chart ---
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(comparison_df))
    width = 0.15
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
    for i, metric in enumerate(metrics):
        ax.bar(x + i * width, comparison_df[metric], width, label=metric)
    ax.set_xlabel('Models')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(comparison_df['Model'], rotation=30, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1])
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'comparison.png'), dpi=300)
    print(f"Bar chart saved to {save_dir}/comparison.png")

    # --- ROC curves ---
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA']
    for (name, (fpr, tpr, auc)), color in zip(roc_data.items(), colors):
        ax.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})', color=color)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'roc_curves.png'), dpi=300)
    print(f"ROC curves saved to {save_dir}/roc_curves.png")


# ====================================================================
# Save models for Streamlit UI
# ====================================================================
def save_all_models(trained_models, preprocessor, feature_names):
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(models_dir, exist_ok=True)

    name_to_file = {
        "Baseline ANN": "baseline_ann.joblib",
        "GA-Optimized ANN": "ga_ann.joblib",
        "PSO-Optimized ANN": "pso_ann.joblib",
        "Hybrid GA-PSO-ANN": "hybrid_ann.joblib",
    }

    for name, model_obj in trained_models.items():
        fname = name_to_file.get(name)
        if fname is None:
            continue
        # For Hybrid, save the inner trained MLPClassifier
        if hasattr(model_obj, 'final_model'):
            joblib.dump(model_obj.final_model.model,
                        os.path.join(models_dir, fname))
        else:
            joblib.dump(model_obj.model,
                        os.path.join(models_dir, fname))
        print(f"  Saved {fname}")

    # Also save as 'ann_model.joblib' (legacy name used by save_model.py)
    if "Baseline ANN" in trained_models:
        joblib.dump(trained_models["Baseline ANN"].model,
                    os.path.join(models_dir, 'ann_model.joblib'))

    joblib.dump(preprocessor.scaler,
                os.path.join(models_dir, 'scaler.joblib'))
    joblib.dump(preprocessor.label_encoders,
                os.path.join(models_dir, 'label_encoders.joblib'))
    joblib.dump(feature_names,
                os.path.join(models_dir, 'feature_names.joblib'))
    print("  Saved scaler, label_encoders, feature_names")


# ====================================================================
# Main
# ====================================================================
def main():
    print("=" * 80)
    print("HEART DISEASE PREDICTION SYSTEM — EXPERIMENT RUNNER")
    print("=" * 80)

    # --- Step 1: Load & preprocess ---
    print("\n[STEP 1] Loading and Preprocessing Data...")
    print("-" * 80)

    preprocessor = HeartDiseasePreprocessor()

    # Attempt to load all 4 UCI heart disease datasets (Cleveland,
    # Hungarian, Switzerland, VA Long Beach) for ~920 samples.
    search_dirs = ['heart disease', 'heart+disease', 'data']
    site_files = [
        'processed.cleveland.data',
        'processed.hungarian.data',
        'processed.switzerland.data',
        'processed.va.data',
    ]

    all_paths = []
    for d in search_dirs:
        found = [os.path.join(d, f) for f in site_files
                 if os.path.exists(os.path.join(d, f))]
        if found:
            all_paths = found
            break

    if len(all_paths) >= 2:
        print(f"Found {len(all_paths)} dataset files — combining all sites")
        X_train_full, X_test, y_train_full, y_test, feature_names = \
            preprocessor.preprocess_pipeline(file_paths=all_paths)
    elif len(all_paths) == 1:
        print(f"Dataset: {all_paths[0]}")
        X_train_full, X_test, y_train_full, y_test, feature_names = \
            preprocessor.preprocess_pipeline(file_path=all_paths[0])
    else:
        # Fallback: try single cleveland file or download
        single = None
        for d in search_dirs:
            for f in ['processed.cleveland.data', 'cleveland.data']:
                c = os.path.join(d, f)
                if os.path.exists(c):
                    single = c
                    break
            if single:
                break
        if single:
            print(f"Dataset: {single}")
            X_train_full, X_test, y_train_full, y_test, feature_names = \
                preprocessor.preprocess_pipeline(file_path=single)
        else:
            print("Downloading from UCI repository...")
            try:
                X_train_full, X_test, y_train_full, y_test, feature_names = \
                    preprocessor.preprocess_pipeline()
            except Exception as e:
                print(f"\nError: {e}")
                print("Place the datasets in 'heart+disease/' or install ucimlrepo")
                return

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=0.2, random_state=42, stratify=y_train_full)

    print(f"\nTrain: {X_train.shape[0]}  Val: {X_val.shape[0]}  "
          f"Test: {X_test.shape[0]}  Features: {len(feature_names)}")

    # --- Step 2: Single-split comparison ---
    print("\n[STEP 2] Running Model Comparison...")
    comparison_df, roc_data, trained_models = compare_models(
        X_train, y_train, X_val, y_val, X_test, y_test)

    os.makedirs('results', exist_ok=True)
    comparison_df.to_csv('results/comparison_results.csv', index=False)
    joblib.dump(roc_data, 'results/roc_data.joblib')
    print("Results saved to results/")

    # --- Step 3: Plots ---
    print("\n[STEP 3] Generating Plots...")
    plot_results(comparison_df, roc_data)

    # --- Step 4: Cross-validation ---
    print("\n[STEP 4] Cross-Validation...")

    # Rebuild full preprocessed feature set for CV
    X_full = pd.concat([X_train_full, X_test], ignore_index=True)
    y_full = pd.concat([y_train_full, y_test], ignore_index=True)
    cv_df = cross_validate_models(X_full, y_full, n_folds=5)
    cv_df.to_csv('results/cv_results.csv', index=False)
    print("CV results saved to results/cv_results.csv")

    # --- Step 5: Save models for UI ---
    print("\n[STEP 5] Saving Models for Web UI...")
    save_all_models(trained_models, preprocessor, feature_names)

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE!  Next: streamlit run app.py")
    print("=" * 80)


if __name__ == "__main__":
    main()
