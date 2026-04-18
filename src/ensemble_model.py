"""
Seed-Averaging Ensemble for the best-performing hybrid configuration.

Trains the SAME hyperparameter configuration with N different random seeds
and averages their probability outputs at inference time. This is a cheap,
consistent +1-2% accuracy gain on small tabular datasets and eliminates
seed-lottery noise from the final reported numbers.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve,
)

from baseline_ann import BaselineANN
import warnings
warnings.filterwarnings("ignore")


class SeedEnsembleANN:
    """Average predict_proba across N ANNs trained with different seeds."""

    def __init__(self, params: dict[str, Any], n_models: int = 5,
                 base_seed: int = 42, max_iter: int = 1000):
        self.params = params
        self.n_models = n_models
        self.base_seed = base_seed
        self.max_iter = max_iter
        self.models: list[BaselineANN] = []

    def train(self, X_train, y_train, verbose: bool = True):
        if verbose:
            print(f"Training seed ensemble ({self.n_models} models)...")
        self.models = []
        for i in range(self.n_models):
            seed = self.base_seed + i
            ann = BaselineANN(
                hidden_layer_sizes=self.params["hidden_layer_sizes"],
                activation=self.params.get("activation", "relu"),
                learning_rate_init=self.params["learning_rate"],
                alpha=self.params["alpha"],
                momentum=self.params.get("momentum", 0.9),
                max_iter=self.max_iter,
                random_state=seed,
                early_stopping=True,
            )
            ann.train(X_train, y_train, verbose=False)
            self.models.append(ann)
            if verbose:
                print(f"  Model {i + 1}/{self.n_models} trained (seed={seed})")
        return self

    def predict_proba(self, X):
        if not self.models:
            raise RuntimeError("Ensemble not trained.")
        return np.mean([m.predict_proba(X) for m in self.models], axis=0)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def evaluate(self, X_test, y_test, verbose: bool = True,
                 model_name: str = "ENSEMBLE"):
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)[:, 1]
        metrics = {
            "accuracy":  accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall":    recall_score(y_test, y_pred, zero_division=0),
            "f1_score":  f1_score(y_test, y_pred, zero_division=0),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
        }
        try:
            metrics["auc"] = roc_auc_score(y_test, y_proba)
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            metrics["fpr"], metrics["tpr"] = fpr, tpr
        except ValueError:
            metrics["auc"] = 0.0
            metrics["fpr"] = np.array([0, 1])
            metrics["tpr"] = np.array([0, 1])

        if verbose:
            print(f"\n{'=' * 60}\n{model_name} EVALUATION\n{'=' * 60}")
            print(f"Accuracy:  {metrics['accuracy']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall:    {metrics['recall']:.4f}")
            print(f"F1-Score:  {metrics['f1_score']:.4f}")
            print(f"AUC:       {metrics['auc']:.4f}")
            print("=" * 60)
        return metrics
