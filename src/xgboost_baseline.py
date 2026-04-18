"""
XGBoost baseline for heart disease prediction.

Used as an independent "ceiling reference" against the ANN-based models.
Tree-based gradient boosting is strong on this dataset because it handles
missing-value patterns and feature interactions naturally.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve,
)
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")


class XGBoostBaseline:
    """Thin wrapper so the main runner can treat it identically to the ANNs."""

    def __init__(self, n_estimators: int = 400, max_depth: int = 4,
                 learning_rate: float = 0.05, subsample: float = 0.9,
                 colsample_bytree: float = 0.9, random_state: int = 42):
        self.model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            eval_metric="logloss",
            n_jobs=-1,
            tree_method="hist",
        )

    def train(self, X_train, y_train, verbose: bool = True):
        if verbose:
            print("Training XGBoost baseline...")
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def evaluate(self, X_test, y_test, verbose: bool = True,
                 model_name: str = "XGBOOST"):
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
