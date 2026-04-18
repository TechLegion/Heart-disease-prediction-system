"""
Baseline Artificial Neural Network Model for Heart Disease Prediction.

A thin, configurable wrapper around scikit-learn's MLPClassifier so that the
GA / PSO / Hybrid optimisers can treat every hyperparameter uniformly.
"""

from __future__ import annotations

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve,
)
import warnings
warnings.filterwarnings("ignore")


class BaselineANN:
    """Wrapper around scikit-learn's MLPClassifier with a stable interface."""

    def __init__(
        self,
        hidden_layer_sizes: tuple[int, ...] = (100, 50),
        max_iter: int = 500,
        random_state: int = 42,
        learning_rate_init: float = 0.001,
        alpha: float = 0.0001,
        activation: str = "relu",
        solver: str = "adam",
        momentum: float = 0.9,
        batch_size: int | str = "auto",
        early_stopping: bool = True,
        n_iter_no_change: int = 15,
    ):
        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            random_state=random_state,
            learning_rate_init=learning_rate_init,
            alpha=alpha,
            activation=activation,
            solver=solver,
            momentum=momentum,
            batch_size=batch_size,
            early_stopping=early_stopping,
            validation_fraction=0.1 if early_stopping else 0.0,
            n_iter_no_change=n_iter_no_change,
        )
        self.history = None

    def train(self, X_train, y_train, verbose: bool = True):
        if verbose:
            print("Training ANN...")
        self.model.fit(X_train, y_train)
        if verbose:
            print(f"  Final loss: {self.model.loss_:.4f}  "
                  f"iters: {self.model.n_iter_}")
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def evaluate(self, X_test, y_test, verbose: bool = True,
                 model_name: str = "ANN"):
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
            print(f"Confusion matrix:\n{metrics['confusion_matrix']}")
            print("=" * 60)
        return metrics

    def get_model_info(self):
        return {
            "hidden_layer_sizes": self.model.hidden_layer_sizes,
            "activation": self.model.activation,
            "learning_rate_init": self.model.learning_rate_init,
            "alpha": self.model.alpha,
            "momentum": self.model.momentum,
            "n_iter": getattr(self.model, "n_iter_", None),
            "loss": getattr(self.model, "loss_", None),
        }


if __name__ == "__main__":
    from data_preprocessing import HeartDiseasePreprocessor
    pp = HeartDiseasePreprocessor()
    X_tr, X_te, y_tr, y_te, _ = pp.preprocess_pipeline()
    ann = BaselineANN()
    ann.train(X_tr, y_tr)
    ann.evaluate(X_te, y_te, model_name="BASELINE ANN")
