"""
Baseline Artificial Neural Network Model for Heart Disease Prediction

Implements a standard ANN using scikit-learn's MLPClassifier as the baseline
for comparison with GA-optimized, PSO-optimized, and Hybrid GA-PSO-ANN variants.
"""

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve,
)
import warnings
warnings.filterwarnings('ignore')


class BaselineANN:
    """
    Baseline ANN model using scikit-learn's Multi-Layer Perceptron classifier.
    """

    def __init__(self, hidden_layer_sizes=(100, 50), max_iter=1000,
                 random_state=42, learning_rate_init=0.001,
                 activation='relu', solver='adam', early_stopping=True):
        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            random_state=random_state,
            learning_rate_init=learning_rate_init,
            activation=activation,
            solver=solver,
            early_stopping=early_stopping,
            validation_fraction=0.1 if early_stopping else 0.0,
            n_iter_no_change=10,
        )
        self.history = None

    def train(self, X_train, y_train):
        print("Training baseline ANN model...")
        self.model.fit(X_train, y_train)
        print(f"Training completed. Final loss: {self.model.loss_:.4f}")
        print(f"Number of iterations: {self.model.n_iter_}")
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def evaluate(self, X_test, y_test, verbose=True, model_name="BASELINE ANN"):
        """Evaluate model and return metrics dict including AUC and ROC data."""
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
        recall = recall_score(y_test, y_pred, average='binary', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)
        cm = confusion_matrix(y_test, y_pred)

        try:
            auc = roc_auc_score(y_test, y_pred_proba)
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        except ValueError:
            auc = 0.0
            fpr, tpr = np.array([0, 1]), np.array([0, 1])

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'confusion_matrix': cm,
            'fpr': fpr,
            'tpr': tpr,
        }

        if verbose:
            print("\n" + "=" * 60)
            print(f"{model_name} MODEL EVALUATION")
            print("=" * 60)
            print(f"Accuracy:  {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall:    {recall:.4f}")
            print(f"F1-Score:  {f1:.4f}")
            print(f"AUC:       {auc:.4f}")
            print("\nConfusion Matrix:")
            print(cm)
            print("=" * 60)

        return metrics

    # --- weight access (kept for compatibility) ---
    def get_weights(self):
        return {'weights': self.model.coefs_, 'biases': self.model.intercepts_}

    def set_weights(self, weights, biases):
        self.model.coefs_ = weights
        self.model.intercepts_ = biases

    def get_model_info(self):
        return {
            'hidden_layer_sizes': self.model.hidden_layer_sizes,
            'n_layers': self.model.n_layers_,
            'n_outputs': self.model.n_outputs_,
            'activation': self.model.activation,
            'solver': self.model.solver,
            'learning_rate_init': self.model.learning_rate_init,
            'n_iter': self.model.n_iter_,
            'loss': self.model.loss_,
        }


if __name__ == "__main__":
    from data_preprocessing import HeartDiseasePreprocessor

    preprocessor = HeartDiseasePreprocessor()
    try:
        X_train, X_test, y_train, y_test, feature_names = preprocessor.preprocess_pipeline()
        ann = BaselineANN(hidden_layer_sizes=(100, 50), max_iter=500)
        ann.train(X_train, y_train)
        ann.evaluate(X_test, y_test)
    except Exception as e:
        print(f"Error: {e}")
