"""
Hybrid GA-PSO-ANN Model for Heart Disease Prediction

Three-stage optimisation pipeline:
  Stage 1 – GA   : searches architecture + learning-rate space
  Stage 2 – PSO  : fine-tunes learning-rate + L2-regularisation for
                    the GA-selected architecture
  Stage 3 – Final: trains the ANN with all optimised hyper-parameters

Based on the methodology described in the research proposal.
"""

import numpy as np
from baseline_ann import BaselineANN
from ga_optimizer import GAOptimizer
from pso_optimizer import PSOOptimizer
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')


class HybridGAPSOANN:
    """Hybrid GA-PSO-ANN model."""

    def __init__(self, hidden_layer_sizes=(100, 50),
                 ga_params=None, pso_params=None,
                 random_state=42):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.random_state = random_state

        _default_ga = {
            'n_population': 30, 'n_generations': 20,
            'crossover_prob': 0.7, 'mutation_prob': 0.2,
        }
        self.ga_params = {**_default_ga, **(ga_params or {})}

        _default_pso = {
            'n_particles': 20, 'iterations': 30,
            'options': {'c1': 0.5, 'c2': 0.3, 'w': 0.9},
        }
        self.pso_params = {**_default_pso, **(pso_params or {})}

        self.ga_optimizer = None
        self.pso_optimizer = None
        self.final_model = None
        self.ga_history = None
        self.pso_history = None
        self.best_params = {}

    # ------------------------------------------------------------------ #
    #  Training pipeline                                                  #
    # ------------------------------------------------------------------ #
    def train(self, X_train, y_train, X_val, y_val, verbose=True):
        print("\n" + "=" * 70)
        print("HYBRID GA-PSO-ANN MODEL TRAINING")
        print("=" * 70)

        # ---- Stage 1: GA — architecture + learning-rate search --------
        if verbose:
            print("\n[STAGE 1] Genetic Algorithm — architecture search")
            print("-" * 70)

        self.ga_optimizer = GAOptimizer(
            X_train, y_train, X_val, y_val,
            n_population=self.ga_params['n_population'],
            n_generations=self.ga_params['n_generations'],
            crossover_prob=self.ga_params['crossover_prob'],
            mutation_prob=self.ga_params['mutation_prob'],
            random_state=self.random_state,
        )
        ga_results = self.ga_optimizer.optimize(verbose=verbose)
        self.ga_history = ga_results['fitness_history']
        ga_params = self.ga_optimizer.get_best_params()

        ga_ann = self.ga_optimizer.get_optimized_ann()
        ga_ann.train(X_train, y_train)
        ga_val_acc = accuracy_score(y_val, ga_ann.predict(X_val))
        if verbose:
            print(f"\nGA validation accuracy: {ga_val_acc:.4f}")

        # ---- Stage 2: PSO — fine-tune lr + alpha ---------------------
        if verbose:
            print("\n[STAGE 2] PSO — fine-tuning training parameters")
            print("-" * 70)

        self.pso_optimizer = PSOOptimizer(
            X_train, y_train, X_val, y_val,
            n_particles=self.pso_params['n_particles'],
            iterations=self.pso_params['iterations'],
            hidden_layer_sizes=ga_params['hidden_layer_sizes'],
            options=self.pso_params['options'],
            random_state=self.random_state,
        )
        pso_results = self.pso_optimizer.optimize(
            initial_ann=ga_ann, verbose=verbose,
        )
        self.pso_history = pso_results['fitness_history']
        pso_params = self.pso_optimizer.get_best_params()

        # ---- Stage 3: Train final model with all optimised params ----
        if verbose:
            print("\n[STAGE 3] Final model training with optimised parameters")
            print("-" * 70)

        self.best_params = {
            'hidden_layer_sizes': ga_params['hidden_layer_sizes'],
            'learning_rate': pso_params['learning_rate'],
            'alpha': pso_params['alpha'],
        }

        self.final_model = BaselineANN(
            hidden_layer_sizes=self.best_params['hidden_layer_sizes'],
            max_iter=1000,
            learning_rate_init=self.best_params['learning_rate'],
            random_state=self.random_state,
        )
        self.final_model.model.alpha = self.best_params['alpha']
        self.final_model.train(X_train, y_train)

        final_acc = accuracy_score(y_val, self.final_model.predict(X_val))
        if verbose:
            print(f"Final hybrid model validation accuracy: {final_acc:.4f}")
            print(f"Optimised parameters: {self.best_params}")
            print("=" * 70)

        return self

    # ------------------------------------------------------------------ #
    #  Inference / evaluation                                             #
    # ------------------------------------------------------------------ #
    def predict(self, X):
        if self.final_model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        return self.final_model.predict(X)

    def predict_proba(self, X):
        if self.final_model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        return self.final_model.predict_proba(X)

    def evaluate(self, X_test, y_test, verbose=True):
        if self.final_model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        return self.final_model.evaluate(
            X_test, y_test, verbose=verbose,
            model_name="HYBRID GA-PSO-ANN",
        )

    def get_optimization_history(self):
        return {
            'ga_fitness_history': self.ga_history,
            'pso_fitness_history': self.pso_history,
        }


if __name__ == "__main__":
    from data_preprocessing import HeartDiseasePreprocessor
    from sklearn.model_selection import train_test_split

    preprocessor = HeartDiseasePreprocessor()
    try:
        X_tr_full, X_test, y_tr_full, y_test, _ = \
            preprocessor.preprocess_pipeline()
        X_train, X_val, y_train, y_val = train_test_split(
            X_tr_full, y_tr_full, test_size=0.2,
            random_state=42, stratify=y_tr_full)

        hybrid = HybridGAPSOANN(
            ga_params={'n_population': 15, 'n_generations': 10},
            pso_params={'n_particles': 10, 'iterations': 10},
        )
        hybrid.train(X_train, y_train, X_val, y_val)
        hybrid.evaluate(X_test, y_test)
    except Exception as e:
        import traceback; traceback.print_exc()
