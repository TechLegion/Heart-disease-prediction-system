"""
Hybrid GA-PSO-ANN Model for Heart Disease Prediction.

Three-stage optimisation pipeline:
  Stage 1 — GA: evolves a population of architectures + training params,
                returns the TOP-K distinct architectures.
  Stage 2 — PSO: fine-tunes training hyperparameters (lr, alpha, momentum)
                 for each of the K GA candidates. The best CV score wins.
  Stage 3 — Final model trained with the winning config.

This fixes the key weakness of the v1 Hybrid: we no longer commit to a single
GA architecture before PSO, so a slightly-worse GA candidate with much better
training-parameter potential can still win.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

from baseline_ann import BaselineANN
from ga_optimizer import GAOptimizer
from pso_optimizer import PSOOptimizer
import warnings
warnings.filterwarnings("ignore")


class HybridGAPSOANN:
    """GA → top-K → PSO-per-candidate → best-of-K → final training."""

    def __init__(self, top_k: int = 3,
                 ga_params: dict[str, Any] | None = None,
                 pso_params: dict[str, Any] | None = None,
                 cv_folds: int = 3, random_state: int = 42):
        self.top_k = top_k
        self.random_state = random_state
        self.cv_folds = cv_folds

        _default_ga = {
            "n_population": 30, "n_generations": 20,
            "crossover_prob": 0.7, "mutation_prob": 0.25,
            "cv_folds": cv_folds, "elitism": 2,
        }
        self.ga_params = {**_default_ga, **(ga_params or {})}

        _default_pso = {
            "n_particles": 15, "iterations": 20,
            "cv_folds": cv_folds,
            "options": {"c1": 0.5, "c2": 0.3, "w": 0.9},
        }
        self.pso_params = {**_default_pso, **(pso_params or {})}

        self.ga_optimizer: GAOptimizer | None = None
        self.pso_runs: list[dict[str, Any]] = []
        self.final_model: BaselineANN | None = None
        self.best_params: dict[str, Any] = {}
        self.ga_history: list[float] | None = None
        self.pso_history: list[float] | None = None

    # ------------------------------------------------------------------ #
    def _cv_accuracy(self, model_factory, X, y) -> float:
        X = np.asarray(X); y = np.asarray(y)
        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True,
                              random_state=self.random_state)
        scores = []
        for tr, va in skf.split(X, y):
            ann = model_factory()
            ann.train(X[tr], y[tr], verbose=False)
            scores.append(accuracy_score(y[va], ann.predict(X[va])))
        return float(np.mean(scores))

    # ------------------------------------------------------------------ #
    def train(self, X_train, y_train, X_val=None, y_val=None,
              verbose: bool = True):
        print("\n" + "=" * 70)
        print("HYBRID GA-PSO-ANN MODEL TRAINING (v2: top-K GA → PSO)")
        print("=" * 70)

        # --- Stage 1: GA -------------------------------------------------
        if verbose:
            print("\n[STAGE 1] Genetic Algorithm — architecture search")
        self.ga_optimizer = GAOptimizer(
            X_train, y_train,
            n_population=self.ga_params["n_population"],
            n_generations=self.ga_params["n_generations"],
            crossover_prob=self.ga_params["crossover_prob"],
            mutation_prob=self.ga_params["mutation_prob"],
            cv_folds=self.ga_params["cv_folds"],
            elitism=self.ga_params["elitism"],
            random_state=self.random_state,
        )
        ga_result = self.ga_optimizer.optimize(verbose=verbose)
        self.ga_history = ga_result["fitness_history"]

        top_params = self.ga_optimizer.get_topk_params(k=self.top_k)
        print(f"\nGA returned {len(top_params)} top candidates:")
        for i, p in enumerate(top_params, 1):
            print(f"  #{i}: arch={p['hidden_layer_sizes']}  act={p['activation']}  "
                  f"lr={p['learning_rate']:.5f}  α={p['alpha']:.1e}")

        # --- Stage 2: PSO per candidate ---------------------------------
        if verbose:
            print("\n[STAGE 2] PSO fine-tuning for each GA candidate")
        best_overall = {"cv_score": -np.inf, "params": None, "config_idx": -1}

        for idx, params in enumerate(top_params, 1):
            print(f"\n  --- PSO on candidate #{idx} "
                  f"(arch={params['hidden_layer_sizes']}, "
                  f"act={params['activation']}) ---")
            seed_ann = BaselineANN(
                hidden_layer_sizes=params["hidden_layer_sizes"],
                activation=params["activation"],
                learning_rate_init=params["learning_rate"],
                alpha=params["alpha"],
                random_state=self.random_state,
            )
            pso = PSOOptimizer(
                X_train, y_train,
                n_particles=self.pso_params["n_particles"],
                iterations=self.pso_params["iterations"],
                base_params={
                    "hidden_layer_sizes": params["hidden_layer_sizes"],
                    "activation": params["activation"],
                },
                cv_folds=self.pso_params["cv_folds"],
                options=self.pso_params["options"],
                random_state=self.random_state,
            )
            pso_result = pso.optimize(initial_ann=seed_ann, verbose=False)
            cv_score = pso_result["best_fitness"]
            best_tp = pso_result["best_params"]

            self.pso_runs.append({
                "candidate": params,
                "cv_score": cv_score,
                "best_training_params": best_tp,
                "history": pso_result["fitness_history"],
            })

            print(f"    CV acc = {cv_score:.4f}  "
                  f"(lr={best_tp['learning_rate']:.5f}, "
                  f"α={best_tp['alpha']:.1e}, "
                  f"momentum={best_tp['momentum']:.3f})")

            if cv_score > best_overall["cv_score"]:
                best_overall["cv_score"] = cv_score
                best_overall["params"] = {
                    "hidden_layer_sizes": params["hidden_layer_sizes"],
                    "activation": params["activation"],
                    **best_tp,
                }
                best_overall["config_idx"] = idx
                self.pso_history = pso_result["fitness_history"]

        self.best_params = best_overall["params"] or {}
        print(f"\nBest Hybrid config (CV acc = {best_overall['cv_score']:.4f}): "
              f"candidate #{best_overall['config_idx']} → {self.best_params}")

        # --- Stage 3: Final training -----------------------------------
        if verbose:
            print("\n[STAGE 3] Final training with best hybrid config")
        self.final_model = BaselineANN(
            hidden_layer_sizes=self.best_params["hidden_layer_sizes"],
            activation=self.best_params["activation"],
            learning_rate_init=self.best_params["learning_rate"],
            alpha=self.best_params["alpha"],
            momentum=self.best_params["momentum"],
            max_iter=1000, random_state=self.random_state,
            early_stopping=True,
        )
        self.final_model.train(X_train, y_train, verbose=False)
        print("=" * 70)
        return self

    # ------------------------------------------------------------------ #
    def predict(self, X):
        if self.final_model is None:
            raise ValueError("Model not trained yet.")
        return self.final_model.predict(X)

    def predict_proba(self, X):
        if self.final_model is None:
            raise ValueError("Model not trained yet.")
        return self.final_model.predict_proba(X)

    def evaluate(self, X_test, y_test, verbose: bool = True):
        if self.final_model is None:
            raise ValueError("Model not trained yet.")
        return self.final_model.evaluate(
            X_test, y_test, verbose=verbose,
            model_name="HYBRID GA-PSO-ANN",
        )

    def get_optimization_history(self):
        return {
            "ga_fitness_history": self.ga_history,
            "pso_fitness_history": self.pso_history,
            "pso_runs": self.pso_runs,
        }


if __name__ == "__main__":
    from data_preprocessing import HeartDiseasePreprocessor
    pp = HeartDiseasePreprocessor()
    X_tr, X_te, y_tr, y_te, _ = pp.preprocess_pipeline()
    hybrid = HybridGAPSOANN(
        top_k=2,
        ga_params={"n_population": 15, "n_generations": 8},
        pso_params={"n_particles": 8, "iterations": 8},
    )
    hybrid.train(X_tr, y_tr)
    hybrid.evaluate(X_te, y_te)
