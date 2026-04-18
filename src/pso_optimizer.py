"""
Particle Swarm Optimization for ANN Training-Parameter Tuning.

Improvements over v1:
  * 3-D search: learning_rate, alpha (L2), momentum — richer than the old
    2-D lr/alpha space.
  * CV-based fitness (same motivation as in GAOptimizer).
  * Can fine-tune ANY fixed architecture+activation passed in via
    `base_params`, which lets the Hybrid stage explore multiple GA
    candidates instead of just one.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pyswarms.single import GlobalBestPSO
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

from baseline_ann import BaselineANN
import warnings
warnings.filterwarnings("ignore")


def _decode_particle(pos: np.ndarray) -> dict[str, float]:
    p = np.clip(pos, 0.0, 1.0)
    lr       = 10 ** (-3.5 + p[0] * 2.0)     # [~3e-4, ~3e-2]
    alpha    = 10 ** (-5 + p[1] * 4)         # [1e-5, 1e-1]
    momentum = 0.5 + p[2] * 0.49             # [0.5, 0.99]
    return {"learning_rate": float(lr), "alpha": float(alpha),
            "momentum": float(momentum)}


class PSOOptimizer:
    """PSO optimiser over 3 continuous training hyperparameters."""

    DIMENSION = 3

    def __init__(self, X_train, y_train, X_val=None, y_val=None,
                 n_particles: int = 20, iterations: int = 25,
                 base_params: dict[str, Any] | None = None,
                 options: dict[str, float] | None = None,
                 cv_folds: int = 3, eval_max_iter: int = 200,
                 random_state: int = 42):
        self.X_train = np.asarray(X_train)
        self.y_train = np.asarray(y_train)
        self.X_val, self.y_val = X_val, y_val

        self.n_particles = n_particles
        self.iterations = iterations
        self.cv_folds = cv_folds
        self.eval_max_iter = eval_max_iter
        self.random_state = random_state
        self.options = options or {"c1": 0.5, "c2": 0.3, "w": 0.9}

        # Architecture + activation held fixed during PSO. The Hybrid stage
        # passes these in per candidate architecture.
        self.base_params: dict[str, Any] = base_params or {
            "hidden_layer_sizes": (100, 50),
            "activation": "relu",
        }

        np.random.seed(random_state)

        self.best_position = None
        self.best_fitness = None
        self.fitness_history: list[float] = []

    # ------------------------------------------------------------------ #
    def _fitness_function(self, particles: np.ndarray) -> np.ndarray:
        n = particles.shape[0]
        costs = np.zeros(n)
        skf = StratifiedKFold(
            n_splits=self.cv_folds, shuffle=True, random_state=self.random_state,
        )
        for i in range(n):
            try:
                p = _decode_particle(particles[i])
                scores = []
                for tr_idx, va_idx in skf.split(self.X_train, self.y_train):
                    ann = BaselineANN(
                        hidden_layer_sizes=self.base_params["hidden_layer_sizes"],
                        activation=self.base_params.get("activation", "relu"),
                        learning_rate_init=p["learning_rate"],
                        alpha=p["alpha"],
                        momentum=p["momentum"],
                        max_iter=self.eval_max_iter,
                        random_state=self.random_state,
                        early_stopping=True,
                    )
                    ann.train(self.X_train[tr_idx], self.y_train[tr_idx],
                              verbose=False)
                    scores.append(accuracy_score(
                        self.y_train[va_idx],
                        ann.predict(self.X_train[va_idx])))
                costs[i] = -float(np.mean(scores))  # PSO minimises
            except Exception:
                costs[i] = 0.0
        return costs

    # ------------------------------------------------------------------ #
    def optimize(self, initial_ann: BaselineANN | None = None,
                 verbose: bool = True):
        print("=" * 60)
        print("PARTICLE SWARM OPTIMIZATION (CV-fitness)")
        print(f"Particles: {self.n_particles} | Iters: {self.iterations} | "
              f"CV: {self.cv_folds}-fold")
        print(f"Base arch: {self.base_params['hidden_layer_sizes']}  "
              f"act: {self.base_params.get('activation', 'relu')}")
        print("=" * 60)

        init_pos = None
        if initial_ann is not None:
            lr  = initial_ann.model.learning_rate_init
            alp = max(initial_ann.model.alpha, 1e-8)
            mom = getattr(initial_ann.model, "momentum", 0.9)
            seed = np.array([
                np.clip((np.log10(lr) + 3.5) / 2.0, 0.01, 0.99),
                np.clip((np.log10(alp) + 5) / 4.0, 0.01, 0.99),
                np.clip((mom - 0.5) / 0.49, 0.01, 0.99),
            ])
            init_pos = np.tile(seed, (self.n_particles, 1))
            init_pos += np.random.normal(0, 0.05, init_pos.shape)
            init_pos = np.clip(init_pos, 0.01, 0.99)

        bounds = (np.zeros(self.DIMENSION), np.ones(self.DIMENSION))
        optimizer = GlobalBestPSO(
            n_particles=self.n_particles,
            dimensions=self.DIMENSION,
            options=self.options,
            bounds=bounds,
            init_pos=init_pos,
        )
        cost, pos = optimizer.optimize(
            self._fitness_function, iters=self.iterations, verbose=verbose,
        )
        self.best_position = pos
        self.best_fitness = -cost
        self.fitness_history = [-c for c in optimizer.cost_history]

        best = _decode_particle(pos)
        print("=" * 60)
        print(f"PSO complete. Best CV accuracy: {self.best_fitness:.4f}")
        print(f"Best params: lr={best['learning_rate']:.5f}, "
              f"α={best['alpha']:.1e}, momentum={best['momentum']:.3f}")
        print("=" * 60)

        return {
            "best_position": pos,
            "best_fitness": self.best_fitness,
            "fitness_history": self.fitness_history,
            "best_params": best,
        }

    # ------------------------------------------------------------------ #
    def get_best_params(self) -> dict[str, float]:
        if self.best_position is None:
            raise ValueError("Run optimize() first.")
        return _decode_particle(self.best_position)

    def get_optimized_ann(self, max_iter: int = 500) -> BaselineANN:
        if self.best_position is None:
            raise ValueError("Run optimize() first.")
        p = _decode_particle(self.best_position)
        return BaselineANN(
            hidden_layer_sizes=self.base_params["hidden_layer_sizes"],
            activation=self.base_params.get("activation", "relu"),
            learning_rate_init=p["learning_rate"],
            alpha=p["alpha"],
            momentum=p["momentum"],
            max_iter=max_iter,
            random_state=self.random_state,
            early_stopping=True,
        )


if __name__ == "__main__":
    from data_preprocessing import HeartDiseasePreprocessor
    pp = HeartDiseasePreprocessor()
    X_tr, X_te, y_tr, y_te, _ = pp.preprocess_pipeline()
    pso = PSOOptimizer(X_tr, y_tr, n_particles=10, iterations=10)
    pso.optimize()
    ann = pso.get_optimized_ann()
    ann.train(X_tr, y_tr)
    ann.evaluate(X_te, y_te, model_name="PSO-OPTIMIZED ANN")
