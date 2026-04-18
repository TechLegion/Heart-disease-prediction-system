"""
Genetic Algorithm Optimizer for ANN Hyperparameter Optimization.

Improvements over v1:
  * Larger search space (5 genes): learning_rate, hidden_layer_1,
    hidden_layer_2, alpha (L2), activation.
  * K-fold CV fitness instead of single-split validation accuracy — much
    lower variance, which is the main reason the old Hybrid pipeline used
    to underperform PSO-alone.
  * Elitism + top-K retrieval so the Hybrid stage can pick the best
    architectures (not just one) before PSO fine-tuning.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
from deap import base, creator, tools
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

from baseline_ann import BaselineANN
import warnings
warnings.filterwarnings("ignore")


ACTIVATIONS = ["relu", "tanh", "logistic"]


def _decode_individual(individual) -> dict[str, Any]:
    """Map [0, 1] genes to concrete hyperparameters."""
    g = [float(np.clip(v, 0.0, 1.0)) for v in individual]
    lr     = 10 ** (-3.5 + g[0] * 2.0)                    # [~3e-4, ~3e-2]
    h1     = int(32 + g[1] * 192)                         # [32, 224]
    h2     = int(16 + g[2] * 112)                         # [16, 128]
    alpha  = 10 ** (-5 + g[3] * 4)                        # [1e-5, 1e-1]
    act    = ACTIVATIONS[min(int(g[4] * len(ACTIVATIONS)),
                             len(ACTIVATIONS) - 1)]
    return {
        "learning_rate": lr,
        "hidden_layer_sizes": (h1, h2),
        "alpha": alpha,
        "activation": act,
    }


def build_ann_from_params(params: dict[str, Any], random_state: int,
                          max_iter: int = 500) -> BaselineANN:
    """Instantiate an (untrained) BaselineANN from a decoded parameter dict."""
    return BaselineANN(
        hidden_layer_sizes=params["hidden_layer_sizes"],
        max_iter=max_iter,
        learning_rate_init=params["learning_rate"],
        alpha=params["alpha"],
        activation=params["activation"],
        random_state=random_state,
        early_stopping=True,
    )


class GAOptimizer:
    """GA over 5 continuous-in-[0,1] genes; fitness = CV accuracy."""

    GENE_COUNT = 5

    def __init__(self, X_train, y_train, X_val=None, y_val=None,
                 n_population: int = 30, n_generations: int = 25,
                 crossover_prob: float = 0.7, mutation_prob: float = 0.25,
                 cv_folds: int = 3, eval_max_iter: int = 200,
                 elitism: int = 2, random_state: int = 42):
        self.X_train = X_train
        self.y_train = y_train
        # X_val / y_val are accepted for API compatibility with v1 but no
        # longer drive fitness — we use CV on the training set.
        self.X_val, self.y_val = X_val, y_val

        self.n_population = n_population
        self.n_generations = n_generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.cv_folds = cv_folds
        self.eval_max_iter = eval_max_iter
        self.elitism = elitism
        self.random_state = random_state

        random.seed(random_state)
        np.random.seed(random_state)

        self._setup_deap()
        self._final_population = None
        self.best_individual = None
        self.best_fitness = None
        self.fitness_history = []

    # ------------------------------------------------------------------ #
    def _setup_deap(self):
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMax)

        self.toolbox = base.Toolbox()
        self.toolbox.register("attr_gene", random.random)
        self.toolbox.register(
            "individual", tools.initRepeat,
            creator.Individual, self.toolbox.attr_gene, n=self.GENE_COUNT,
        )
        self.toolbox.register("population", tools.initRepeat, list,
                              self.toolbox.individual)
        self.toolbox.register("evaluate", self._evaluate_individual)
        self.toolbox.register("mate", tools.cxBlend, alpha=0.5)
        self.toolbox.register("mutate", self._mutate_individual, indpb=0.3)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    @staticmethod
    def _mutate_individual(individual, indpb):
        for i in range(len(individual)):
            if random.random() < indpb:
                individual[i] += random.gauss(0, 0.1)
                individual[i] = max(0.0, min(1.0, individual[i]))
        return (individual,)

    # ------------------------------------------------------------------ #
    def _evaluate_individual(self, individual) -> tuple[float]:
        try:
            params = _decode_individual(individual)
        except Exception:
            return (0.0,)
        try:
            skf = StratifiedKFold(
                n_splits=self.cv_folds, shuffle=True,
                random_state=self.random_state,
            )
            scores = []
            X_arr = np.asarray(self.X_train)
            y_arr = np.asarray(self.y_train)
            for tr_idx, va_idx in skf.split(X_arr, y_arr):
                ann = build_ann_from_params(
                    params, self.random_state, max_iter=self.eval_max_iter)
                ann.train(X_arr[tr_idx], y_arr[tr_idx], verbose=False)
                scores.append(accuracy_score(
                    y_arr[va_idx], ann.predict(X_arr[va_idx])))
            return (float(np.mean(scores)),)
        except Exception:
            return (0.0,)

    # ------------------------------------------------------------------ #
    def optimize(self, verbose: bool = True):
        print("=" * 60)
        print("GENETIC ALGORITHM OPTIMIZATION (CV-fitness)")
        print(f"Pop: {self.n_population} | Gens: {self.n_generations} | "
              f"CV: {self.cv_folds}-fold | Elitism: {self.elitism}")
        print("=" * 60)

        population = self.toolbox.population(n=self.n_population)
        for ind, fit in zip(population, map(self.toolbox.evaluate, population)):
            ind.fitness.values = fit

        best = max(ind.fitness.values[0] for ind in population)
        self.fitness_history.append(best)
        if verbose:
            print(f"Gen 0: best CV acc = {best:.4f}")

        for gen in range(1, self.n_generations + 1):
            elites = tools.selBest(population, self.elitism)
            elites = [self.toolbox.clone(e) for e in elites]

            offspring = self.toolbox.select(
                population, len(population) - self.elitism)
            offspring = [self.toolbox.clone(o) for o in offspring]

            for c1, c2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.crossover_prob:
                    self.toolbox.mate(c1, c2)
                    del c1.fitness.values
                    del c2.fitness.values

            for mut in offspring:
                if random.random() < self.mutation_prob:
                    self.toolbox.mutate(mut)
                    del mut.fitness.values

            invalids = [i for i in offspring if not i.fitness.valid]
            for ind, fit in zip(invalids, map(self.toolbox.evaluate, invalids)):
                ind.fitness.values = fit

            population[:] = elites + offspring
            best = max(ind.fitness.values[0] for ind in population)
            self.fitness_history.append(best)

            if verbose and (gen % 5 == 0 or gen == self.n_generations):
                top = tools.selBest(population, 1)[0]
                params = _decode_individual(top)
                print(f"Gen {gen}: best CV acc = {best:.4f}  "
                      f"(lr={params['learning_rate']:.5f}, "
                      f"arch={params['hidden_layer_sizes']}, "
                      f"α={params['alpha']:.1e}, act={params['activation']})")

        self._final_population = population
        self.best_individual = tools.selBest(population, 1)[0]
        self.best_fitness = self.best_individual.fitness.values[0]

        params = _decode_individual(self.best_individual)
        print("=" * 60)
        print(f"GA complete. Best CV accuracy: {self.best_fitness:.4f}")
        print(f"Best params: {params}")
        print("=" * 60)

        return {
            "best_individual": self.best_individual,
            "best_fitness": self.best_fitness,
            "fitness_history": self.fitness_history,
            "best_params": params,
        }

    # ------------------------------------------------------------------ #
    def get_best_params(self) -> dict[str, Any]:
        if self.best_individual is None:
            raise ValueError("Run optimize() first.")
        return _decode_individual(self.best_individual)

    def get_topk_params(self, k: int = 3) -> list[dict[str, Any]]:
        """Return the K best *distinct* parameter dicts in the final population."""
        if self._final_population is None:
            raise ValueError("Run optimize() first.")
        sorted_pop = sorted(
            self._final_population,
            key=lambda ind: ind.fitness.values[0], reverse=True,
        )
        seen: set = set()
        result: list[dict[str, Any]] = []
        for ind in sorted_pop:
            params = _decode_individual(ind)
            key = (params["hidden_layer_sizes"], params["activation"])
            if key in seen:
                continue
            seen.add(key)
            result.append(params)
            if len(result) >= k:
                break
        return result

    def get_optimized_ann(self) -> BaselineANN:
        params = self.get_best_params()
        return build_ann_from_params(params, self.random_state, max_iter=500)


if __name__ == "__main__":
    from data_preprocessing import HeartDiseasePreprocessor
    pp = HeartDiseasePreprocessor()
    X_tr, X_te, y_tr, y_te, _ = pp.preprocess_pipeline()
    ga = GAOptimizer(X_tr, y_tr, n_population=15, n_generations=10)
    ga.optimize()
    ann = ga.get_optimized_ann()
    ann.train(X_tr, y_tr)
    ann.evaluate(X_te, y_te, model_name="GA-OPTIMIZED ANN")
