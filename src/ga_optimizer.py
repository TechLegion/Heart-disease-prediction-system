"""
Genetic Algorithm Optimizer for ANN Hyperparameter Optimization

Evolves ANN architecture and learning rate using selection, crossover,
and mutation.  Each chromosome encodes three genes that map to:
  - learning_rate_init  [0.0001 .. 0.1]
  - hidden-layer-1 size [20 .. 200]
  - hidden-layer-2 size [10 .. 100]

Based on the methodology described in the research proposal.
"""

import numpy as np
import random
from deap import base, creator, tools
from baseline_ann import BaselineANN
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')


class GAOptimizer:
    """GA optimizer that evolves ANN hyperparameters."""

    def __init__(self, X_train, y_train, X_val, y_val,
                 n_population=50, n_generations=30,
                 crossover_prob=0.7, mutation_prob=0.2,
                 random_state=42):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.n_population = n_population
        self.n_generations = n_generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.random_state = random_state

        random.seed(random_state)
        np.random.seed(random_state)

        self._setup_deap()

        self.best_individual = None
        self.best_fitness = None
        self.fitness_history = []

    # ------------------------------------------------------------------ #
    #  Gene <-> hyperparameter mapping                                    #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _decode(individual):
        """Map [0, 1] genes to concrete hyperparameters."""
        lr = 10 ** (-3.3 + np.clip(individual[0], 0, 1) * 1.6)  # [~0.0005, ~0.02]
        h1 = int(50 + np.clip(individual[1], 0, 1) * 100)       # [50, 150]
        h2 = int(20 + np.clip(individual[2], 0, 1) * 55)        # [20, 75]
        return lr, (h1, h2)

    # ------------------------------------------------------------------ #
    #  DEAP setup                                                         #
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
            creator.Individual, self.toolbox.attr_gene, n=3,
        )
        self.toolbox.register("population", tools.initRepeat, list,
                              self.toolbox.individual)
        self.toolbox.register("evaluate", self._evaluate_individual)
        self.toolbox.register("mate", tools.cxBlend, alpha=0.5)
        self.toolbox.register("mutate", self._mutate_individual, indpb=0.3)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    # ------------------------------------------------------------------ #
    #  Fitness evaluation                                                 #
    # ------------------------------------------------------------------ #
    def _evaluate_individual(self, individual):
        try:
            lr, hidden = self._decode(individual)
            ann = BaselineANN(
                hidden_layer_sizes=hidden,
                max_iter=300,
                learning_rate_init=lr,
                random_state=self.random_state,
                early_stopping=True,
            )
            ann.train(self.X_train, self.y_train)
            return (accuracy_score(self.y_val, ann.predict(self.X_val)),)
        except Exception:
            return (0.0,)

    def _individual_to_ann(self, individual):
        """Return an *untrained* BaselineANN with the decoded hyper-params."""
        lr, hidden = self._decode(individual)
        return BaselineANN(
            hidden_layer_sizes=hidden,
            max_iter=500,
            learning_rate_init=lr,
            random_state=self.random_state,
        )

    @staticmethod
    def _mutate_individual(individual, indpb):
        for i in range(len(individual)):
            if random.random() < indpb:
                individual[i] += random.gauss(0, 0.1)
                individual[i] = max(0.0, min(1.0, individual[i]))
        return (individual,)

    # ------------------------------------------------------------------ #
    #  Main optimisation loop                                             #
    # ------------------------------------------------------------------ #
    def optimize(self, verbose=True):
        print("=" * 60)
        print("GENETIC ALGORITHM OPTIMIZATION")
        print("=" * 60)
        print(f"Population: {self.n_population}  |  Generations: {self.n_generations}")
        print(f"Crossover: {self.crossover_prob}  |  Mutation: {self.mutation_prob}")
        print("Optimizing: learning_rate, hidden_layer_1, hidden_layer_2")
        print("=" * 60)

        population = self.toolbox.population(n=self.n_population)

        fitnesses = list(map(self.toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        best = max(ind.fitness.values[0] for ind in population)
        self.fitness_history.append(best)
        if verbose:
            print(f"Gen 0: best = {best:.4f}")

        for gen in range(1, self.n_generations + 1):
            offspring = self.toolbox.select(population, len(population))
            offspring = list(map(self.toolbox.clone, offspring))

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
            fits = map(self.toolbox.evaluate, invalids)
            for ind, fit in zip(invalids, fits):
                ind.fitness.values = fit

            population[:] = offspring
            best = max(ind.fitness.values[0] for ind in population)
            self.fitness_history.append(best)

            if verbose and (gen % 5 == 0 or gen == self.n_generations):
                lr, hidden = self._decode(tools.selBest(population, 1)[0])
                print(f"Gen {gen}: best = {best:.4f}  "
                      f"(lr={lr:.5f}, arch={hidden})")

        self.best_individual = tools.selBest(population, 1)[0]
        self.best_fitness = self.best_individual.fitness.values[0]

        lr, hidden = self._decode(self.best_individual)
        print("=" * 60)
        print(f"GA complete.  Best accuracy: {self.best_fitness:.4f}")
        print(f"Best params : lr={lr:.5f}, architecture={hidden}")
        print("=" * 60)

        return {
            'best_individual': self.best_individual,
            'best_fitness': self.best_fitness,
            'fitness_history': self.fitness_history,
            'best_params': {'learning_rate': lr,
                            'hidden_layer_sizes': hidden},
        }

    # ------------------------------------------------------------------ #
    #  Retrieve result                                                    #
    # ------------------------------------------------------------------ #
    def get_optimized_ann(self):
        if self.best_individual is None:
            raise ValueError("Run optimize() first.")
        return self._individual_to_ann(self.best_individual)

    def get_best_params(self):
        if self.best_individual is None:
            raise ValueError("Run optimize() first.")
        lr, hidden = self._decode(self.best_individual)
        return {'learning_rate': lr, 'hidden_layer_sizes': hidden}


if __name__ == "__main__":
    from data_preprocessing import HeartDiseasePreprocessor
    from sklearn.model_selection import train_test_split

    preprocessor = HeartDiseasePreprocessor()
    try:
        X_train_full, X_test, y_train_full, y_test, _ = \
            preprocessor.preprocess_pipeline()
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=0.2,
            random_state=42, stratify=y_train_full)

        ga = GAOptimizer(X_train, y_train, X_val, y_val,
                         n_population=20, n_generations=10)
        ga.optimize()
        ann = ga.get_optimized_ann()
        ann.train(X_train_full, y_train_full)
        ann.evaluate(X_test, y_test, model_name="GA-OPTIMIZED ANN")
    except Exception as e:
        import traceback; traceback.print_exc()
