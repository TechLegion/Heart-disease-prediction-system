"""
Particle Swarm Optimization for ANN Training-Parameter Tuning

Each particle encodes two continuous training hyper-parameters:
  - learning_rate_init  [0.0001 .. 0.1]
  - alpha (L2 penalty)  [0.00001 .. 0.1]

The swarm explores this 2-D space; fitness = validation accuracy of an
ANN trained with those parameters.  Can optionally start from a
pre-trained model's settings (e.g. after GA).

Based on the methodology described in the research proposal.
"""

import numpy as np
from pyswarms.single import GlobalBestPSO
from baseline_ann import BaselineANN
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')


class PSOOptimizer:
    """PSO optimizer for ANN continuous training hyper-parameters."""

    def __init__(self, X_train, y_train, X_val, y_val,
                 n_particles=30, iterations=50,
                 hidden_layer_sizes=(100, 50),
                 options=None, random_state=42):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.n_particles = n_particles
        self.iterations = iterations
        self.hidden_layer_sizes = hidden_layer_sizes
        self.options = options or {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
        self.random_state = random_state

        np.random.seed(random_state)

        self.best_position = None
        self.best_fitness = None
        self.fitness_history = []
        self.dimension = 2  # learning_rate gene, alpha gene

    # ------------------------------------------------------------------ #
    #  Particle <-> hyper-parameter mapping                               #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _decode(particle):
        lr    = 10 ** (-3.3 + np.clip(particle[0], 0, 1) * 1.6)  # [~0.0005, ~0.02]
        alpha = 10 ** (-5 + np.clip(particle[1], 0, 1) * 4)      # [1e-5, 0.1]
        return lr, alpha

    # ------------------------------------------------------------------ #
    #  Fitness function (called by pyswarms for all particles at once)    #
    # ------------------------------------------------------------------ #
    def _fitness_function(self, particles):
        n = particles.shape[0]
        costs = np.zeros(n)
        for i in range(n):
            try:
                lr, alpha = self._decode(particles[i])
                ann = BaselineANN(
                    hidden_layer_sizes=self.hidden_layer_sizes,
                    max_iter=300,
                    learning_rate_init=lr,
                    random_state=self.random_state,
                    early_stopping=True,
                )
                ann.model.alpha = alpha
                ann.train(self.X_train, self.y_train)
                costs[i] = -accuracy_score(self.y_val, ann.predict(self.X_val))
            except Exception:
                costs[i] = 0.0  # worst (PSO minimises; 0 > -acc)
        return costs

    # ------------------------------------------------------------------ #
    #  Main optimisation                                                  #
    # ------------------------------------------------------------------ #
    def optimize(self, initial_ann=None, verbose=True):
        print("=" * 60)
        print("PARTICLE SWARM OPTIMIZATION")
        print("=" * 60)
        print(f"Particles: {self.n_particles}  |  Iterations: {self.iterations}")
        print(f"Architecture: {self.hidden_layer_sizes}")
        print(f"Options: {self.options}")
        print("Optimizing: learning_rate, alpha (L2 regularisation)")
        print("=" * 60)

        if initial_ann is not None:
            self.hidden_layer_sizes = initial_ann.model.hidden_layer_sizes

        # Build initial swarm positions near the starting model
        init_pos = None
        if initial_ann is not None:
            lr  = initial_ann.model.learning_rate_init
            alp = initial_ann.model.alpha
            lr_gene  = np.clip((np.log10(lr) + 3.3) / 1.6, 0, 1)
            alp_gene = np.clip((np.log10(alp) + 5) / 4, 0, 1)
            init_pos = np.tile([lr_gene, alp_gene], (self.n_particles, 1))
            init_pos += np.random.normal(0, 0.03, init_pos.shape)
            init_pos = np.clip(init_pos, 0.01, 0.99)

        bounds = (np.zeros(self.dimension), np.ones(self.dimension))

        optimizer = GlobalBestPSO(
            n_particles=self.n_particles,
            dimensions=self.dimension,
            options=self.options,
            bounds=bounds,
            init_pos=init_pos,
        )

        cost, pos = optimizer.optimize(
            self._fitness_function,
            iters=self.iterations,
            verbose=verbose,
        )

        self.best_position = pos
        self.best_fitness = -cost
        self.fitness_history = [-c for c in optimizer.cost_history]

        lr, alpha = self._decode(pos)
        print("=" * 60)
        print(f"PSO complete.  Best accuracy: {self.best_fitness:.4f}")
        print(f"Best params : lr={lr:.6f}, alpha={alpha:.6f}")
        print("=" * 60)

        return {
            'best_position': self.best_position,
            'best_fitness': self.best_fitness,
            'fitness_history': self.fitness_history,
            'best_params': {'learning_rate': lr, 'alpha': alpha},
        }

    # ------------------------------------------------------------------ #
    #  Retrieve result                                                    #
    # ------------------------------------------------------------------ #
    def get_optimized_ann(self):
        """Return an *untrained* BaselineANN configured with the best params."""
        if self.best_position is None:
            raise ValueError("Run optimize() first.")
        lr, alpha = self._decode(self.best_position)
        ann = BaselineANN(
            hidden_layer_sizes=self.hidden_layer_sizes,
            max_iter=500,
            learning_rate_init=lr,
            random_state=self.random_state,
        )
        ann.model.alpha = alpha
        return ann

    def get_best_params(self):
        if self.best_position is None:
            raise ValueError("Run optimize() first.")
        lr, alpha = self._decode(self.best_position)
        return {'learning_rate': lr, 'alpha': alpha}


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

        initial = BaselineANN(hidden_layer_sizes=(100, 50), max_iter=200)
        initial.train(X_train, y_train)

        pso = PSOOptimizer(X_train, y_train, X_val, y_val,
                           n_particles=15, iterations=10)
        pso.optimize(initial_ann=initial)
        ann = pso.get_optimized_ann()
        ann.train(X_train_full, y_train_full)
        ann.evaluate(X_test, y_test, model_name="PSO-OPTIMIZED ANN")
    except Exception as e:
        import traceback; traceback.print_exc()
