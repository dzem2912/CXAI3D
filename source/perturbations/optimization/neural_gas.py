import numpy as np


class NeuralGas:
    def __init__(self, num_rep_points: int,
                 max_iterations: int,
                 initial_learning_rate: float = 0.1, lambda_decay: float = 0.99):
        self.num_rep_points = num_rep_points
        self.max_iterations = max_iterations
        self.learning_rate = initial_learning_rate
        self.lambda_decay = lambda_decay

    def fit(self, segment: np.ndarray):
        # [1.] Initialization: Select M representative points from rand points within S with all features
        indices = np.random.choice(len(segment), self.num_rep_points, replace=False)
        w = segment[indices]

        # [2.] Sampling: For each iteration I, sample a random point x_j from S.
        #                Compute distance from x_j and w_i
        for _ in range(self.max_iterations):
            point_index = np.random.randint(0, len(segment))
            x_j = segment[point_index]
            distances = np.linalg.norm(w - x_j, axis=1)

            # [3.] Ranking: Rank distances, smallest corresponding to w_0^i (closest feature vector).
            #               Let k represent the rank index, where k = 0 for the closest vector.
            ranked_indices = np.argsort(distances)

            # [4.] Updating: Each feature vector is updated s: w_i = w_i + \epsilon * h * d(x_j, w_i)
            for k, w_i_index in enumerate(ranked_indices):
                h = np.exp(- k / (self.num_rep_points * self.lambda_decay))
                w[w_i_index] += self.learning_rate * h * (x_j - w[w_i_index])
            self.learning_rate *= self.lambda_decay

        return w
