import numpy as np
from sklearn.datasets import make_spd_matrix
from scipy.stats import multivariate_normal
import sys


class GMM:
    def __init__(self, dims, n_comps):
        """
        GMM model constructor
        :param dims: number of dimensions of the input data
        :param n_comps: number of components
        """
        self.n_comps = n_comps
        self.means = np.array([np.random.random(dims) for _ in range(n_comps)])
        self.covs = np.array([make_spd_matrix(dims) for _ in range(n_comps)])
        self.priors = np.random.dirichlet(np.ones(n_comps))

    def _expectation(self, X):
        """
        Performs an E-step (Expectation step)
        :param X: input data of dim self.dims
        :return: preds: posterior distributions of components,
                log_likelihood: the log-likelihood
        """
        # probs is π_k * f(x), where f(x) is the probability of X
        # based on the Gaussian i with mean[i] and cov[i],
        # and π_k is the prior probability of each Gaussian
        probs = []
        for i in range(self.n_comps):
            mean, cov, prior = self.means[i], self.covs[i], self.priors[i]
            probs.append(np.expand_dims((prior * multivariate_normal.pdf(X, mean, cov)), -1))
        probs = np.hstack(probs)
        log_likelihood = np.log(probs.sum(1)).sum()
        # preds are the posterior distributions of the n_comp components
        preds = probs / probs.sum(1, keepdims=True)
        return preds, log_likelihood

    def _maximization(self, X, comp_posteriors):
        """
        Performs an M-step (Maximization step)
        :param X: input data of dim self.dims
        :param comp_posteriors: posterior distributions of components
        """
        new_means, new_covs, new_priors = [], [], []
        effective_n_points = comp_posteriors.sum(0)
        for i in range(self.n_comps):
            comp_posteriors_expanded = np.expand_dims(comp_posteriors[:, i], -1)

            new_mean = (comp_posteriors_expanded * X).sum(0) \
                       / effective_n_points[i]
            new_means.append(new_mean)

            mean_shifted = X - new_mean
            new_cov = np.dot((comp_posteriors_expanded * mean_shifted).T, mean_shifted) \
                      / effective_n_points[i]
            new_covs.append(new_cov)

            new_prior = effective_n_points[i] / effective_n_points.sum()
            new_priors.append(new_prior)

        self.means, self.covs, self.priors = np.array(new_means), \
                                             np.array(new_covs), \
                                             np.array(new_priors)

    def train(self, X, epochs, epsilon):
        """
        Trains the model for a specified number of epochs or
        until the log-likelihood changes less than an epsilon value
        :param X: input data of dim self.dims
        :param epochs: number of epochs for the model to be trained
        :param epsilon: epsilon log-likelihood threshold.
        When the log-likelihood changes less than epsilon in an epoch, training will halt.
        """
        prev_log_likelihood = sys.float_info.max
        for i in range(epochs):
            # Expectation step
            comp_posteriors, log_likelihood = self._expectation(X)
            # Evaluate the log likelihood
            if abs(prev_log_likelihood - log_likelihood) < epsilon:
                break
            # Maximization step
            self._maximization(X, comp_posteriors)
            prev_log_likelihood = log_likelihood
            if i % 10 == 0:
                print("Epoch {}: Log-likelihood {}".format(i + 1, log_likelihood))

    def predict(self, X):
        """
        Returns the mean of the component that maximizes the posterior probability
        for each point in X
        :param X: input data of dim self.dims
        :return: component means
        """
        comp_posteriors = self._expectation(X)[0]
        X_preds = np.argmax(comp_posteriors, axis=1)
        return self.means[X_preds]
