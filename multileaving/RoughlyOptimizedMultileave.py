# -*- coding: utf-8 -*-

from OptimizedMultileave import OptimizedMultileave
from itertools import product
import numpy as np
import pulp


class RoughlyOptimizedMultileave(OptimizedMultileave):

    def __init__(self, num_data_features, n_samples=10, k=10, bias_weight=1.0):
        self._name = 'Roughly Optimized Multileave'
        self._n_samples = n_samples
        self._k = k
        self._bias_weight = bias_weight
        self.needs_inverted = True
        self.needs_descending = True
        self.needs_oracle = False
        self.vector_aggregation = False

    def _sensitivity(self, C):
        # Compute the mean of each multileaved list
        num_lists, len_ranking, num_rankers = C.shape
        biased_C = C * np.reshape(
            1.0 / (np.arange(len_ranking) + 1),  # Click probability
            (1, len_ranking, 1),
        )
        mu = np.sum(biased_C, axis=(1, 2)) / num_rankers
        mu = np.reshape(mu, (num_lists, 1))

        # Compute the variance
        list_C = np.sum(biased_C, axis=1)  # C for entire multileaved list
        return np.sum((list_C - mu) ** 2, axis=1)

    def _cumulate(self, C):
        return np.cumsum(C, axis=1)

    def _solve(self, C):
        num_lists, len_ranking, num_rankers = C.shape

        # Problem
        prob = pulp.LpProblem('ROM', pulp.LpMinimize)

        # Variables and boundaries
        probabilities, lambdas = [], []
        for l in range(num_lists):
            probabilities.append(pulp.LpVariable('p_%i' % l,
                                                 lowBound=0.0,
                                                 upBound=1.0))
        for i in range(len_ranking):
            lambdas.append(pulp.LpVariable('lambda_%i' % i,
                                           lowBound=0.0))

        # Objective function
        terms = []
        for (s, p) in zip(self._sensitivity(C), probabilities):
            terms.append(s * p)
        for l in lambdas:
            terms.append(self._bias_weight * l)
        prob += pulp.lpSum(terms)

        # Inequarity constraints
        credits = self._cumulate(C)
        for i, ranker_a, ranker_b in product(range(len_ranking),
                                             range(num_rankers),
                                             range(num_rankers)):
            if ranker_a == ranker_b:
                continue
            terms = []
            biases = credits[:, i, ranker_a] - credits[:, i, ranker_b]
            for (b, p) in zip(biases, probabilities):
                terms.append(b * p)
            label = "%ivs.%i@%i" % (ranker_a, ranker_b, i + 1)
            prob += pulp.lpSum(terms) <= lambdas[i], label

        # Equarity constraint
        prob += pulp.lpSum(probabilities) == 1.0

        # Solve!
        prob.solve()

        return (
            np.array([p.varValue for p in probabilities]),
            np.array([l.varValue for l in lambdas]),
        )

    def make_multileaving(self, descending_rankings, inverted_rankings):
        self.create_possible_lists(descending_rankings)

        # L x k x r
        C = []
        for ml in self._allowed_leavings:
            C.append(1./(inverted_rankings[:,ml].T+1))
        C = np.array(C)

        probs = self._solve(C)[0]
        probs[probs < 0] = 0

        if np.all(probs==0):
            choice = np.random.choice(np.arange(len(self._allowed_leavings)))
            self._credits = np.zeros(C[0].shape)
        else:
            choice = np.random.choice(np.arange(len(self._allowed_leavings)), p=probs/np.sum(probs))

        self._credits = C[choice]

        return self._allowed_leavings[choice]
