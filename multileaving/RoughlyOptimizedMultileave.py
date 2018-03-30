# -*- coding: utf-8 -*-

from interleaving.roughly_optimized import RoughlyOptimized
from OptimizedMultileave import OptimizedMultileave
import numpy as np
import pulp
import time


class RoughlyOptimizedMultileave(OptimizedMultileave):

    def __init__(self, num_data_features, k=10, bias_weight=1.0):
        self._name = 'Roughly Optimized Multileave'
        self._k = k
        self._bias_weight = bias_weight
        self.needs_inverted = True
        self.needs_descending = True
        self.needs_oracle = False
        self.vector_aggregation = False

    def convert_input(self, rankings, C):
        '''
            Input:
                rankings: {team_id => {rank => doc_id}}
                C:        {list_id => {rank => {team_id => credit}}}
                lists:    {list_id => {rank => doc_id}}
            Output:
                xlists:    {team_id => {rank => doc_id}}
                xrankings: {list_id => xranking}
                    xranking: {rank => doc_id}
                    xranking.credits: {team_id => {doc_id => credit}}
        '''
        class XRanking(list):
            pass

        lists = self._allowed_leavings

        xlists = []
        for ranking in rankings:
            xlists.append(ranking.tolist())
        xrankings = []
        assert len(C) == len(lists)
        for c, l in zip(C, lists):
            xranking = XRanking()
            xranking += l
            xcredits = {}
            for rank, doc_id in enumerate(l):
                for team_id, credit in enumerate(c[rank]):
                    if team_id not in xcredits:
                        xcredits[team_id] = {}
                    assert doc_id not in xcredits[team_id]
                    xcredits[team_id][doc_id] = credit
            xranking.credits = xcredits
            xrankings.append(xranking)
        return (xlists, xrankings)


    def make_multileaving(self, descending_rankings, inverted_rankings):
        self.create_possible_lists(descending_rankings)

        # L x k x r
        C = []
        for ml in self._allowed_leavings:
            C.append(1./(inverted_rankings[:,ml].T+1))

        n_docs = descending_rankings.shape[1]
        n_rankers = descending_rankings.shape[0]

        length = min(self._k,n_docs)

        # --

        lists, rankings = self.convert_input(descending_rankings, C)
        ro = RoughlyOptimized(lists, length, 10)
        _, ps, _ = ro._compute_probabilities(lists, rankings)

        # --

        probs = np.array(ps)
        probs[probs < 0] = 0

        if np.all(probs==0):
            choice = np.random.choice(np.arange(len(self._allowed_leavings)))
            self._credits = np.zeros(C[0].shape)
        else:
            choice = np.random.choice(np.arange(len(self._allowed_leavings)), p=probs/np.sum(probs))

        self._credits = C[choice]

        return self._allowed_leavings[choice]
