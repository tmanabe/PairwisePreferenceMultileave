# -*- coding: utf-8 -*-

from multileaving.RoughlyOptimizedMultileave import RoughlyOptimizedMultileave as ROM
from numpy import array
from numpy import allclose
from utils.rankings import invert_rankings


class TestRoughlyOptimized(object):

    C = array([  # C[list_id, rank, team_id] == credit
        [
            [1.0,   1.0/3],
            [0.5,   1.0],
        ],
        [
            [0.5,   1.0],
            [1.0,   1.0/3],
        ],
        [
            [0.5,   1.0],
            [1.0/3, 0.5],
        ],
    ])

    C2 = array([
        [
            [1.0,   1.0/3],
            [0.5,   0.0],
        ],
        [
            [0.5,   1.0/3],
            [1.0,   0.5],
        ],
        [
            [0.5,   1.0/3],
            [1.0/3, 0.0],
        ],
    ])

    def test__sensitivity(self):
        # Ref: https://github.com/mpkato/interleaving/blob/master/tests/test_optimized.py#L98-L116
        expect = array([
            (1 + 0.5 * 0.5 - (1.75 + 1.0/3) / 2) ** 2 + \
                (1.0/3 + 0.5 - (1.75 + 1.0/3) / 2) ** 2,
            (0.5 + 0.5 - (2.0 + 0.5/3) / 2) ** 2 + \
                (1.0 + 1.0/3 * 0.5 - (2.0 + 0.5/3) / 2) ** 2,
            (0.5 + 0.5 * 1.0/3 - (1.75 + 0.5/3) / 2) ** 2 + \
                (1.0 + 0.5 * 0.5 - (1.75 + 0.5/3) / 2) ** 2,
        ])
        assert allclose(expect, ROM(2, 2)._sensitivity(self.C))

    def test__cumulate(self):
        expect = array([
            [
                [1.0,   1.0/3],
                [1.5,   4.0/3],
            ],
            [
                [0.5,   1.0],
                [1.5,   4.0/3],
            ],
            [
                [0.5,   1.0],
                [5.0/6, 1.5],
            ],
        ])
        assert allclose(expect, ROM(2, 2)._cumulate(self.C))

    def test__solve(self):
        # Ref: https://github.com/mpkato/interleaving/blob/master/tests/test_roughly_optimized.py#L33-L57
        probs, lambdas = ROM(2, 2)._solve(self.C2)
        assert allclose(array([0.0, 0.0, 1.0]), probs)
        assert allclose(
            array([0.5 - 1.0/3, 0.5 - 1.0/3 + 1.0/3 - 0.0]),
            lambdas,
        )
