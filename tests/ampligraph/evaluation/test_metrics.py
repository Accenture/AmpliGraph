# Copyright 2019 The AmpliGraph Authors. All Rights Reserved.
#
# This file is Licensed under the Apache License, Version 2.0.
# A copy of the Licence is available in LICENCE, or at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
import numpy as np
from ampligraph.evaluation.metrics import rank_score, mrr_score, hits_at_n_score, mr_score


def test_rank_score():
    y_pred = np.array([.434, .65, .21, .84])
    y_true = np.array([0, 0, 1, 0])
    rank_actual = rank_score(y_true, y_pred)
    assert rank_actual == 4


def test_mrr_score():
    y_pred_true = np.array([[[0, 1, 0], [.32, .84, .73]],
                            [[0, 1, 0], [.66, .11, .33]]])

    rankings = []
    for y_pred_true_k in y_pred_true:
        rankings.append(rank_score(y_pred_true_k[0], y_pred_true_k[1]))
    mrr_actual = mrr_score(rankings)
    np.testing.assert_almost_equal(mrr_actual, 0.66666, decimal=5)


def test_hits_at_n_score():
    y_pred_true = np.array([[[0, 1, 0], [.32, .84, .73]],
                            [[0, 1, 0], [.66, .11, .33]]])
    rankings = []
    for y_pred_true_k in y_pred_true:
        rankings.append(rank_score(y_pred_true_k[0], y_pred_true_k[1]))
    hits_actual = hits_at_n_score(rankings, n=2)
    assert hits_actual == 0.5


def test_mr_score():
    rank = np.array([.2, .4, .6, .8])
    mr = mr_score(rank)
    assert mr == 0.5
