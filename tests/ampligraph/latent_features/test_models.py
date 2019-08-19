# Copyright 2019 The AmpliGraph Authors. All Rights Reserved.
#
# This file is Licensed under the Apache License, Version 2.0.
# A copy of the Licence is available in LICENCE, or at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
import numpy as np
import pytest

from ampligraph.latent_features import TransE, DistMult, ComplEx, HolE, RandomBaseline
from ampligraph.latent_features import set_entity_threshold, reset_entity_threshold
from ampligraph.datasets import load_wn18
from ampligraph.evaluation import evaluate_performance, hits_at_n_score


def test_large_graph_mode():
    set_entity_threshold(10)
    X = load_wn18()
    model = ComplEx(batches_count=100, seed=555, epochs=1, k=50, loss='multiclass_nll', loss_params={'margin': 5},
                   verbose=True, optimizer='sgd', optimizer_params={'lr': 0.001})
    model.fit(X['train'])
    X_filter = np.concatenate((X['train'], X['valid'], X['test']), axis=0)
    ranks_all = evaluate_performance(X['test'][::1000], model, X_filter, verbose=True, corrupt_side='s+o',
                                 use_default_protocol=True)
    
    y = model.predict(X['test'][:1])
    print(y)
    reset_entity_threshold()
    
    
def test_large_graph_mode_adam():
    set_entity_threshold(10)
    X = load_wn18()
    model = ComplEx(batches_count=100, seed=555, epochs=1, k=50, loss='multiclass_nll', loss_params={'margin': 5},
                   verbose=True, optimizer='adam', optimizer_params={'lr': 0.001})
    try:
        model.fit(X['train'])
    except Exception as e:
        print(str(e))
        
    reset_entity_threshold()
    
    
def test_fit_predict_TransE_early_stopping_with_filter():
    X = load_wn18()
    model = TransE(batches_count=1, seed=555, epochs=7, k=50, loss='pairwise', loss_params={'margin': 5},
                   verbose=True, optimizer='adagrad', optimizer_params={'lr': 0.1})
    X_filter = np.concatenate((X['train'], X['valid'], X['test']))
    model.fit(X['train'], True, {'x_valid': X['valid'][::100], 
                                 'criteria': 'mrr',
                                 'x_filter': X_filter,
                                 'stop_interval': 2, 
                                 'burn_in': 1,
                                 'check_interval': 2})
    
    y = model.predict(X['test'][:1])
    print(y)


def test_fit_predict_TransE_early_stopping_without_filter():
    X = load_wn18()
    model = TransE(batches_count=1, seed=555, epochs=7, k=50, loss='pairwise', loss_params={'margin': 5},
                   verbose=True, optimizer='adagrad', optimizer_params={'lr': 0.1})
    model.fit(X['train'], True, {'x_valid': X['valid'][::100], 
                                 'criteria': 'mrr',
                                 'stop_interval': 2, 
                                 'burn_in': 1,
                                 'check_interval': 2})
    
    y = model.predict(X['test'][:1])
    print(y)


def test_evaluate_RandomBaseline():
    model = RandomBaseline(seed=0)
    X = load_wn18()
    model.fit(X["train"])
    ranks = evaluate_performance(X["test"], 
                                 model=model, 
                                 use_default_protocol=False,
                                 corrupt_side='s+o',
                                 verbose=False)
    hits10 = hits_at_n_score(ranks, n=10)
    hits1 = hits_at_n_score(ranks, n=1)
    assert hits10 < 0.01 and hits1 == 0.0


def test_fit_predict_transE():
    model = TransE(batches_count=1, seed=555, epochs=20, k=10, loss='pairwise', loss_params={'margin': 5}, 
                   optimizer='adagrad', optimizer_params={'lr': 0.1})
    X = np.array([['a', 'y', 'b'],
                  ['b', 'y', 'a'],
                  ['a', 'y', 'c'],
                  ['c', 'y', 'a'],
                  ['a', 'y', 'd'],
                  ['c', 'y', 'd'],
                  ['b', 'y', 'c'],
                  ['f', 'y', 'e']])
    model.fit(X)
    y_pred = model.predict(np.array([['f', 'y', 'e'], ['b', 'y', 'd']]))
    print(y_pred)
    assert y_pred[0] > y_pred[1]


def test_fit_predict_DistMult():
    model = DistMult(batches_count=2, seed=555, epochs=20, k=10, loss='pairwise', loss_params={'margin': 5}, 
                     optimizer='adagrad', optimizer_params={'lr': 0.1})
    X = np.array([['a', 'y', 'b'],
                  ['b', 'y', 'a'],
                  ['a', 'y', 'c'],
                  ['c', 'y', 'a'],
                  ['a', 'y', 'd'],
                  ['c', 'y', 'd'],
                  ['b', 'y', 'c'],
                  ['f', 'y', 'e']])
    model.fit(X)
    y_pred = model.predict(np.array([['f', 'y', 'e'], ['b', 'y', 'd']]))
    print(y_pred)
    assert y_pred[0] > y_pred[1]


def test_fit_predict_CompleEx():
    model = ComplEx(batches_count=1, seed=555, epochs=20, k=10,
                    loss='pairwise', loss_params={'margin': 1}, regularizer='LP',
                    regularizer_params={'lambda': 0.1, 'p': 2}, 
                    optimizer='adagrad', optimizer_params={'lr': 0.1})
    X = np.array([['a', 'y', 'b'],
                  ['b', 'y', 'a'],
                  ['a', 'y', 'c'],
                  ['c', 'y', 'a'],
                  ['a', 'y', 'd'],
                  ['c', 'y', 'd'],
                  ['b', 'y', 'c'],
                  ['f', 'y', 'e']])
    model.fit(X)
    y_pred = model.predict(np.array([['f', 'y', 'e'], ['b', 'y', 'd']]))
    print(y_pred)
    assert y_pred[0] > y_pred[1]


def test_fit_predict_HolE():
    model = HolE(batches_count=1, seed=555, epochs=20, k=10,
                 loss='pairwise', loss_params={'margin': 1}, regularizer='LP',
                 regularizer_params={'lambda': 0.1, 'p': 2}, 
                 optimizer='adagrad', optimizer_params={'lr': 0.1})
    X = np.array([['a', 'y', 'b'],
                  ['b', 'y', 'a'],
                  ['a', 'y', 'c'],
                  ['c', 'y', 'a'],
                  ['a', 'y', 'd'],
                  ['c', 'y', 'd'],
                  ['b', 'y', 'c'],
                  ['f', 'y', 'e']])
    model.fit(X)
    y_pred = model.predict(np.array([['f', 'y', 'e'], ['b', 'y', 'd']]))
    print(y_pred)
    assert y_pred[0] > y_pred[1]


def test_retrain():
    model = ComplEx(batches_count=1, seed=555, epochs=20, k=10,
                    loss='pairwise', loss_params={'margin': 1}, regularizer='LP',
                    regularizer_params={'lambda': 0.1, 'p': 2}, 
                    optimizer='adagrad', optimizer_params={'lr': 0.1})
    X = np.array([['a', 'y', 'b'],
                  ['b', 'y', 'a'],
                  ['a', 'y', 'c'],
                  ['c', 'y', 'a'],
                  ['a', 'y', 'd'],
                  ['c', 'y', 'd'],
                  ['b', 'y', 'c'],
                  ['f', 'y', 'e']])
    model.fit(X)
    y_pred_1st = model.predict(np.array([['f', 'y', 'e'], ['b', 'y', 'd']]))
    model.fit(X)
    y_pred_2nd = model.predict(np.array([['f', 'y', 'e'], ['b', 'y', 'd']]))
    np.testing.assert_array_equal(y_pred_1st, y_pred_2nd)


def test_fit_predict_wn18_TransE():
    X = load_wn18()
    model = TransE(batches_count=1, seed=555, epochs=5, k=100, loss='pairwise',
                   loss_params={'margin': 5},
                   verbose=True, optimizer='adagrad',
                   optimizer_params={'lr': 0.1})
    model.fit(X['train'])
    y = model.predict(X['test'][:1])

    print(y)


def test_missing_entity_ComplEx():

    X = np.array([['a', 'y', 'b'],
                  ['b', 'y', 'a'],
                  ['a', 'y', 'c'],
                  ['c', 'y', 'a'],
                  ['a', 'y', 'd'],
                  ['c', 'y', 'd'],
                  ['b', 'y', 'c'],
                  ['f', 'y', 'e']])
    model = ComplEx(batches_count=1, seed=555, epochs=2, k=5)
    model.fit(X)
    with pytest.raises(ValueError):
        model.predict(['a', 'y', 'zzzzzzzzzzz'])
    with pytest.raises(ValueError):
        model.predict(['a', 'xxxxxxxxxx', 'e'])
    with pytest.raises(ValueError):
        model.predict(['zzzzzzzz', 'y', 'e'])


def test_fit_predict_wn18_ComplEx():
    X = load_wn18()
    model = ComplEx(batches_count=1, seed=555, epochs=5, k=100,
                    loss='pairwise', loss_params={'margin': 1}, regularizer='LP',
                    regularizer_params={'lambda': 0.1, 'p': 2}, 
                    optimizer='adagrad', optimizer_params={'lr': 0.1})
    model.fit(X['train'])
    y = model.predict(X['test'][:1])
    print(y)


def test_lookup_embeddings():
    model = DistMult(batches_count=2, seed=555, epochs=20, k=10, loss='pairwise', loss_params={'margin': 5}, 
                     optimizer='adagrad', optimizer_params={'lr': 0.1})
    X = np.array([['a', 'y', 'b'],
                  ['b', 'y', 'a'],
                  ['a', 'y', 'c'],
                  ['c', 'y', 'a'],
                  ['a', 'y', 'd'],
                  ['c', 'y', 'd'],
                  ['b', 'y', 'c'],
                  ['f', 'y', 'e']])
    model.fit(X)
    model.get_embeddings(['a', 'b'], embedding_type='entity')


def test_is_fitted_on():

    model = DistMult(batches_count=2, seed=555, epochs=1, k=10,
                     loss='pairwise', loss_params={'margin': 5},
                     optimizer='adagrad', optimizer_params={'lr': 0.1})
    X = np.array([['a', 'y', 'b'],
                  ['b', 'y', 'a'],
                  ['a', 'y', 'c'],
                  ['c', 'z', 'a'],
                  ['a', 'z', 'd']])

    model.fit(X)

    X1 = np.array([['a', 'y', 'b'],
                   ['b', 'y', 'a'],
                   ['a', 'y', 'c'],
                   ['c', 'z', 'a'],
                   ['g', 'z', 'd']])

    X2 = np.array([['a', 'y', 'b'],
                   ['b', 'y', 'a'],
                   ['a', 'y', 'c'],
                   ['c', 'z', 'a'],
                   ['a', 'x', 'd']])

    # Fits the train triples
    assert model.is_fitted_on(X) is True
    # Doesn't fit the extra entity triples
    assert model.is_fitted_on(X1) is False
    # Doesn't fit the extra relationship triples
    assert model.is_fitted_on(X2) is False
