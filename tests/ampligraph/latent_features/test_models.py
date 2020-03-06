# Copyright 2019 The AmpliGraph Authors. All Rights Reserved.
#
# This file is Licensed under the Apache License, Version 2.0.
# A copy of the Licence is available in LICENCE, or at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
import numpy as np
import pytest
import os

from ampligraph.latent_features import TransE, DistMult, ComplEx, HolE, RandomBaseline, ConvKB, ConvE
from ampligraph.latent_features import set_entity_threshold, reset_entity_threshold
from ampligraph.datasets import load_wn18, load_wn18rr
from ampligraph.utils import save_model, restore_model
from ampligraph.evaluation import evaluate_performance, hits_at_n_score
from ampligraph.datasets import OneToNDatasetAdapter
from ampligraph.utils import save_model, restore_model
from ampligraph.evaluation.protocol import to_idx


def test_conve_bce_combo():
    # no exception
    model = ConvE(loss='bce')

    # no exception
    model = TransE(loss='nll')

    # Invalid combination. Hence exception.
    with pytest.raises(ValueError):
        model = TransE(loss='bce')

    # Invalid combination. Hence exception.
    with pytest.raises(ValueError):
        model = ConvE(loss='nll')


def test_large_graph_mode():
    set_entity_threshold(10)
    X = load_wn18()
    model = ComplEx(batches_count=100, seed=555, epochs=1, k=50, loss='multiclass_nll', loss_params={'margin': 5},
                    verbose=True, optimizer='sgd', optimizer_params={'lr': 0.001})
    model.fit(X['train'])
    X_filter = np.concatenate((X['train'], X['valid'], X['test']), axis=0)
    evaluate_performance(X['test'][::1000], model, X_filter, verbose=True, corrupt_side='s,o')

    y = model.predict(X['test'][:1])
    print(y)
    reset_entity_threshold()


def test_output_sizes():
    ''' Test to check whether embedding matrix sizes match the input data (num rel/ent and k)
    '''
    def perform_test():
        X = load_wn18rr()
        k = 5
        unique_entities = np.unique(np.concatenate([X['train'][:, 0],
                                                    X['train'][:, 2]], 0))
        unique_relations = np.unique(X['train'][:, 1])
        model = TransE(batches_count=100, seed=555, epochs=1, k=k, loss='multiclass_nll', loss_params={'margin': 5},
                        verbose=True, optimizer='sgd', optimizer_params={'lr': 0.001})
        model.fit(X['train'])
        # verify ent and rel shapes
        assert(model.trained_model_params[0].shape[0] == len(unique_entities))
        assert(model.trained_model_params[1].shape[0] == len(unique_relations))
        # verify k
        assert(model.trained_model_params[0].shape[1] == k)
        assert(model.trained_model_params[1].shape[1] == k)

    # Normal mode
    perform_test()

    # Large graph mode
    set_entity_threshold(10)
    perform_test()
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
                                 corrupt_side='s+o',
                                 verbose=False)
    hits10 = hits_at_n_score(ranks, n=10)
    hits1 = hits_at_n_score(ranks, n=1)
    assert ranks.shape == (len(X['test']), )
    assert hits10 < 0.01 and hits1 == 0.0

    ranks = evaluate_performance(X["test"],
                                 model=model,
                                 corrupt_side='s,o',
                                 verbose=False)
    hits10 = hits_at_n_score(ranks, n=10)
    hits1 = hits_at_n_score(ranks, n=1)
    assert ranks.shape == (len(X['test']), 2)
    assert hits10 < 0.01 and hits1 == 0.0

    ranks_filtered = evaluate_performance(X["test"],
                                          filter_triples=np.concatenate((X['train'], X['valid'], X['test'])),
                                          model=model,
                                          corrupt_side='s,o',
                                          verbose=False)
    hits10 = hits_at_n_score(ranks_filtered, n=10)
    hits1 = hits_at_n_score(ranks_filtered, n=1)
    assert ranks_filtered.shape == (len(X['test']), 2)
    assert hits10 < 0.01 and hits1 == 0.0
    assert np.all(ranks_filtered <= ranks)
    assert np.any(ranks_filtered != ranks)


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


def test_conve_fit_predict_save_restore():

    X = load_wn18()
    model = ConvE(batches_count=100, seed=22, epochs=1, k=10,
                  embedding_model_params={'conv_filters': 16, 'conv_kernel_size': 3},
                  optimizer='adam', optimizer_params={'lr': 0.01},
                  loss='bce', loss_params={},
                  regularizer=None, regularizer_params={'p': 2, 'lambda': 1e-5},
                  verbose=True, low_memory=True)

    model.fit(X['train'])

    y1 = model.predict(X['test'][:5])

    save_model(model, 'model.tmp')
    del model
    model = restore_model('model.tmp')

    y2 = model.predict(X['test'][:5])

    assert np.all(y1 == y2)
    os.remove('model.tmp')


def test_conve_evaluation_protocol():
    X = load_wn18()
    model = ConvE(batches_count=200, seed=22, epochs=1, k=10,
                  embedding_model_params={'conv_filters': 16, 'conv_kernel_size': 3},
                  optimizer='adam', optimizer_params={'lr': 0.01},
                  loss='bce', loss_params={},
                  regularizer=None, regularizer_params={'p': 2, 'lambda': 1e-5},
                  verbose=True, low_memory=True)

    model.fit(X['train'])

    y1 = model.predict(X['test'][:5])

    save_model(model, 'model.tmp')
    del model
    model = restore_model('model.tmp')

    y2 = model.predict(X['test'][:5])

    assert np.all(y1 == y2)

    os.remove('model.tmp')


def test_convkb_train_predict():

    model = ConvKB(batches_count=2, seed=22, epochs=1, k=10, eta=1,
                   embedding_model_params={'num_filters': 16,
                                           'filter_sizes': [1],
                                           'dropout': 0.0,
                                           'is_trainable': True},
                   optimizer='adam',
                   optimizer_params={'lr': 0.001},
                   loss='pairwise',
                   loss_params={},
                   verbose=True)

    X = load_wn18()
    model.fit(X['train'])

    y1 = model.predict(X['test'][:5])

    save_model(model, 'convkb.tmp')
    del model

    model = restore_model('convkb.tmp')

    y2 = model.predict(X['test'][:5])

    assert np.all(y1 == y2)


def test_convkb_save_restore():

    model = ConvKB(batches_count=2, seed=22, epochs=1, k=10, eta=1,
                   embedding_model_params={'num_filters': 16,
                                           'filter_sizes': [1],
                                           'dropout': 0.0,
                                           'is_trainable': True},
                   optimizer='adam',
                   optimizer_params={'lr': 0.001},
                   loss='pairwise',
                   loss_params={},
                   verbose=True)

    X = load_wn18()
    model.fit(X['train'])
    y1 = model.predict(X['test'][:10])

    save_model(model, 'convkb.tmp')
    del model
    model = restore_model('convkb.tmp')

    y2 = model.predict(X['test'][:10])

    assert np.all(y1 == y2)

    os.remove('convkb.tmp')


def test_predict():
    model = DistMult(batches_count=2, seed=555, epochs=1, k=10,
                     loss='pairwise', loss_params={'margin': 5},
                     optimizer='adagrad', optimizer_params={'lr': 0.1})
    X = np.array([['a', 'y', 'b'],
                  ['b', 'y', 'a'],
                  ['a', 'y', 'c'],
                  ['c', 'z', 'a'],
                  ['a', 'z', 'd']])
    model.fit(X)

    preds1 = model.predict(X)
    preds2 = model.predict(to_idx(X, model.ent_to_idx, model.rel_to_idx), from_idx=True)

    np.testing.assert_array_equal(preds1, preds2)


def test_predict_twice():
    model = DistMult(batches_count=2, seed=555, epochs=1, k=10,
                     loss='pairwise', loss_params={'margin': 5},
                     optimizer='adagrad', optimizer_params={'lr': 0.1})
    X = np.array([['a', 'y', 'b'],
                  ['b', 'y', 'a'],
                  ['a', 'y', 'c'],
                  ['c', 'z', 'a'],
                  ['a', 'z', 'd']])
    model.fit(X)

    X_test1 = np.array([['a', 'y', 'b'],
                        ['b', 'y', 'a']])

    X_test2 = np.array([['a', 'y', 'c'],
                        ['c', 'z', 'a']])

    preds1 = model.predict(X_test1)
    preds2 = model.predict(X_test2)

    assert not np.array_equal(preds1, preds2)


def test_calibrate_with_corruptions():
    model = DistMult(batches_count=2, seed=555, epochs=1, k=10,
                     loss='pairwise', loss_params={'margin': 5},
                     optimizer='adagrad', optimizer_params={'lr': 0.1})
    X = np.array([['a', 'y', 'b'],
                  ['b', 'y', 'a'],
                  ['a', 'y', 'c'],
                  ['c', 'z', 'a'],
                  ['a', 'z', 'd']])
    model.fit(X)

    X_pos = np.array([['a', 'y', 'b'],
                      ['b', 'y', 'a'],
                      ['a', 'y', 'c'],
                      ['c', 'z', 'a'],
                      ['d', 'z', 'd']])

    with pytest.raises(RuntimeError):
        model.predict_proba(X_pos)

    with pytest.raises(ValueError):
        model.calibrate(X_pos, batches_count=2, epochs=10)

    model.calibrate(X_pos, positive_base_rate=0.5, batches_count=2, epochs=10)

    probas = model.predict_proba(X_pos)

    assert np.logical_and(probas > 0, probas < 1).all()


def test_calibrate_with_negatives():
    model = DistMult(batches_count=2, seed=555, epochs=1, k=10,
                     loss='pairwise', loss_params={'margin': 5},
                     optimizer='adagrad', optimizer_params={'lr': 0.1})

    X = np.array([['a', 'y', 'b'],
                  ['b', 'y', 'a'],
                  ['a', 'y', 'c'],
                  ['c', 'z', 'a'],
                  ['a', 'z', 'd']])
    model.fit(X)

    X_pos = np.array([['a', 'y', 'b'],
                      ['b', 'y', 'a'],
                      ['a', 'y', 'c'],
                      ['c', 'z', 'a'],
                      ['d', 'z', 'd']])

    X_neg = np.array([['a', 'y', 'd'],
                      ['d', 'y', 'a'],
                      ['c', 'y', 'a'],
                      ['a', 'z', 'd']])

    with pytest.raises(RuntimeError):
        model.predict_proba(X_pos)

    with pytest.raises(ValueError):
        model.calibrate(X_pos, X_neg, positive_base_rate=50, batches_count=2, epochs=10)

    model.calibrate(X_pos, X_neg, batches_count=2, epochs=10)

    probas = model.predict_proba(np.concatenate((X_pos, X_neg)))

    assert np.logical_and(probas > 0, probas < 1).all()