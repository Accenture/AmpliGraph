import numpy as np
import pytest
from ampligraph.latent_features import TransE, DistMult, ComplEx
from ampligraph.evaluation import evaluate_performance, generate_corruptions_for_eval, \
    generate_corruptions_for_fit, to_idx, create_mappings, mrr_score, hits_at_n_score, select_best_model_ranking
from ampligraph.datasets import load_wn18


@pytest.mark.skip(reason="Speeding up jenkins")
def test_select_best_model_ranking():
    X = load_wn18()
    model_class = ComplEx
    param_grid = {'batches_count': [10],
                  'seed': [0],
                  'epochs': [1],
                  'k': [50, 150],
                  'pairwise_margin': [1],
                  'lr': [.1],
                  'eta': [2],
                  'loss': ['pairwise']}
    best_model, best_params, best_mrr_train, ranks_test, mrr_test = select_best_model_ranking(model_class, X,
                                                                                              param_grid,
                                                                                              filter_retrain=True,
                                                                                              eval_splits=50)
    print(type(best_model).__name__, best_params, best_mrr_train, mrr_test)
    assert best_params['k'] == 150


@pytest.mark.skip(reason="Speeding up jenkins")
def test_evaluate_performance():
    X = load_wn18()
    model = ComplEx(batches_count=10, seed=0, epochs=10, k=150, lr=.1, eta=10, loss='pairwise', lambda_reg=0.01,
                    pairwise_margin=5, regularizer=None, optimizer='adagrad', verbose=True)
    model.fit(np.concatenate((X['train'], X['valid'])))

    filter = np.concatenate((X['train'], X['valid'], X['test']))
    ranks = evaluate_performance(X['test'][:200], model=model, filter_triples=filter, verbose=True)

    mrr = mrr_score(ranks)
    hits_10 = hits_at_n_score(ranks, n=10)
    print("ranks: %s" % ranks)
    print("MRR: %f" % mrr)
    print("Hits@10: %f" % hits_10)

    # TODO: add test condition (MRR raw for WN18 and TransE should be ~ 0.335 - check papers)


@pytest.mark.skip(reason="Speeding up jenkins")
def test_evaluate_performance_nll_complex():
    X = load_wn18()
    model = ComplEx(batches_count=10, seed=0, epochs=10, k=150, lr=.1, eta=10, loss='nll', lambda_reg=0.01,
                    regularizer=None, optimizer='adagrad', verbose=True)
    model.fit(np.concatenate((X['train'], X['valid'])))

    filter = np.concatenate((X['train'], X['valid'], X['test']))
    ranks = evaluate_performance(X['test'][:200], model=model, filter_triples=filter, verbose=True)

    mrr = mrr_score(ranks)
    hits_10 = hits_at_n_score(ranks, n=10)
    print("ranks: %s" % ranks)
    print("MRR: %f" % mrr)
    print("Hits@10: %f" % hits_10)


@pytest.mark.skip(reason="Speeding up jenkins")
def test_evaluate_performance_TransE():
    X = load_wn18()
    model = TransE(batches_count=10, seed=0, epochs=100, k=100, lr=.1, eta=5,
                   pairwise_margin=5, loss='pairwise', optimizer='adagrad')
    model.fit(np.concatenate((X['train'], X['valid'])))

    filter = np.concatenate((X['train'], X['valid'], X['test']))
    ranks = evaluate_performance(X['test'][:200], model=model, filter_triples=filter, splits=1)
    # ranks = evaluate_performance(X['test'][:200], model=model)

    mrr = mrr_score(ranks)
    hits_10 = hits_at_n_score(ranks, n=10)
    print("ranks: %s" % ranks)
    print("MRR: %f" % mrr)
    print("Hits@10: %f" % hits_10)

    # TODO: add test condition (MRR raw for WN18 and TransE should be ~ 0.335 - check papers)


def test_generate_corruptions_for_eval():
    x = np.array([0, 0, 1])
    idx_entities = np.array([0, 1, 2, 3])
    x_n_actual = generate_corruptions_for_eval(x, idx_entities=idx_entities)
    x_n_expected = np.array([[1, 0, 1],
                             [2, 0, 1],
                             [3, 0, 1],
                             [0, 0, 0],
                             [0, 0, 2],
                             [0, 0, 3]])
    np.testing.assert_array_equal(x_n_actual, x_n_expected)


def test_generate_corruptions_for_eval_filtered():
    x = np.array([0, 0, 1])
    idx_entities = np.array([0, 1, 2, 3])
    filter_triples = np.array(([1, 0, 1], [2, 0, 1]))

    x_n_actual = generate_corruptions_for_eval(x, idx_entities=idx_entities, filter=filter_triples)
    x_n_expected = np.array([[3, 0, 1],
                             [0, 0, 0],
                             [0, 0, 2],
                             [0, 0, 3]])
    np.testing.assert_array_equal(np.sort(x_n_actual, axis=0), np.sort(x_n_expected, axis=0))


def test_generate_corruptions_for_eval_filtered_object():
    x = np.array([0, 0, 1])
    idx_entities = np.array([0, 1, 2, 3])
    filter_triples = np.array(([1, 0, 1], [2, 0, 1]))

    x_n_actual = generate_corruptions_for_eval(x, idx_entities=idx_entities, filter=filter_triples, side='o')
    x_n_expected = np.array([[0, 0, 0],
                             [0, 0, 2],
                             [0, 0, 3]])
    np.testing.assert_array_equal(np.sort(x_n_actual, axis=0), np.sort(x_n_expected, axis=0))


def test_to_idx():

    X = np.array([['a', 'x', 'b'], ['c', 'y', 'd']])
    X_idx_expected = [[0, 0, 1], [2, 1, 3]]
    rel_to_idx, ent_to_idx = create_mappings(X)
    X_idx = to_idx(X, ent_to_idx=ent_to_idx, rel_to_idx=rel_to_idx)

    np.testing.assert_array_equal(X_idx, X_idx_expected)


# @pytest.mark.skip(reason="excluded to try out jenkins.")   # TODO: re-enable this
def test_generate_corruptions_for_fit():
    X = np.array([['a', 'x', 'b'],
                  ['c', 'x', 'd'],
                  ['e', 'x', 'f'],
                  ['g', 'x', 'h'],
                  ['i', 'x', 'l']])
    rel_to_idx, ent_to_idx = create_mappings(X)
    X = to_idx(X, ent_to_idx=ent_to_idx, rel_to_idx=rel_to_idx)

    rnd = np.random.RandomState(seed=0)
    X_corr = generate_corruptions_for_fit(X, ent_to_idx=ent_to_idx, rnd=rnd)

    # these values occur when seed=0
    # X_corr_exp = [[8, 0, 1],
    #               [4, 0, 3],
    #               [4, 0, 1],
    #               [6, 0, 0],
    #               [8, 0, 8]]

    X_corr_exp = [[5, 0, 1],
                  [0, 0, 3],
                  [3, 0, 5],
                  [6, 0, 3],
                  [8, 0, 7]]

    np.testing.assert_array_equal(X_corr, X_corr_exp)
