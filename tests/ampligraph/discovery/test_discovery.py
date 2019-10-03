import numpy as np
import pytest
from sklearn.cluster import DBSCAN
from ampligraph.discovery.discovery import discover_facts, generate_candidates, _setdiff2d, find_clusters, \
    find_duplicates, query_topn
from ampligraph.latent_features import ComplEx

def test_discover_facts():

    X = np.array([['a', 'y', 'b'],
                  ['b', 'y', 'a'],
                  ['a', 'y', 'c'],
                  ['c', 'y', 'a'],
                  ['a', 'y', 'd'],
                  ['c', 'y', 'd'],
                  ['b', 'y', 'c'],
                  ['f', 'y', 'e']])
    model = ComplEx(batches_count=1, seed=555, epochs=2, k=5)

    with pytest.raises(ValueError):
        discover_facts(X, model)

    model.fit(X)

    with pytest.raises(ValueError):
        discover_facts(X, model, strategy='error')

    with pytest.raises(ValueError):
        discover_facts(X, model, strategy='random_uniform', target_rel='error')


def test_generate_candidates():

    X = np.stack([['entity_{}'.format(np.mod(x, 15)) for x in range(50)],
                  ['rel_{}'.format(np.mod(x, 5)) for x in range(50)],
                  ['entity_{}'.format(np.mod(x, 20)) for x in range(50)]], axis=1)

    # Not sure this should be an error
    # with pytest.raises(ValueError):
    #     generate_candidates(X, strategy='error', target_rel='y',
    #                         max_candidates=4)


    # with pytest.raises(ValueError):
    #     generate_candidates(X, strategy='random_uniform', target_rel='y',
    #  max_candidates=0)

    # Test
    X_candidates = generate_candidates(X, strategy='random_uniform', target_rel='rel_0',
                                       max_candidates=15, consolidate_sides=False, seed=1916)

    assert X_candidates.shape == (15, 3)

    # Test
    X_candidates = generate_candidates(X, strategy='random_uniform', target_rel='rel_1', max_candidates=20,
                                       consolidate_sides=True, seed=1916)

    assert X_candidates.shape == (20, 3)
    assert X_candidates[0, 0] == 'entity_16'
    assert np.all(X_candidates[:, 1] == 'rel_1')

    # Test that consolidate_sides LHS and RHS is respected
    X_candidates = generate_candidates(X, strategy='random_uniform', target_rel='rel_0', max_candidates=20,
                                       consolidate_sides=False, seed=0)

    assert np.all(np.isin(X_candidates[:, 0], np.unique(X[:, 0])))
    assert np.all(np.isin(X_candidates[:, 2], np.unique(X[:, 2])))

    # Test that consolidate_sides=True LHS and RHS entities are mixed
    X_candidates = generate_candidates(X, strategy='random_uniform', target_rel='rel_0', max_candidates=100,
                                       consolidate_sides=True, seed=1)

    # Check that any of the head or tail entities from X has been found
    # on the OTHER side of the candidates
    assert np.logical_or(np.any(np.isin(X_candidates[:, 2], np.unique(X[:, 0]))),
                         np.all(np.isin(X_candidates[:, 0], np.unique(X[:, 2]))))


    # Test entity frequency generation
    X_candidates = generate_candidates(X, strategy='entity_frequency', target_rel='rel_0', max_candidates=20,
                                       consolidate_sides=False, seed=1)
    assert X_candidates.shape == (20, 3)

    X_candidates = generate_candidates(X, strategy='graph_degree', target_rel='rel_0', max_candidates=30,
                                       consolidate_sides=False, seed=1)
    assert X_candidates.shape == (30, 3)

    X_candidates = generate_candidates(X, strategy='cluster_coefficient', target_rel='rel_0', max_candidates=30,
                                       consolidate_sides=False, seed=2)
    assert X_candidates.shape == (30, 3)

    X_candidates = generate_candidates(X, strategy='cluster_triangles', target_rel='rel_0', max_candidates=50,
                                       consolidate_sides=False, seed=1)
    assert X_candidates.shape == (50, 3)

    X_candidates = generate_candidates(X, strategy='cluster_squares', target_rel='rel_0', max_candidates=60,
                                       consolidate_sides=False, seed=1)
    assert X_candidates.shape == (60, 3)

def test_setdiff2d():

    X = np.array([['a', 'y', 'b'],
                  ['b', 'y', 'a'],
                  ['a', 'y', 'c'],
                  ['c', 'y', 'a'],
                  ['a', 'y', 'd'],
                  ['c', 'y', 'd'],
                  ['b', 'y', 'c'],
                  ['f', 'y', 'e']])

    Y = np.array([['a', 'z', 'b'],
                  ['b', 'z', 'a'],
                  ['a', 'z', 'c'],
                  ['c', 'z', 'a'],
                  ['a', 'y', 'd'],
                  ['c', 'y', 'd'],
                  ['b', 'y', 'c'],
                  ['f', 'y', 'e']])

    ret1 = np.array([['a', 'y', 'b'],
                     ['b', 'y', 'a'],
                     ['a', 'y', 'c'],
                     ['c', 'y', 'a']])

    ret2 = np.array([['a', 'z', 'b'],
                     ['b', 'z', 'a'],
                     ['a', 'z', 'c'],
                     ['c', 'z', 'a']])

    assert np.array_equal(ret1, _setdiff2d(X, Y))
    assert np.array_equal(ret2, _setdiff2d(Y, X))

    # i.e., don't use it as setdiff1d
    with pytest.raises(RuntimeError):
        X = np.array([1, 2, 3, 4, 5, 6])
        Y = np.array([1, 2, 3, 7, 8, 9])
        _setdiff2d(X, Y)


def test_find_clusters():
    X = np.array([['a', 'y', 'b'],
                  ['b', 'y', 'a'],
                  ['a', 'y', 'c'],
                  ['c', 'y', 'a'],
                  ['a', 'y', 'd'],
                  ['c', 'x', 'd'],
                  ['b', 'y', 'c'],
                  ['f', 'y', 'e']])
    model = ComplEx(k=2, batches_count=2)
    model.fit(X)
    clustering_algorithm = DBSCAN(eps=1e-3, min_samples=1)

    labels = find_clusters(X, model, clustering_algorithm, mode='triple')
    assert np.array_equal(labels, np.array([0, 1, 2, 3, 4, 5, 6, 7]))

    labels = find_clusters(np.unique(X[:, 0]), model, clustering_algorithm, mode='entity')
    assert np.array_equal(labels, np.array([0, 1, 2, 3]))

    labels = find_clusters(np.unique(X[:, 1]), model, clustering_algorithm, mode='relation')
    assert np.array_equal(labels, np.array([0, 1]))

    labels = find_clusters(np.unique(X[:, 2]), model, clustering_algorithm, mode='entity')
    assert np.array_equal(labels, np.array([0, 1, 2, 3, 4]))

    with pytest.raises(ValueError):
        find_clusters(X, model, clustering_algorithm, mode='hah')
    with pytest.raises(ValueError):
        find_clusters(X, model, clustering_algorithm, mode='entity')
    with pytest.raises(ValueError):
        find_clusters(X, model, clustering_algorithm, mode='relation')
    with pytest.raises(ValueError):
        find_clusters(np.unique(X[:, 0]), model, clustering_algorithm, mode='triple')


def test_find_duplicates():
    X = np.array([['a', 'y', 'b'],
                  ['b', 'y', 'a'],
                  ['a', 'y', 'c'],
                  ['c', 'y', 'a'],
                  ['a', 'y', 'd'],
                  ['c', 'x', 'd'],
                  ['b', 'y', 'c'],
                  ['f', 'y', 'e']])
    model = ComplEx(k=2, batches_count=2)
    model.fit(X)

    entities = set('a b c d e f'.split())
    relations = set('x y'.split())

    def asserts(tol, dups, ent_rel, subspace):
        assert tol > 0.0
        assert len(dups) <= len(ent_rel)
        assert all(len(d) <= len(ent_rel) for d in dups)
        assert all(d.issubset(subspace) for d in dups)

    dups, tol = find_duplicates(X, model, mode='triple', tolerance='auto', expected_fraction_duplicates=0.5)
    asserts(tol, dups, X, {tuple(x) for x in X})

    dups, tol = find_duplicates(X, model, mode='triple', tolerance=1.0)
    assert tol == 1.0
    asserts(tol, dups, X, {tuple(x) for x in X})

    dups, tol = find_duplicates(np.unique(X[:, 0]), model, mode='entity', tolerance='auto', expected_fraction_duplicates=0.5)
    asserts(tol, dups, entities, entities)

    dups, tol = find_duplicates(np.unique(X[:, 2]), model, mode='entity', tolerance='auto', expected_fraction_duplicates=0.5)
    asserts(tol, dups, entities, entities)

    dups, tol = find_duplicates(np.unique(X[:, 1]), model, mode='relation', tolerance='auto', expected_fraction_duplicates=0.5)
    asserts(tol, dups, relations, relations)

    with pytest.raises(ValueError):
        find_duplicates(X, model, mode='hah')
    with pytest.raises(ValueError):
        find_duplicates(X, model, mode='entity')
    with pytest.raises(ValueError):
        find_duplicates(X, model, mode='relation')
    with pytest.raises(ValueError):
        find_duplicates(np.unique(X[:, 0]), model, mode='triple')


def test_query_topn():

    X = np.array([['a', 'y', 'b'],
                  ['b', 'y', 'a'],
                  ['a', 'y', 'c'],
                  ['c', 'y', 'a'],
                  ['a', 'y', 'd'],
                  ['c', 'x', 'd'],
                  ['b', 'y', 'c'],
                  ['f', 'y', 'e'],
                  ['a', 'z', 'f'],
                  ['c', 'z', 'f'],
                  ['b', 'z', 'f'],
                  ])

    model = ComplEx(k=2, batches_count=2)

    with pytest.raises(ValueError): # Model not fitted
        query_topn(model, top_n=2)

    model.fit(X)

    with pytest.raises(ValueError):
        query_topn(model, top_n=2)
    with pytest.raises(ValueError):
        query_topn(model, top_n=2, head='a')
    with pytest.raises(ValueError):
        query_topn(model, top_n=2, relation='y')
    with pytest.raises(ValueError):
        query_topn(model, top_n=2, tail='e')
    with pytest.raises(ValueError):
        query_topn(model, top_n=2, head='a', relation='y', tail='e')
    with pytest.raises(ValueError):
        query_topn(model, top_n=2, head='xx', relation='y')
    with pytest.raises(ValueError):
        query_topn(model, top_n=2, head='a', relation='yakkety')
    with pytest.raises(ValueError):
        query_topn(model, top_n=2, head='a', tail='sax')
    with pytest.raises(ValueError):
        query_topn(model, top_n=2, head='a', relation='x', rels_to_consider=['y', 'z'])
    with pytest.raises(ValueError):
        query_topn(model, top_n=2, head='a', tail='f', rels_to_consider=['y', 'z', 'error'])
    with pytest.raises(ValueError):
        query_topn(model, top_n=2, head='a', tail='e', rels_to_consider='y')
    with pytest.raises(ValueError):
        query_topn(model, top_n=2, head='a', relation='x', ents_to_consider=['zz', 'top'])
    with pytest.raises(ValueError):
        query_topn(model, top_n=2, head='a', tail='e', ents_to_consider=['a', 'b'])

    subj, pred, obj, top_n = 'a', 'x', 'e', 3

    Y, S = query_topn(model, top_n=top_n, head=subj, relation=pred)
    assert len(Y) == len(S)
    assert len(Y) == top_n
    assert np.all(Y[:, 0] == subj)
    assert np.all(Y[:, 1] == pred)

    Y, S = query_topn(model, top_n=top_n, relation=pred, tail=obj)
    assert np.all(Y[:, 1] == pred)
    assert np.all(Y[:, 2] == obj)

    ents_to_con = ['a', 'b', 'c', 'd']
    Y, S = query_topn(model, top_n=top_n, relation=pred, tail=obj, ents_to_consider=ents_to_con)
    assert np.all([x in ents_to_con for x in Y[:, 0]])

    rels_to_con = ['y', 'x']
    Y, S = query_topn(model, top_n=10, head=subj, tail=obj, rels_to_consider=rels_to_con)
    assert np.all([x in rels_to_con for x in Y[:, 1]])

    Y, S = query_topn(model, top_n=10, relation=pred, tail=obj)
    assert all(S[i] >= S[i + 1] for i in range(len(S) - 1))

