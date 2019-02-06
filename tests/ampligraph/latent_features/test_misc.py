import numpy as np
import pytest


from ampligraph.latent_features.misc import get_entity_triples, get_neighbour_triples_head_tail, \
    get_neighbour_triples_weighted_walks, get_neighbour_triples_uniform_walks, \
    get_neighbour_triples_exhaustive, get_neighbour_matrix_for_triplets, \
    get_schema_matrix_for_triplets, calculate_frequency_map, calculate_graph_pagerank, \
    normalize_page_rank_weights, softmax, normalize_probabilities


def test_get_neighbour_triples_weighted_walks():
    # Tests length of weighted walks (k=2)
    # Tests LAST triple in weighted walk - random seed ensures identical results

    # Graph
    G = np.array([['a', 'y', 'b'], ['a', 'z', 'c'], ['a', 'y', 'c'], ['b', 'y', 'd'], ['c', 'y', 'a'],
                  ['c', 'z', 'b'], ['c', 'z', 'd'], ['d', 'y', 'e'], ['d', 'y', 'j'], ['d', 'z', 'e'],
                  ['d', 'z', 'h'], ['e', 'y', 'f'], ['e', 'y', 'g'], ['e', 'z', 'g'], ['e', 'y', 'd'],
                  ['e', 'z', 'd'], ['f', 'y', 'c'], ['g', 'y', 'd'], ['g', 'y', 'h'], ['g', 'z', 'j'],
                  ['h', 'y', 'i'], ['i', 'y', 'j'], ['i', 'y', 'd']])

    # Candidate triple
    t = ['d', 'y', 'e']

    N = get_neighbour_triples_weighted_walks(t, G, weighting='object_frequency', k=2)
    assert (len(N) == 3)
    assert (np.all(N[2] == ['c', 'z', 'd']))

    N = get_neighbour_triples_weighted_walks(t, G, weighting='inverse_object_frequency', k=2)
    assert (len(N) == 3)
    assert (np.all(N[2] == ['e', 'y', 'g']))

    N = get_neighbour_triples_weighted_walks(t, G, weighting='predicate_frequency', k=2)
    assert (len(N) == 3)
    assert (np.all(N[2] == ['e', 'y', 'g']))

    N = get_neighbour_triples_weighted_walks(t, G, weighting='inverse_predicate_frequency', k=2)
    assert (len(N) == 3)
    assert (np.all(N[2] == ['g', 'y', 'd']))

    N = get_neighbour_triples_weighted_walks(t, G, weighting='predicate_object_frequency', k=2)
    assert (len(N) == 3)
    assert (np.all(N[2] == ['c', 'z', 'd']))

    N = get_neighbour_triples_weighted_walks(t, G, weighting='inverse_predicate_object_frequency', k=2)
    assert (len(N) == 3)
    assert (np.all(N[2] == ['e', 'y', 'g']))

    N = get_neighbour_triples_weighted_walks(t, G, weighting='pagerank', k=2)
    assert (len(N) == 3)
    assert (np.all(N[2] == ['g', 'y', 'd']))


def test_get_neighbour_triples():

    # Graph
    X = np.array([['a', 'y', 'b'],
                  ['a', 'y', 'c'],
                  ['c', 'y', 'a'],
                  ['d', 'y', 'e'],

                  ['e', 'y', 'f'],
                  ['f', 'y', 'c']])

    # Entity of interest
    u = 'c'

    # Neighbours of u
    XN = np.array([['a', 'y', 'c'],
                   ['c', 'y', 'a'],
                   ['f', 'y', 'c']])

    # Call function
    N = get_entity_triples(u, X)

    assert(np.all(N == XN))


def test_get_neighbour_triples_uniform_walks():

    # Graph
    G = np.array([['a', 'y', 'b'],
                  ['a', 'y', 'c'],
                  ['b', 'y', 'd'],
                  ['c', 'y', 'a'],
                  ['d', 'y', 'e'],
                  ['d', 'y', 'j'],
                  ['e', 'y', 'f'],
                  ['e', 'y', 'g'],
                  ['f', 'y', 'c'],
                  ['g', 'y', 'd'],
                  ['g', 'y', 'h'],
                  ['h', 'y', 'i'],
                  ['i', 'y', 'j']])

    # Candidate triple
    t = ['i', 'y', 'j']

    # Call function
    N = get_neighbour_triples_uniform_walks(t, G, k=2)
    assert (len(N) == 3)

    N = get_neighbour_triples_uniform_walks(t, G, k=3)
    assert (len(N) == 4)


@pytest.mark.skip(reason="excluded to try out jenkins.")   # TODO: re-enable this
def test_get_neighbour_triples_exhaustive():

    # Graph
    G = np.array([['a', 'y', 'b'],
                  ['a', 'y', 'c'],
                  ['b', 'y', 'd'],
                  ['c', 'y', 'a'],
                  ['d', 'y', 'e'],
                  ['d', 'y', 'j'],
                  ['e', 'y', 'f'],
                  ['e', 'y', 'g'],
                  ['f', 'y', 'c'],
                  ['g', 'y', 'd'],
                  ['g', 'y', 'h'],
                  ['h', 'y', 'i'],
                  ['i', 'y', 'j']])

    # Candidate triple
    t = ['f', 'y', 'c']

    # Neighbourhood of t, k=2
    NY_k2 = np.array([['a', 'y', 'b'],
                      ['a', 'y', 'c'],
                      ['c', 'y', 'a'],
                      ['d', 'y', 'e'],
                      ['e', 'y', 'f'],
                      ['e', 'y', 'g']])

    # Neighbourhood of t, k=3
    NY_k3 = np.array([['a', 'y', 'b'],
                      ['a', 'y', 'c'],
                      ['b', 'y', 'd'],
                      ['c', 'y', 'a'],
                      ['d', 'y', 'e'],
                      ['d', 'y', 'j'],
                      ['e', 'y', 'f'],
                      ['e', 'y', 'g'],
                      ['g', 'y', 'd'],
                      ['g', 'y', 'h']])

    # Get neighbours
    N_k2 = get_neighbour_triples_exhaustive(t, G, k=2, limit=None)
    N_k3 = get_neighbour_triples_exhaustive(t, G, k=3, limit=None)
    N_k3_l4 = get_neighbour_triples_exhaustive(t, G, k=3, limit=4)

    assert (np.all(N_k2 == NY_k2))
    assert (np.all(N_k3 == NY_k3))
    assert (np.all(N_k3_l4 == NY_k3[:3,:]))


def test_get_neighbour_matrix_for_triples():
    # Graph
    X = np.array([[0, 0, 1],
                  [0, 0, 2],
                  [2, 0, 0],
                  [4, 0, 3],
                  [3, 0, 5],
                  [5, 0, 2]])

    # Neighbours, counts and max_num_rel
    full_array, num_array, max_num_rel, head_map, tail_map = get_neighbour_matrix_for_triplets(X)

    assert (max_num_rel == 2)
    # head neighbour of entity 0 is 2 (3rd row)
    assert (np.all(full_array[0][0] == [2]))
    assert (num_array[0][0] == 1)
    # tail neigbour of entity 0 are 1,2 (first 2 rows)
    assert (np.all(full_array[0][1] == [1, 2]))
    assert (num_array[0][1] == 2)
    # head neighbour of entity 1 is 0 (1st row)
    assert (np.all(full_array[0][2] == [0]))
    assert (num_array[0][2] == 1)
    # head neighbour of entity 1 is none, so 2 pads
    assert (np.all(full_array[0][3] == []))
    assert (num_array[0][3] == 0)


def test_get_neighbour_triples_head_tail():
    # Graph
    X = np.array([['a', 'y', 'b'],
                  ['a', 'y', 'c'],
                  ['c', 'y', 'a'],
                  ['d', 'y', 'e'],
                  ['e', 'y', 'f'],
                  ['f', 'y', 'c']])
    # Entity of interest
    u = 'c'

    # Head neighbours of u
    HN = np.array([['a', 'y', 'c'],
                   ['f', 'y', 'c']])

    TN = np.array([['c', 'y', 'a']])

    # Call function
    H, T = get_neighbour_triples_head_tail(u, X)

    assert (np.all(HN == H))
    assert (np.all(TN == T))


def test_calculate_frequency_map():

    # Sample graph with trivial frequency
    G = np.array([['a', 'y', 'b'],
                  ['a', 'z', 'c'],
                  ['a', 'y', 'c'],
                  ['b', 'y', 'd'],
                  ['c', 'y', 'a']])

    object_freq_map = calculate_frequency_map(G, type='object', inverse_freq=False)
    assert(object_freq_map['a'] == 0.2)

    object_freq_map_inv = calculate_frequency_map(G, type='object', inverse_freq=True)
    assert (object_freq_map_inv['a'] == 0.8)

    predicate_freq_map = calculate_frequency_map(G, type='predicate', inverse_freq=False)
    assert (predicate_freq_map['z'] == 0.2)

    predicate_freq_map_inv = calculate_frequency_map(G, type='predicate', inverse_freq=True)
    assert (predicate_freq_map_inv['z'] == 0.8)

    pred_object_freq_map = calculate_frequency_map(G, type='predicate-object', inverse_freq=False)
    assert (pred_object_freq_map['z']['a'] == 0.0)
    assert (pred_object_freq_map['y']['a'] == 0.2)

    pred_object_freq_map_inv = calculate_frequency_map(G, type='predicate-object', inverse_freq=True)
    assert (pred_object_freq_map_inv['z']['a'] == 1.0)
    assert (pred_object_freq_map_inv['y']['a'] == 0.8)


@pytest.mark.skip(reason="excluded to try out jenkins.")   # TODO: re-enable this
def test_get_schema_matrix_for_triplets():

    # Graph
    X = np.array([['0', '0', '1'],
                  ['0', '0', '2'],
                  ['2', '0', '0'],
                  ['3', '0', '4'],
                  ['4', '0', '5'],
                  ['5', '0', '2']])

    # Entity of interest
    S = np.array([['0', 'typeOf', '1'],
                  ['0', 'typeOf', '3'],
                  ['1', 'typeOf', '2'],
                  ['2', 'typeOf', '4'],
                  ['3', 'typeOf', '5'],
                  ['2', 'typeOf', '6']])

    c, n = get_schema_matrix_for_triplets(X, S, pad_value=100)

    assert np.all(c[0][0] == [1, 3])
    assert np.all(n[0][0] == 2)
    assert np.all(c[0][1] == [2, 100])
    assert np.all(n[0][1] == 1)

    assert np.all(c[2][0] == [4, 6])
    assert np.all(n[2][0] == 2)
    assert np.all(c[2][1] == [1, 2])
    assert np.all(n[2][1] == 2)

    assert np.all(c[5][0] == [100, 100])
    assert np.all(n[5][0] == 0)
    assert np.all(c[5][1] == [4, 6])
    assert np.all(n[5][1] == 2)


def test_softmax():

    L = [0, 1]
    L2 = softmax(L)
    assert (np.all (L2 == [0.2689414213699951, 0.7310585786300049]))

def test_normalize_probabilities():

    L = [0.1, 0.2, 0.1]
    L2 = normalize_probabilities(L)
    assert (np.all(L2 == [0.25, 0.5, 0.25]) )


def test_calculate_graph_pagerank():

    # Calculates graph pagerank, tests value of 'a'
    # Normalizes the graph, tests that normalized values sum to 1.0 (with tolerance of 1e-15)

    # Graph
    G = np.array([['a', 'y', 'b'], ['a', 'z', 'c'], ['a', 'y', 'c'], ['b', 'y', 'd'], ['c', 'y', 'a'],
                  ['c', 'z', 'b'], ['c', 'z', 'd'], ['d', 'y', 'e'], ['d', 'y', 'j'], ['d', 'z', 'e'],
                  ['d', 'z', 'h'], ['e', 'y', 'f'], ['e', 'y', 'g'], ['e', 'z', 'g'], ['e', 'y', 'd'],
                  ['e', 'z', 'd'], ['f', 'y', 'c'], ['g', 'y', 'd'], ['g', 'y', 'h'], ['g', 'z', 'j'],
                  ['h', 'y', 'i'], ['i', 'y', 'j'], ['i', 'y', 'd']])

    page_ranks = calculate_graph_pagerank(G, d=0.85, max_iter=10)

    assert (len(page_ranks) == 10)
    assert (page_ranks['a'] == 0.3049898180354944)

    W = normalize_page_rank_weights(G, page_ranks)

    for k in W.keys():
        total = 0
        for k2 in W[k].keys():
            total += W[k][k2]

        if total > 0:
            assert (np.abs(total - 1.0) < 1e-15)


if __name__ == "__main__":
    test_get_neighbour_matrix_for_triples()