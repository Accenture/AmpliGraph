# Copyright 2019 The AmpliGraph Authors. All Rights Reserved.
#
# This file is Licensed under the Apache License, Version 2.0.
# A copy of the Licence is available in LICENCE, or at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
from ampligraph.datasets import load_wn18, load_fb15k, load_fb15k_237, load_yago3_10, load_wn18rr, load_wn11, \
    load_fb13, OneToNDatasetAdapter
from ampligraph.datasets.datasets import _clean_data
import numpy as np
import pytest


def test_clean_data():
    X = {
        'train': np.array([['a', 'b', 'c'], ['d', 'e', 'f'], ['g', 'h', 'i'], ['j', 'k', 'l']]),
        'valid': np.array([['a', 'b', 'c'], ['x', 'e', 'f'], ['g', 'a', 'i'], ['j', 'k', 'y']]),
        'test':  np.array([['a', 'b', 'c'], ['d', 'e', 'x'], ['g', 'b', 'i'], ['y', 'k', 'l']]),
    }

    clean_X, valid_idx, test_idx = _clean_data(X, return_idx=True)

    np.testing.assert_array_equal(clean_X['train'], X['train'])
    np.testing.assert_array_equal(clean_X['valid'], np.array([['a', 'b', 'c']]))
    np.testing.assert_array_equal(clean_X['test'],  np.array([['a', 'b', 'c'], ['g', 'b', 'i']]))
    np.testing.assert_array_equal(valid_idx,  np.array([True, False, False, False]))
    np.testing.assert_array_equal(test_idx, np.array([True, False, True, False]))


def test_load_wn18():
    wn18 = load_wn18()
    assert len(wn18['train']) == 141442
    assert len(wn18['valid']) == 5000
    assert len(wn18['test']) == 5000

    ent_train = np.union1d(np.unique(wn18["train"][:, 0]), np.unique(wn18["train"][:, 2]))
    ent_valid = np.union1d(np.unique(wn18["valid"][:, 0]), np.unique(wn18["valid"][:, 2]))
    ent_test = np.union1d(np.unique(wn18["test"][:, 0]), np.unique(wn18["test"][:, 2]))
    distinct_ent = np.union1d(np.union1d(ent_train, ent_valid), ent_test)
    distinct_rel = np.union1d(np.union1d(np.unique(wn18["train"][:, 1]), np.unique(wn18["train"][:, 1])),
                              np.unique(wn18["train"][:, 1]))

    assert len(distinct_ent) == 40943
    assert len(distinct_rel) == 18


def test_reciprocals():
    """Test for reciprocal relations
    """
    # Create dataset with reciprocal relations and test if the are added
    fb15k = load_fb15k(add_reciprocal_rels=True)
    train_reciprocal = fb15k['train']
    triple = train_reciprocal[0]
    reciprocal_triple = train_reciprocal[train_reciprocal.shape[0]//2]
    assert(triple[0]==reciprocal_triple[2])
    assert(triple[2]==reciprocal_triple[0])
    assert(triple[1]+'_reciprocal'==reciprocal_triple[1])
    
    # create the same dataset without reciprocals. Now the number of triples should be half of prev
    fb15k = load_fb15k(add_reciprocal_rels=False)
    assert(fb15k['train'].shape[0]==train_reciprocal.shape[0]//2)


def test_load_fb15k():
    fb15k = load_fb15k()
    assert len(fb15k['train']) == 483142
    assert len(fb15k['valid']) == 50000
    assert len(fb15k['test']) == 59071

    # ent_train = np.union1d(np.unique(fb15k["train"][:,0]), np.unique(fb15k["train"][:,2]))
    # ent_valid = np.union1d(np.unique(fb15k["valid"][:,0]), np.unique(fb15k["valid"][:,2]))
    # ent_test = np.union1d(np.unique(fb15k["test"][:,0]), np.unique(fb15k["test"][:,2]))
    # distinct_ent = np.union1d(np.union1d(ent_train, ent_valid), ent_test)
    # distinct_rel = np.union1d(np.union1d(np.unique(fb15k["train"][:,1]), np.unique(fb15k["train"][:,1])),
    #                           np.unique(fb15k["train"][:,1]))

    # assert len(distinct_ent) == 14951  
    # assert len(distinct_rel) == 1345  


def test_load_fb15k_237():
    fb15k_237 = load_fb15k_237()
    assert len(fb15k_237['train']) == 272115 
    
    # - 9 because 9 triples containing unseen entities are removed
    assert len(fb15k_237['valid']) == 17535 - 9

    # - 28 because 28 triples containing unseen entities are removed
    assert len(fb15k_237['test']) == 20466 - 28


def test_yago_3_10():
    yago_3_10 = load_yago3_10()
    assert len(yago_3_10['train']) == 1079040 
    assert len(yago_3_10['valid']) == 5000 - 22
    assert len(yago_3_10['test']) == 5000 - 18

    # ent_train = np.union1d(np.unique(yago_3_10["train"][:,0]), np.unique(yago_3_10["train"][:,2]))
    # ent_valid = np.union1d(np.unique(yago_3_10["valid"][:,0]), np.unique(yago_3_10["valid"][:,2]))
    # ent_test = np.union1d(np.unique(yago_3_10["test"][:,0]), np.unique(yago_3_10["test"][:,2]))

    # assert len(set(ent_valid) - set(ent_train)) == 22
    # assert len (set(ent_test) - ((set(ent_valid) & set(ent_train)) | set(ent_train))) == 18

    # distinct_ent = np.union1d(np.union1d(ent_train, ent_valid), ent_test)
    # distinct_rel = np.union1d(np.union1d(np.unique(yago_3_10["train"][:,1]), np.unique(yago_3_10["train"][:,1])),
    #                           np.unique(yago_3_10["train"][:,1]))

    # assert len(distinct_ent) == 123182  
    # assert len(distinct_rel) == 37  


def test_wn18rr():
    wn18rr = load_wn18rr()

    ent_train = np.union1d(np.unique(wn18rr["train"][:, 0]), np.unique(wn18rr["train"][:, 2]))
    ent_valid = np.union1d(np.unique(wn18rr["valid"][:, 0]), np.unique(wn18rr["valid"][:, 2]))
    ent_test = np.union1d(np.unique(wn18rr["test"][:, 0]), np.unique(wn18rr["test"][:, 2]))
    distinct_ent = np.union1d(np.union1d(ent_train, ent_valid), ent_test)
    distinct_rel = np.union1d(np.union1d(np.unique(wn18rr["train"][:, 1]), np.unique(wn18rr["train"][:, 1])),
                              np.unique(wn18rr["train"][:, 1]))

    assert len(wn18rr['train']) == 86835

    # - 210 because 210 triples containing unseen entities are removed
    assert len(wn18rr['valid']) == 3034 - 210

    # - 210 because 210 triples containing unseen entities are removed
    assert len(wn18rr['test']) == 3134 - 210


def test_wn11():
    wn11 = load_wn11(clean_unseen=False)
    assert len(wn11['train']) == 110361
    assert len(wn11['valid']) == 5215
    assert len(wn11['test']) == 21035
    assert len(wn11['valid_labels']) == 5215
    assert len(wn11['test_labels']) == 21035
    assert sum(wn11['valid_labels']) == 2606
    assert sum(wn11['test_labels']) == 10493

    wn11 = load_wn11(clean_unseen=True)
    assert len(wn11['train']) == 110361
    assert len(wn11['valid']) == 5215 - 338
    assert len(wn11['test']) == 21035 - 1329
    assert len(wn11['valid_labels']) == 5215 - 338
    assert len(wn11['test_labels']) == 21035 - 1329
    assert sum(wn11['valid_labels']) == 2409
    assert sum(wn11['test_labels']) == 9706


def test_fb13():
    fb13 = load_fb13(clean_unseen=False)
    assert len(fb13['train']) == 316232
    assert len(fb13['valid']) == 5908 + 5908
    assert len(fb13['test']) == 23733 + 23731
    assert len(fb13['valid_labels']) == 5908 + 5908
    assert len(fb13['test_labels']) == 23733 + 23731
    assert sum(fb13['valid_labels']) == 5908
    assert sum(fb13['test_labels']) == 23733

    fb13 = load_fb13(clean_unseen=True)
    assert len(fb13['train']) == 316232
    assert len(fb13['valid']) == 5908 + 5908
    assert len(fb13['test']) == 23733 + 23731
    assert len(fb13['valid_labels']) == 5908 + 5908
    assert len(fb13['test_labels']) == 23733 + 23731
    assert sum(fb13['valid_labels']) == 5908
    assert sum(fb13['test_labels']) == 23733


def test_oneton_adapter():

    from ampligraph.evaluation.protocol import create_mappings, to_idx

    # Train set
    X = np.array([['a', 'p', 'b'],
                  ['a', 'p', 'd'],
                  ['c', 'p', 'd'],
                  ['c', 'p', 'e'],
                  ['c', 'p', 'f']])

    #              a, b, c, d, e, f
    O = np.array([[0, 1, 0, 1, 0, 0],       # (a, p)
                  [0, 0, 0, 1, 1, 1]])      # (c, p)

    # Test
    T = np.array([['a', 'p', 'c'],
                  ['c', 'p', 'b']])

    #               a, b, c, d, e, f
    OT1 = np.array([[0, 1, 0, 1, 0, 0],    # (a, p)     # test set onehots when output mapping is from train set
                    [0, 0, 0, 1, 1, 1]]),  # (c, p)
    OT2 = np.array([[0, 0, 1, 0, 0, 0],    # (a, p)     # test set onehots when output mapping is from test set
                    [0, 1, 0, 0, 0, 0]]),  # (c, p)

    # Filter
    filter = np.concatenate((X, T))
    #               a, b, c, d, e, f
    OF = np.array([[0, 1, 1, 1, 0, 0],       # (a, p)   # train set onehots when output mapping is from filter
                   [0, 1, 0, 1, 1, 1]])      # (c, p)

    # Expected input tuple to filtered outputs
    OF_map = {(0, 0): [0, 1, 1, 1, 0, 0],
              (2, 0): [0, 1, 0, 1, 1, 1]}

    rel_to_idx, ent_to_idx = create_mappings(X)
    X = to_idx(X, ent_to_idx, rel_to_idx)

    adapter = OneToNDatasetAdapter()
    adapter.use_mappings(rel_to_idx, ent_to_idx)
    adapter.set_data(X, 'train', mapped_status=True)

    adapter.set_data(T, 'test', mapped_status=False)

    # Adapter internally maps test set
    assert (adapter.mapped_status['test'] == True)

    # Generate output map
    train_output_map = adapter.generate_output_mapping('train')

    # Assert all unique sp pairs are in the output_map keys
    unique_sp = set([(s, p) for s, p in X[:, [0, 1]]])
    for sp in train_output_map.keys():
        assert(sp in unique_sp)

    # ValueError if generating onehot outputs before output_mapping is set
    with pytest.raises(ValueError):
        adapter.generate_outputs('train')

    adapter.set_output_mapping(train_output_map)
    adapter.generate_outputs('train')
    train_iter = adapter.get_next_batch(batches_count=1, dataset_type='train', use_filter=False)
    triples, onehot = next(train_iter)
    assert np.all(np.unique(X[:, [0, 1]], axis=0) == triples[:, [0, 1]])
    assert np.all(O == onehot)

    test_iter = adapter.get_next_batch(batches_count=1, dataset_type='test', use_filter=False)

    triples, onehot = next(test_iter)
    assert np.all(np.unique(X[:, [0, 1]], axis=0) == triples[:, [0, 1]])
    assert np.all(OT1 == onehot)

    # Generate test output map
    test_output_map = adapter.generate_output_mapping('test')
    adapter.set_output_mapping(test_output_map)

    test_iter = adapter.get_next_batch(batches_count=1, dataset_type='test', use_filter=False)

    triples, onehot = next(test_iter)
    assert np.all(np.unique(X[:, [0, 1]], axis=0) == triples[:, [0, 1]])
    assert np.all(OT2 == onehot)

    # Train onehot outputs with filter=True
    adapter.set_filter(filter_triples=filter)
    train_iter = adapter.get_next_batch(batches_count=1, dataset_type='train', use_filter=True)
    triples, onehot = next(train_iter)
    assert np.all(np.unique(X[:, [0, 1]], axis=0) == triples[:, [0, 1]])
    assert np.all(OF == onehot)

    ##  Test adapter clear_outputs
    assert(len(adapter.filtered_status) > 0)
    adapter.clear_outputs()
    assert(len(adapter.filtered_status) == 0)

    # Test verify_outputs
    adapter.clear_outputs()
    adapter.generate_outputs('train', use_filter=False, unique_pairs=False)
    assert adapter.verify_outputs('train', use_filter=False, unique_pairs=False) == True
    assert adapter.verify_outputs('train', use_filter=True, unique_pairs=True) == False
    assert adapter.verify_outputs('train', use_filter=True, unique_pairs=False) == False
    assert adapter.verify_outputs('train', use_filter=False, unique_pairs=True) == False

    adapter.clear_outputs()
    adapter.generate_outputs('train', use_filter=True, unique_pairs=True)
    assert adapter.verify_outputs('train', use_filter=False, unique_pairs=False) == False
    assert adapter.verify_outputs('train', use_filter=True, unique_pairs=True) == True
    assert adapter.verify_outputs('train', use_filter=True, unique_pairs=False) == False
    assert adapter.verify_outputs('train', use_filter=False, unique_pairs=True) == False

    # Test batch output shapes
    adapter.clear_outputs()
    train_iter = adapter.get_next_batch(batches_count=1, dataset_type='train', use_filter=True, unique_pairs=True)
    out, triples = next(train_iter)
    assert out.shape[0] == 2
    assert triples.shape[0] == 2

    adapter.clear_outputs()
    train_iter = adapter.get_next_batch(batches_count=1, dataset_type='train', use_filter=True, unique_pairs=False)
    out, triples = next(train_iter)
    assert out.shape[0] == 5
    assert triples.shape[0] == 5

    # Test batch subject corruptions
    batch_size = 3
    batch_iter = adapter.get_next_batch_subject_corruptions(batch_size=batch_size, dataset_type='train', use_filter=True)
    triples, out, out_onehot = next(batch_iter)

    assert np.all(X == triples)  # only 1 relation in X, so triples should be the same (ignores batch_size)
    assert triples.shape[1] == 3  # triples should be triples!
    assert out.shape[1] == 3      # batch out should also be triples!
    assert out.shape[0] == batch_size # batch dimension of out should equal batch_size
    assert out_onehot.shape[0] == batch_size   # Onehot batch_dimension should equal batch size
    assert out_onehot.shape[0] == out.shape[0]  # .. and should be same size as number of unique entities
    assert out_onehot.shape[1] == len(adapter.ent_to_idx)

    adapter.clear_outputs()
    batch_iter = adapter.get_next_batch_subject_corruptions(batch_size=-1, dataset_type='train', use_filter=True)
    triples, out, out_onehot = next(batch_iter)

    assert np.all(X == triples)   # only 1 relation in X, so triples should be the same
    assert triples.shape[1] == 3  # triples should be triples!
    assert out.shape[1] == 3      # batch out should also be triples!
    assert out.shape[0] == len(adapter.ent_to_idx)  # batch_size=-1 (entire set), so the batch_dim should equal ents
    assert out_onehot.shape[0] == out_onehot.shape[1] # Onehot should be a square matrix
    assert out_onehot.shape[0] == out.shape[0]  # .. and should be same size as number of unique entities
    assert out_onehot.shape[1] == len(adapter.ent_to_idx)

    # Verify that onehot outputs are as expected in the out array (or if not present in OF_Map then is a zero vector)
    for idx, (s, p, o) in enumerate(out):
        if (s, p) in OF_map.keys():
            onehot = OF_map[(s, p)]
            assert np.all(onehot == out_onehot[idx])
        else:
            assert np.all(out_onehot[idx] == 0)


if __name__ == '__main__':
    test_oneton_adapter()