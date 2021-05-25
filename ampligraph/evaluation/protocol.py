# Copyright 2019-2021 The AmpliGraph Authors. All Rights Reserved.
#
# This file is Licensed under the Apache License, Version 2.0.
# A copy of the Licence is available in LICENCE, or at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#

from collections.abc import Iterable
from itertools import product, islice
import logging
import warnings

import pandas as pd
import numpy as np
from tqdm import tqdm
import tensorflow as tf

from ..evaluation import mrr_score, hits_at_n_score, mr_score
from ..datasets import AmpligraphDatasetAdapter, NumpyDatasetAdapter, OneToNDatasetAdapter
# from ampligraph.latent_features.models import ConvE

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

TOO_MANY_ENTITIES_TH = 50000


def _train_test_split_no_unseen_fast(X, test_size=100, seed=0, allow_duplication=False, filtered_test_predicates=None):
    """Split into train and test sets.

     This function carves out a test set that contains only entities
     and relations which also occur in the training set.
     
     This is an improved version which is much faster - since this doesnt sample like earlier approach but rather 
     shuffles indices and gets the test set of required size by selecting from the shuffled indices only triples 
     which do not disconnect entities/relations.

    Parameters
    ----------
    X : ndarray, size[n, 3]
        The dataset to split.
    test_size : int, float
        If int, the number of triples in the test set.
        If float, the percentage of total triples.
    seed : int
        A random seed used to split the dataset.
    allow_duplication: boolean
        Flag to indicate if the test set can contain duplicated triples.
    filtered_test_predicates: None, list
        If None, all predicate types will be considered for the test set.
        If list, only the predicate types in the list will be considered for
        the test set.

    Returns
    -------
    X_train : ndarray, size[n, 3]
        The training set.
    X_test : ndarray, size[n, 3]
        The test set.

    Examples
    --------

    >>> import numpy as np
    >>> from ampligraph.evaluation import train_test_split_no_unseen
    >>> # load your dataset to X
    >>> X = np.array([['a', 'y', 'b'],
    >>>               ['f', 'y', 'e'],
    >>>               ['b', 'y', 'a'],
    >>>               ['a', 'y', 'c'],
    >>>               ['c', 'y', 'a'],
    >>>               ['a', 'y', 'd'],
    >>>               ['c', 'y', 'd'],
    >>>               ['b', 'y', 'c'],
    >>>               ['f', 'y', 'e']])
    >>> # if you want to split into train/test datasets
    >>> X_train, X_test = train_test_split_no_unseen(X, test_size=2)
    >>> X_train
    array([['a', 'y', 'd'],
       ['b', 'y', 'a'],
       ['a', 'y', 'c'],
       ['f', 'y', 'e'],
       ['a', 'y', 'b'],
       ['c', 'y', 'a'],
       ['b', 'y', 'c']], dtype='<U1')
    >>> X_test
    array([['f', 'y', 'e'],
       ['c', 'y', 'd']], dtype='<U1')
    >>> # if you want to split into train/valid/test datasets, call it 2 times
    >>> X_train_valid, X_test = train_test_split_no_unseen(X, test_size=2)
    >>> X_train, X_valid = train_test_split_no_unseen(X_train_valid, test_size=2)
    >>> X_train
    array([['a', 'y', 'b'],
       ['a', 'y', 'd'],
       ['a', 'y', 'c'],
       ['c', 'y', 'a'],
       ['f', 'y', 'e']], dtype='<U1')
    >>> X_valid
    array([['c', 'y', 'd'],
       ['f', 'y', 'e']], dtype='<U1')
    >>> X_test
    array([['b', 'y', 'c'],
       ['b', 'y', 'a']], dtype='<U1')
    """
    
    if type(test_size) is float:
        test_size = int(len(X) * test_size)

    np.random.seed(seed)
    if filtered_test_predicates:
        candidate_idx = np.isin(X[:, 1], filtered_test_predicates)
        X_test_candidates = X[candidate_idx]
        X_train = X[~candidate_idx]
    else:
        X_train = None
        X_test_candidates = X

    entities, entity_cnt = np.unique(np.concatenate([X_test_candidates[:, 0], 
                                                     X_test_candidates[:, 2]]), return_counts=True)
    rels, rels_cnt = np.unique(X_test_candidates[:, 1], return_counts=True)
    dict_entities = dict(zip(entities, entity_cnt))
    dict_rels = dict(zip(rels, rels_cnt))
    idx_test = []
    idx_train = []
    
    all_indices_shuffled = np.random.permutation(np.arange(X_test_candidates.shape[0]))

    for i, idx in enumerate(all_indices_shuffled):
        test_triple = X_test_candidates[idx]
        # reduce the entity and rel count
        dict_entities[test_triple[0]] = dict_entities[test_triple[0]] - 1
        dict_rels[test_triple[1]] = dict_rels[test_triple[1]] - 1
        dict_entities[test_triple[2]] = dict_entities[test_triple[2]] - 1

        # test if the counts are > 0
        if dict_entities[test_triple[0]] > 0 and \
                dict_rels[test_triple[1]] > 0 and \
                dict_entities[test_triple[2]] > 0:
            
            # Can safetly add the triple to test set
            idx_test.append(idx)
            if len(idx_test) == test_size:
                # Since we found the requested test set of given size
                # add all the remaining indices of candidates to training set
                idx_train.extend(list(all_indices_shuffled[i + 1:]))
                
                # break out of the loop
                break
            
        else:
            # since removing this triple results in unseen entities, add it to training
            dict_entities[test_triple[0]] = dict_entities[test_triple[0]] + 1
            dict_rels[test_triple[1]] = dict_rels[test_triple[1]] + 1
            dict_entities[test_triple[2]] = dict_entities[test_triple[2]] + 1
            idx_train.append(idx)
            
    if len(idx_test) != test_size:
        # if we cannot get the test set of required size that means we cannot get unique triples
        # in the test set without creating unseen entities
        if allow_duplication:
            # if duplication is allowed, randomly choose from the existing test set and create duplicates
            duplicate_idx = np.random.choice(idx_test, size=(test_size - len(idx_test))).tolist()
            idx_test.extend(list(duplicate_idx))
        else:
            # throw an exception since we cannot get unique triples in the test set without creating 
            # unseen entities
            raise Exception("Cannot create a test split of the desired size. "
                            "Some entities will not occur in both training and test set. "
                            "Set allow_duplication=True," 
                            "remove filter on test predicates or "
                            "set test_size to a smaller value.")
    
    if X_train is None:
        X_train = X_test_candidates[idx_train]
    else:
        X_train_subset = X_test_candidates[idx_train]
        X_train = np.concatenate([X_train, X_train_subset])
    X_test = X_test_candidates[idx_test]
    
    X_train = np.random.permutation(X_train)
    X_test = np.random.permutation(X_test)

    return X_train, X_test 
    

def _train_test_split_no_unseen_old(X, test_size=100, seed=0, allow_duplication=False, filtered_test_predicates=None):
    """Split into train and test sets.

     This function carves out a test set that contains only entities
     and relations which also occur in the training set.
     
     This is very slow as it runs an infinite loop and samples a triples and appends to test set and checks if it is 
     unique or not. This is very time consuming process and highly inefficient.

    Parameters
    ----------
    X : ndarray, size[n, 3]
        The dataset to split.
    test_size : int, float
        If int, the number of triples in the test set.
        If float, the percentage of total triples.
    seed : int
        A random seed used to split the dataset.
    allow_duplication: boolean
        Flag to indicate if the test set can contain duplicated triples.
    filtered_test_predicates: None, list
        If None, all predicate types will be considered for the test set.
        If list, only the predicate types in the list will be considered for
        the test set.

    Returns
    -------
    X_train : ndarray, size[n, 3]
        The training set.
    X_test : ndarray, size[n, 3]
        The test set.

    Examples
    --------

    >>> import numpy as np
    >>> from ampligraph.evaluation import train_test_split_no_unseen
    >>> # load your dataset to X
    >>> X = np.array([['a', 'y', 'b'],
    >>>               ['f', 'y', 'e'],
    >>>               ['b', 'y', 'a'],
    >>>               ['a', 'y', 'c'],
    >>>               ['c', 'y', 'a'],
    >>>               ['a', 'y', 'd'],
    >>>               ['c', 'y', 'd'],
    >>>               ['b', 'y', 'c'],
    >>>               ['f', 'y', 'e']])
    >>> # if you want to split into train/test datasets
    >>> X_train, X_test = train_test_split_no_unseen(X, test_size=2, backward_compatible=True)
    >>> X_train
    array([['a', 'y', 'b'],
        ['f', 'y', 'e'],
        ['b', 'y', 'a'],
        ['c', 'y', 'a'],
        ['c', 'y', 'd'],
        ['b', 'y', 'c'],
        ['f', 'y', 'e']], dtype='<U1')
    >>> X_test
    array([['a', 'y', 'c'],
        ['a', 'y', 'd']], dtype='<U1')
    >>> # if you want to split into train/valid/test datasets, call it 2 times
    >>> X_train_valid, X_test = train_test_split_no_unseen(X, test_size=2, backward_compatible=True)
    >>> X_train, X_valid = train_test_split_no_unseen(X_train_valid, test_size=2, backward_compatible=True)
    >>> X_train
    array([['a', 'y', 'b'],
        ['b', 'y', 'a'],
        ['c', 'y', 'd'],
        ['b', 'y', 'c'],
        ['f', 'y', 'e']], dtype='<U1')
    >>> X_valid
    array([['f', 'y', 'e'],
        ['c', 'y', 'a']], dtype='<U1')
    >>> X_test
    array([['a', 'y', 'c'],
        ['a', 'y', 'd']], dtype='<U1')
    """

    logger.debug('Creating train test split.')
    if type(test_size) is float:
        logger.debug('Test size is of type float. Converting to int.')
        test_size = int(len(X) * test_size)

    rnd = np.random.RandomState(seed)

    subs, subs_cnt = np.unique(X[:, 0], return_counts=True)
    objs, objs_cnt = np.unique(X[:, 2], return_counts=True)
    rels, rels_cnt = np.unique(X[:, 1], return_counts=True)
    dict_subs = dict(zip(subs, subs_cnt))
    dict_objs = dict(zip(objs, objs_cnt))
    dict_rels = dict(zip(rels, rels_cnt))

    idx_test = np.array([], dtype=int)
    logger.debug('Selecting test cases using random search.')

    loop_count = 0
    tolerance = len(X) * 10
    # Set the indices of test set triples. If filtered, reduce candidate triples to certain predicate types.
    if filtered_test_predicates:
        test_triples_idx = np.where(np.isin(X[:, 1], filtered_test_predicates))[0]
    else:
        test_triples_idx = np.arange(len(X))

    while idx_test.shape[0] < test_size:
        i = rnd.choice(test_triples_idx)
        if dict_subs[X[i, 0]] > 1 and dict_objs[X[i, 2]] > 1 and dict_rels[X[i, 1]] > 1:
            dict_subs[X[i, 0]] -= 1
            dict_objs[X[i, 2]] -= 1
            dict_rels[X[i, 1]] -= 1
            if allow_duplication:
                idx_test = np.append(idx_test, i)
            else:
                idx_test = np.unique(np.append(idx_test, i))

        loop_count += 1

        # in case can't find solution
        if loop_count == tolerance:
            if allow_duplication:
                raise Exception("Cannot create a test split of the desired size. "
                                "Some entities will not occur in both training and test set. "
                                "Change seed values, remove filter on test predicates or set "
                                "test_size to a smaller value.")
            else:
                raise Exception("Cannot create a test split of the desired size. "
                                "Some entities will not occur in both training and test set. "
                                "Set allow_duplication=True,"
                                "change seed values, remove filter on test predicates or "
                                "set test_size to a smaller value.")

    logger.debug('Completed random search.')

    idx = np.arange(len(X))
    idx_train = np.setdiff1d(idx, idx_test)
    logger.debug('Train test split completed.')

    return X[idx_train, :], X[idx_test, :]


def train_test_split_no_unseen(X, test_size=100, seed=0, allow_duplication=False, 
                               filtered_test_predicates=None, backward_compatible=False):
    """Split into train and test sets.

     This function carves out a test set that contains only entities
     and relations which also occur in the training set.

    Parameters
    ----------
    X : ndarray, size[n, 3]
        The dataset to split.
    test_size : int, float
        If int, the number of triples in the test set.
        If float, the percentage of total triples.
    seed : int
        A random seed used to split the dataset.
    allow_duplication: boolean
        Flag to indicate if the test set can contain duplicated triples.
    filtered_test_predicates: None, list
        If None, all predicate types will be considered for the test set.
        If list, only the predicate types in the list will be considered for
        the test set.
    backward_compatible: boolean
        Uses the old (slower) version of the API for reproducibility of splits in older pipelines(if any)
        Avoid setting this to True, unless necessary. Set this flag only if you want to use the 
        train_test_split_no_unseen of Ampligraph versions 1.3.2 and below. The older version is slow and inefficient

    Returns
    -------
    X_train : ndarray, size[n, 3]
        The training set.
    X_test : ndarray, size[n, 3]
        The test set.

    Examples
    --------

    >>> import numpy as np
    >>> from ampligraph.evaluation import train_test_split_no_unseen
    >>> # load your dataset to X
    >>> X = np.array([['a', 'y', 'b'],
    >>>               ['f', 'y', 'e'],
    >>>               ['b', 'y', 'a'],
    >>>               ['a', 'y', 'c'],
    >>>               ['c', 'y', 'a'],
    >>>               ['a', 'y', 'd'],
    >>>               ['c', 'y', 'd'],
    >>>               ['b', 'y', 'c'],
    >>>               ['f', 'y', 'e']])
    >>> # if you want to split into train/test datasets
    >>> X_train, X_test = train_test_split_no_unseen(X, test_size=2)
    >>> X_train
    array([['a', 'y', 'd'],
       ['b', 'y', 'a'],
       ['a', 'y', 'c'],
       ['f', 'y', 'e'],
       ['a', 'y', 'b'],
       ['c', 'y', 'a'],
       ['b', 'y', 'c']], dtype='<U1')
    >>> X_test
    array([['f', 'y', 'e'],
       ['c', 'y', 'd']], dtype='<U1')
    >>> # if you want to split into train/valid/test datasets, call it 2 times
    >>> X_train_valid, X_test = train_test_split_no_unseen(X, test_size=2)
    >>> X_train, X_valid = train_test_split_no_unseen(X_train_valid, test_size=2)
    >>> X_train
    array([['a', 'y', 'b'],
       ['a', 'y', 'd'],
       ['a', 'y', 'c'],
       ['c', 'y', 'a'],
       ['f', 'y', 'e']], dtype='<U1')
    >>> X_valid
    array([['c', 'y', 'd'],
       ['f', 'y', 'e']], dtype='<U1')
    >>> X_test
    array([['b', 'y', 'c'],
       ['b', 'y', 'a']], dtype='<U1')
    """
    if backward_compatible:
        return _train_test_split_no_unseen_old(X, test_size, seed, allow_duplication, filtered_test_predicates)
    
    return _train_test_split_no_unseen_fast(X, test_size, seed, allow_duplication, filtered_test_predicates)


def _create_unique_mappings(unique_obj, unique_rel):
    obj_count = len(unique_obj)
    rel_count = len(unique_rel)
    rel_to_idx = dict(zip(unique_rel, range(rel_count)))
    obj_to_idx = dict(zip(unique_obj, range(obj_count)))
    return rel_to_idx, obj_to_idx


def create_mappings(X):
    """Create string-IDs mappings for entities and relations.

    Entities and relations are assigned incremental, unique integer IDs.
    Mappings are preserved in two distinct dictionaries,
    and counters are separated for entities and relations mappings.

    Parameters
    ----------
    X : ndarray, shape [n, 3]
        The triples to extract mappings.

    Returns
    -------
    rel_to_idx : dict
        The relation-to-internal-id associations.
    ent_to_idx: dict
        The entity-to-internal-id associations.

    """
    logger.debug('Creating mappings for entities and relations.')
    unique_ent = np.unique(np.concatenate((X[:, 0], X[:, 2])))
    unique_rel = np.unique(X[:, 1])
    return _create_unique_mappings(unique_ent, unique_rel)


def generate_corruptions_for_eval(X, entities_for_corruption, corrupt_side='s,o'):
    """Generate corruptions for evaluation.

        Create corruptions (subject and object) for a given triple x, in compliance with the
        local closed world assumption (LCWA), as described in :cite:`nickel2016review`.

    Parameters
    ----------
    X : Tensor, shape [1, 3]
        Currently, a single positive triples that will be used to create corruptions.
    entities_for_corruption : Tensor
        All the entity IDs which are to be used for generation of corruptions.
    corrupt_side: string
        Specifies which side of the triple to corrupt:

        - 's': corrupt only subject.
        - 'o': corrupt only object
        - 's+o': corrupt both subject and object
        - 's,o': corrupt both subject and object but ranks are computed separately.

    Returns
    -------
    out : Tensor, shape [n, 3]
        An array of corruptions for the triples for x.

    """

    logger.debug('Generating corruptions for evaluation.')

    logger.debug('Getting repeating subjects.')
    if corrupt_side == 's,o':
        # Both subject and object are corrupted but ranks are computed separately.
        corrupt_side = 's+o'

    if corrupt_side not in ['s+o', 's', 'o']:
        msg = 'Invalid argument value for corruption side passed for evaluation'
        logger.error(msg)
        raise ValueError(msg)

    if corrupt_side in ['s+o', 'o']:  # object is corrupted - so we need subjects as it is
        repeated_subjs = tf.keras.backend.repeat(
            tf.slice(X,
                     [0, 0],  # subj
                     [tf.shape(X)[0], 1]),
            tf.shape(entities_for_corruption)[0])
        repeated_subjs = tf.squeeze(repeated_subjs, 2)

    logger.debug('Getting repeating object.')
    if corrupt_side in ['s+o', 's']:  # subject is corrupted - so we need objects as it is
        repeated_objs = tf.keras.backend.repeat(
            tf.slice(X,
                     [0, 2],  # Obj
                     [tf.shape(X)[0], 1]),
            tf.shape(entities_for_corruption)[0])
        repeated_objs = tf.squeeze(repeated_objs, 2)

    logger.debug('Getting repeating relationships.')
    repeated_relns = tf.keras.backend.repeat(
        tf.slice(X,
                 [0, 1],  # reln
                 [tf.shape(X)[0], 1]),
        tf.shape(entities_for_corruption)[0])
    repeated_relns = tf.squeeze(repeated_relns, 2)

    rep_ent = tf.keras.backend.repeat(tf.expand_dims(entities_for_corruption, 0), tf.shape(X)[0])
    rep_ent = tf.squeeze(rep_ent, 0)

    if corrupt_side == 's+o':
        stacked_out = tf.concat([tf.stack([repeated_subjs, repeated_relns, rep_ent], 1),
                                 tf.stack([rep_ent, repeated_relns, repeated_objs], 1)], 0)

    elif corrupt_side == 'o':
        stacked_out = tf.stack([repeated_subjs, repeated_relns, rep_ent], 1)

    else:
        stacked_out = tf.stack([rep_ent, repeated_relns, repeated_objs], 1)

    out = tf.reshape(tf.transpose(stacked_out, [0, 2, 1]), (-1, 3))

    return out


def generate_corruptions_for_fit(X, entities_list=None, eta=1, corrupt_side='s,o', entities_size=0, rnd=None):
    """Generate corruptions for training.

    Creates corrupted triples for each statement in an array of statements,
    as described by :cite:`trouillon2016complex`.

    .. note::
        Collisions are not checked, as this will be computationally expensive :cite:`trouillon2016complex`.
        That means that some corruptions *may* result in being positive statements (i.e. *unfiltered* settings).

    .. note::
        When processing large knowledge graphs, it may be useful to generate corruptions only using entities from
        a single batch.
        This also brings the benefit of creating more meaningful negatives, as entities used to corrupt are
        sourced locally.
        The function can be configured to generate corruptions *only* using the entities from the current batch.
        You can enable such behaviour be setting ``entities_size=0``. In such case, if ``entities_list=None``
        all entities from the *current batch* will be used to generate corruptions.

    Parameters
    ----------
    X : Tensor, shape [n, 3]
        An array of positive triples that will be used to create corruptions.
    entities_list : list
        List of entities to be used for generating corruptions. (default:None).

        If ``entities_list=None`` and ``entities_size`` is the number of all entities,
        all entities will be used to generate corruptions (default behaviour).

        If ``entities_list=None`` and ``entities_size=0``, the batch entities will be used to generate corruptions.
    eta : int
        The number of corruptions per triple that must be generated.
    corrupt_side: string
        Specifies which side of the triple to corrupt:

        - 's': corrupt only subject.
        - 'o': corrupt only object
        - 's+o': corrupt both subject and object
        - 's,o': corrupt both subject and object
    entities_size: int
        Size of entities to be used while generating corruptions. It assumes entity id's start from 0 and are
        continuous. (default: 0).
        When processing large knowledge graphs, it may be useful to generate corruptions only using entities from
        a single batch.
        This also brings the benefit of creating more meaningful negatives, as entities used to corrupt are
        sourced locally.
        The function can be configured to generate corruptions *only* using the entities from the current batch.
        You can enable such behaviour be setting ``entities_size=0``. In such case, if ``entities_list=None``
        all entities from the *current batch* will be used to generate corruptions.
    rnd: numpy.random.RandomState
        A random number generator.

    Returns
    -------

    out : Tensor, shape [n * eta, 3]
        An array of corruptions for a list of positive triples X. For each row in X the corresponding corruption
        indexes can be found at [index+i*n for i in range(eta)]

    """
    logger.debug('Generating corruptions for fit.')
    if corrupt_side == 's,o':
        # Both subject and object are corrupted but ranks are computed separately.
        corrupt_side = 's+o'

    if corrupt_side not in ['s+o', 's', 'o']:
        msg = 'Invalid argument value {} for corruption side passed for evaluation.'.format(corrupt_side)
        logger.error(msg)
        raise ValueError(msg)

    dataset = tf.reshape(tf.tile(tf.reshape(X, [-1]), [eta]), [tf.shape(X)[0] * eta, 3])

    if corrupt_side == 's+o':
        keep_subj_mask = tf.cast(tf.random_uniform([tf.shape(X)[0] * eta], 0, 2, dtype=tf.int32, seed=rnd), tf.bool)
    else:
        keep_subj_mask = tf.cast(tf.ones(tf.shape(X)[0] * eta, tf.int32), tf.bool)
        if corrupt_side == 's':
            keep_subj_mask = tf.logical_not(keep_subj_mask)

    keep_obj_mask = tf.logical_not(keep_subj_mask)
    keep_subj_mask = tf.cast(keep_subj_mask, tf.int32)
    keep_obj_mask = tf.cast(keep_obj_mask, tf.int32)

    logger.debug('Created corruption masks.')

    if entities_size != 0:
        replacements = tf.random_uniform([tf.shape(dataset)[0]], 0, entities_size, dtype=tf.int32, seed=rnd)
    else:
        if entities_list is None:
            # use entities in the batch
            entities_list, _ = tf.unique(tf.squeeze(
                tf.concat([tf.slice(X, [0, 0], [tf.shape(X)[0], 1]),
                           tf.slice(X, [0, 2], [tf.shape(X)[0], 1])],
                          0)))

        random_indices = tf.random.uniform(shape=(tf.shape(dataset)[0],),
                                           maxval=tf.shape(entities_list)[0],
                                           dtype=tf.int32,
                                           seed=rnd)
        replacements = tf.gather(entities_list, random_indices)

    subjects = tf.math.add(tf.math.multiply(keep_subj_mask, dataset[:, 0]),
                           tf.math.multiply(keep_obj_mask, replacements))
    logger.debug('Created corrupted subjects.')
    relationships = dataset[:, 1]
    logger.debug('Retained relationships.')
    objects = tf.math.add(tf.math.multiply(keep_obj_mask, dataset[:, 2]),
                          tf.math.multiply(keep_subj_mask, replacements))
    logger.debug('Created corrupted objects.')

    out = tf.transpose(tf.stack([subjects, relationships, objects]))

    logger.debug('Returning corruptions for fit.')
    return out


def _convert_to_idx(X, ent_to_idx, rel_to_idx, obj_to_idx):
    unseen_msg = 'Input triples include one or more {concept_type} not present in the training set. ' \
                 'Please filter all concepts in X that do not occur in the training test ' \
                 '(set filter_unseen=True in evaluate_performance) or retrain the model on a ' \
                 'training set that includes all the desired concept types.'

    try:
        x_idx_s = np.vectorize(ent_to_idx.get)(X[:, 0])
        x_idx_p = np.vectorize(rel_to_idx.get)(X[:, 1])
        x_idx_o = np.vectorize(obj_to_idx.get)(X[:, 2])
    except TypeError:
        unseen_msg = unseen_msg.format(**{'concept_type': 'concepts'})
        logger.error(unseen_msg)
        raise ValueError(unseen_msg)

    if None in x_idx_s or None in x_idx_o:
        unseen_msg = unseen_msg.format(**{'concept_type': 'entities'})
        logger.error(unseen_msg)
        raise ValueError(unseen_msg)

    if None in x_idx_p:
        unseen_msg = unseen_msg.format(**{'concept_type': 'relations'})
        logger.error(unseen_msg)
        raise ValueError(unseen_msg)

    return np.dstack([x_idx_s, x_idx_p, x_idx_o]).reshape((-1, 3))


def to_idx(X, ent_to_idx, rel_to_idx):
    """Convert statements (triples) into integer IDs.

    Parameters
    ----------
    X : ndarray
        The statements to be converted.
    ent_to_idx : dict
        The mappings between entity strings and internal IDs.
    rel_to_idx : dict
        The mappings between relation strings and internal IDs.
    Returns
    -------
    X : ndarray, shape [n, 3]
        The ndarray of converted statements.
    """
    logger.debug('Converting statements to integer ids.')
    if X.ndim == 1:
        X = X[np.newaxis, :]
    return _convert_to_idx(X, ent_to_idx, rel_to_idx, ent_to_idx)


def evaluate_performance(X, model, filter_triples=None, verbose=False, filter_unseen=True, entities_subset=None,
                         corrupt_side='s,o', ranking_strategy='worst', use_default_protocol=False):
    """Evaluate the performance of an embedding model.

    The evaluation protocol follows the procedure defined in :cite:`bordes2013translating` and can be summarised as:

    #. Artificially generate negative triples by corrupting first the subject and then the object.

    #. Remove the positive triples from the set returned by (1) -- positive triples \
    are usually the concatenation of training, validation and test sets.

    #. Rank each test triple against all remaining triples returned by (2).


    With the ranks of both object and subject corruptions, one may compute metrics such as the MRR by
    calculating them separately and then averaging them out.
    Note that the metrics implemented in AmpliGraph's ``evaluate.metrics`` module will already work that way
    when provided with the input returned by ``evaluate_performance``.

    The artificially generated negatives are compliant with the local closed world assumption (LCWA),
    as described in :cite:`nickel2016review`. In practice, that means only one side of the triple is corrupted at a time
    (i.e. either the subject or the object).

    .. note::
        The evaluation protocol assigns the worst rank
        to a positive test triple in case of a tie with negatives. This is the agreed upon behaviour in literature.

    .. hint::
        When ``entities_subset=None``, the method will use all distinct entities in the knowledge graph ``X``
        to generate negatives to rank against. This might slow down the eval. Some of the corruptions may not even
        make sense for the task that one may be interested in.

        For eg, consider the case <Actor, acted_in, ?>, where we are mainly interested in such movies that an actor
        has acted in. A sensible way to evaluate this would be to rank against all the movie entities and compute
        the desired metrics. In such cases, where focus us on particular task, it is recommended to pass the desired
        entities to use to generate corruptions to ``entities_subset``. Besides, trying to rank a positive against an
        extremely large number of negatives may be overkilling.

        As a reference, the popular FB15k-237 dataset has ~15k distinct entities. The evaluation protocol ranks each
        positives against 15k corruptions per side.

    Parameters
    ----------
    X : ndarray, shape [n, 3]
        An array of test triples.
    model : EmbeddingModel
        A knowledge graph embedding model
    filter_triples : ndarray of shape [n, 3] or None
        The triples used to filter negatives.

        .. note::
            When *filtered* mode is enabled (i.e. `filtered_triples` is not ``None``),
            to speed up the procedure, we use a database based filtering. This strategy is as described below:

            * Store the filter_triples in the DB
            * For each test triple, we generate corruptions for evaluation and score them.
            * The corruptions may contain some False Negatives. We find such statements by quering the database.
            * From the computed scores we retrieve the scores of the False Negatives.
            * We compute the rank of the test triple by comparing against ALL the corruptions.
            * We then compute the number of False negatives that are ranked higher than the test triple; and then
              subtract this value from the above computed rank to yield the final filtered rank.

            **Execution Time:** This method takes ~4 minutes on FB15K using ComplEx
            (Intel Xeon Gold 6142, 64 GB Ubuntu 16.04 box, Tesla V100 16GB)

    verbose : bool
        Verbose mode
    filter_unseen : bool
        This can be set to False to skip filtering of unseen entities if train_test_split_unseen() was used to
        split the original dataset.

    entities_subset: array-like
        List of entities to use for corruptions. If None, will generate corruptions
        using all distinct entities. Default is None.
    corrupt_side: string
        Specifies which side of the triple to corrupt:

        - 's': corrupt only subject.
        - 'o': corrupt only object.
        - 's+o': corrupt both subject and object.
        - 's,o': corrupt subject and object sides independently and return 2 ranks. This corresponds to the \
        evaluation protocol used in literature, where head and tail corruptions are evaluated separately.

        .. note::
            When ``corrupt_side='s,o'`` the function will return 2*n ranks as a [n, 2] array.
            The first column of the array represents the subject corruptions.
            The second column of the array represents the object corruptions.
            Otherwise, the function returns n ranks as [n] array.

    ranking_strategy: string
        Specifies the type of score comparison strategy to use while ranking:

        - 'worst': assigns the worst rank when scores are equal
        - 'best': assigns the best rank when scores are equal
        - 'middle': assigns the middle rank when scores are equal

        Our recommendation is to use ``worst``.
        Think of a model which assigns constant score to any triples. If you use the ``best`` strategy then 
        the ranks will always be 1 (which is incorrect because the model has not learnt anything). If you choose 
        this model and try to do knowledge discovery, you will not be able to deduce anything as all triples will 
        get the same scores. So to be on safer side while choosing the model, we would recommend either ``worst``
        or ``middle`` strategy.

    use_default_protocol: bool
        Flag to indicate whether to use the standard protocol used in literature defined in
        :cite:`bordes2013translating` (default: False).
        If set to `True`, ``corrupt_side`` will be set to `'s,o'`.
        This corresponds to the evaluation protocol used in literature, where head and tail corruptions
        are evaluated separately, i.e. in corrupt_side='s,o' mode

    Returns
    -------
    ranks : ndarray, shape [n] or [n,2] depending on the value of corrupt_side.
        An array of ranks of test triples.
        When ``corrupt_side='s,o'`` the function returns [n,2]. The first column represents the rank against
        subject corruptions and the second column represents the rank against object corruptions.
        In other cases, it returns [n] i.e. rank against the specified corruptions.

    Examples
    --------
    >>> import numpy as np
    >>> from ampligraph.datasets import load_wn18
    >>> from ampligraph.latent_features import ComplEx
    >>> from ampligraph.evaluation import evaluate_performance, mrr_score, hits_at_n_score
    >>>
    >>> X = load_wn18()
    >>> model = ComplEx(batches_count=10, seed=0, epochs=10, k=150, eta=1,
    >>>                 loss='nll', optimizer='adam')
    >>> model.fit(np.concatenate((X['train'], X['valid'])))
    >>>
    >>> filter_triples = np.concatenate((X['train'], X['valid'], X['test']))
    >>> ranks = evaluate_performance(X['test'][:5], model=model,
    >>>                              filter_triples=filter_triples,
    >>>                              corrupt_side='s+o',
    >>>                              use_default_protocol=False)
    >>> ranks
    array([  1, 582, 543,   6,  31])
    >>> mrr_score(ranks)
    0.24049691297347323
    >>> hits_at_n_score(ranks, n=10)
    0.4
    """

    from ampligraph.latent_features import ConvE  # avoids circular import hell

    dataset_handle = None

    # try-except block is mainly to handle clean up in case of exception or manual stop in jupyter notebook
    try:
        if use_default_protocol:
            logger.warning('DeprecationWarning: use_default_protocol will be removed in future. '
                           'Please use corrupt_side argument instead.')
            corrupt_side = 's,o'

        logger.debug('Evaluating the performance of the embedding model.')
        assert corrupt_side in ['s', 'o', 's+o', 's,o'], 'Invalid value for corrupt_side.'
        if isinstance(X, np.ndarray):

            if filter_unseen:
                X = filter_unseen_entities(X, model, verbose=verbose)
            else:
                logger.warning("If your test set or filter triples contain unseen entities you may get a"
                               "runtime error. You can filter them by setting filter_unseen=True")

            if isinstance(model, ConvE):
                dataset_handle = OneToNDatasetAdapter()
            else:
                dataset_handle = NumpyDatasetAdapter()

            dataset_handle.use_mappings(model.rel_to_idx, model.ent_to_idx)
            dataset_handle.set_data(X, 'test')

        elif isinstance(X, AmpligraphDatasetAdapter):
            dataset_handle = X
        else:
            msg = "X must be either a numpy array or an AmpligraphDatasetAdapter."
            logger.error(msg)
            raise ValueError(msg)

        if filter_triples is not None:
            if isinstance(filter_triples, np.ndarray):
                logger.debug('Getting filtered triples.')

                if filter_unseen:
                    filter_triples = filter_unseen_entities(filter_triples, model, verbose=verbose)
                dataset_handle.set_filter(filter_triples)
                model.set_filter_for_eval()
            elif isinstance(X, AmpligraphDatasetAdapter):
                if not isinstance(filter_triples, bool):
                    raise Exception('Expected a boolean type')
                if filter_triples is True:
                    model.set_filter_for_eval()
            else:
                raise Exception('Invalid datatype for filter. Expected a numpy array or preset data in the adapter.')

        eval_dict = {}

        # #186: print warning when trying to evaluate with too many entities.
        #      Thus will likely result in shooting in your feet, as the protocol will be excessively hard.
        check_filter_size(model, entities_subset)

        if entities_subset is not None:
            idx_entities = np.asarray([idx for uri, idx in model.ent_to_idx.items() if uri in entities_subset])
            eval_dict['corruption_entities'] = idx_entities

        logger.debug('Evaluating the test set by corrupting side : {}'.format(corrupt_side))
        eval_dict['corrupt_side'] = corrupt_side

        assert ranking_strategy in ['worst', 'best', 'middle'], 'Invalid ranking_strategy!'

        eval_dict['ranking_strategy'] = ranking_strategy

        logger.debug('Configuring evaluation protocol.')
        model.configure_evaluation_protocol(eval_dict)

        logger.debug('Making predictions.')
        ranks = model.get_ranks(dataset_handle)

        logger.debug('Ending Evaluation')
        model.end_evaluation()

        logger.debug('Returning ranks of positive test triples obtained by corrupting {}.'.format(corrupt_side))
        return np.array(ranks)

    except BaseException as e:
        model.end_evaluation()
        if dataset_handle is not None:
            dataset_handle.cleanup()
        raise e


def check_filter_size(model, corruption_entities):
    """ Raise a warning when trying to evaluate with too many entities.

        Doing so will likely result in shooting in your feet, as the protocol will be excessively hard,
        hence the warning message.

        Addresses #186.

    Parameters
    ----------
    model : the model
    corruption_entities : the corruption_entities used in the protocol

    Returns
    -------
    None.

    """

    warn_msg = """You are attempting to use %d distinct entities to generate synthetic negatives in the evaluation
    protocol. This may be unnecessary and will lead to a 'harder' task. Besides, it will lead to a much slower
    evaluation procedure. We recommended to set the 'corruption_entities' argument to a reasonably sized set
    of entities. The size of corruption_entities depends on your domain-specific task."""

    if corruption_entities is None:
        ent_for_corruption_size = len(model.ent_to_idx)
    else:
        ent_for_corruption_size = len(corruption_entities)

    if ent_for_corruption_size >= TOO_MANY_ENTITIES_TH:
        warnings.warn(warn_msg % ent_for_corruption_size)
        logger.warning(warn_msg, ent_for_corruption_size)


def filter_unseen_entities(X, model, verbose=False):
    """Filter unseen entities in the test set.

    Parameters
    ----------
    X : ndarray, shape [n, 3]
        An array of test triples.
    model : ampligraph.latent_features.EmbeddingModel
        A knowledge graph embedding model.
    verbose : bool
        Verbose mode.

    Returns
    -------
    filtered X : ndarray, shape [n, 3]
        An array of test triples containing no unseen entities.
    """
    logger.debug('Finding entities in the dataset that are not previously seen by model')
    ent_seen = np.unique(list(model.ent_to_idx.keys()))
    df = pd.DataFrame(X, columns=['s', 'p', 'o'])
    filtered_df = df[df.s.isin(ent_seen) & df.o.isin(ent_seen)]
    n_removed_ents = df.shape[0] - filtered_df.shape[0]
    if n_removed_ents > 0:
        msg = 'Removing {} triples containing unseen entities. '.format(n_removed_ents)
        if verbose:
            logger.info(msg)
        logger.debug(msg)
        return filtered_df.values
    return X


def _remove_unused_params(params):
    """
    Removed unused parameters considering the registries.

    For example, if the regularization is None, there is no need for the regularization parameter lambda.

    Parameters
    ----------
    params: dict
        Dictionary with parameters.

    Returns
    -------
    params: dict
        Param dict without unused parameters.
    """
    from ..latent_features import LOSS_REGISTRY, REGULARIZER_REGISTRY, MODEL_REGISTRY, \
        OPTIMIZER_REGISTRY, INITIALIZER_REGISTRY

    def _param_without_unused(param, registry, category_type, category_type_params):
        """Remove one particular nested param (if unused) given a registry"""
        if category_type_params in param and category_type in registry:
            expected_params = registry[category_type].external_params
            params[category_type_params] = {k: v for k, v in param[category_type_params].items() if
                                            k in expected_params}
        else:
            params[category_type_params] = {}

    params = params.copy()

    if "loss" in params and "loss_params" in params:
        _param_without_unused(params, LOSS_REGISTRY, params["loss"], "loss_params")
    if "regularizer" in params and "regularizer_params" in params:
        _param_without_unused(params, REGULARIZER_REGISTRY, params["regularizer"], "regularizer_params")
    if "optimizer" in params and "optimizer_params" in params:
        _param_without_unused(params, OPTIMIZER_REGISTRY, params["optimizer"], "optimizer_params")
    if "initializer" in params and "initializer_params" in params:
        _param_without_unused(params, INITIALIZER_REGISTRY, params["initializer"], "initializer_params")
    if "embedding_model_params" in params and "model_name" in params:
        _param_without_unused(params, MODEL_REGISTRY, params["model_name"], "embedding_model_params")

    return params


def _flatten_nested_keys(dictionary):
    """
    Flatten the nested values of a dictionary into tuple keys
    E.g. {"a": {"b": [1], "c": [2]}} becomes {("a", "b"): [1], ("a", "c"): [2]}
    """
    # Find the parameters that are nested dictionaries
    nested_keys = {k for k, v in dictionary.items() if type(v) is dict}
    # Flatten them into tuples
    flattened_nested_keys = {(nk, k): dictionary[nk][k] for nk in nested_keys for k in dictionary[nk]}
    # Get original dictionary without the nested keys
    dictionary_without_nested_keys = {k: v for k, v in dictionary.items() if k not in nested_keys}
    # Return merged dicts
    return {**dictionary_without_nested_keys, **flattened_nested_keys}


def _unflatten_nested_keys(dictionary):
    """
    Unflatten the nested values of a dictionary based on the keys that are tuples
    E.g. {("a", "b"): [1], ("a", "c"): [2]} becomes {"a": {"b": [1], "c": [2]}}
    """
    # Find the parameters that are nested dictionaries
    nested_keys = {k[0] for k in dictionary if type(k) is tuple}
    # Select the parameters which were originally nested and unflatten them
    nested_dict = {nk: {k[1]: v for k, v in dictionary.items() if k[0] == nk} for nk in nested_keys}
    # Get original dictionary without the nested keys
    dictionary_without_nested_keys = {k: v for k, v in dictionary.items() if type(k) is not tuple}
    # Return merged dicts
    return {**dictionary_without_nested_keys, **nested_dict}


def _get_param_hash(param):
    """
    Get the hash of a param dictionary.
    It first unflattens nested dicts, removes unused nested parameters, nests them again and then create a frozenset
    based on the resulting items (tuples).
    Note that the flattening and unflattening dict functions are idempotent.

    Parameters
    ----------
    param: dict
        Parameter configuration.
        Example::
            param_grid = {"k": 50, "eta": 2, "optimizer_params": {"lr": 0.1}}

    Returns
    -------
    str
        Hash of the param dictionary.
    """
    # Remove parameters that are not used by particular configurations
    # For example, if the regularization is None, there is no need for the regularization lambda
    flattened_params = _flatten_nested_keys(_remove_unused_params(_unflatten_nested_keys(param)))
    return hash(frozenset(flattened_params.items()))


class ParamHistory(object):
    """
    Used to evaluates whether a particular parameter configuration has already been previously seen or not.
    To achieve that, we hash each parameter configuration, removing unused parameters first.
    """
    def __init__(self):
        """The param history is a set of hashes."""
        self.param_hash_history = set()

    def add(self, param):
        """Add hash of parameter configuration to history."""
        self.param_hash_history.add(_get_param_hash(param))

    def __contains__(self, other):
        """Verify whether hash of parameter configuration is present in history."""
        return _get_param_hash(other) in self.param_hash_history


def _next_hyperparam(param_grid):
    """
    Iterator that gets the next parameter combination from a dictionary containing lists of parameters.
    The parameter combinations are deterministic and go over all possible combinations present in the parameter grid.

    Parameters
    ----------
    param_grid: dict
        Parameter configurations.
        Example::
            param_grid = {"k": [50, 100], "eta": [1, 2, 3]}

    Returns
    -------
    params: iterator
        One particular combination of parameters.

    """
    param_history = ParamHistory()

    # Flatten nested dictionaries so we can apply itertools.product to get all possible parameter combinations
    flattened_param_grid = _flatten_nested_keys(param_grid)

    for values in product(*flattened_param_grid.values()):
        # Get one single parameter combination as a flattened dictionary
        param = dict(zip(flattened_param_grid.keys(), values))

        # Only yield unique parameter combinations
        if param in param_history:
            continue
        else:
            param_history.add(param)
            # Yields nested configuration (unflattened) without useless parameters
            yield _remove_unused_params(_unflatten_nested_keys(param))


def _sample_parameters(param_grid):
    """
    Given a param_grid with callables and lists, execute callables and sample lists to return of random combination
    of parameters.

    Parameters
    ----------

    param_grid: dict
        Parameter configurations.
        Example::
            param_grid = {"k": [50, 100], "eta": lambda: np.random.choice([1, 2, 3])}

    Returns
    -------

    param: dict
        Return dictionary containing sampled parameters.

    """
    param = {}
    for k, v in param_grid.items():
        if callable(v):
            param[k] = v()
        elif type(v) is dict:
            param[k] = _sample_parameters(v)
        elif isinstance(v, Iterable) and type(v) is not str:
            param[k] = np.random.choice(v)
        else:
            param[k] = v
    return param


def _next_hyperparam_random(param_grid):
    """
    Iterator that gets the next parameter combination from a dictionary containing lists of parameters or callables.
    The parameter combinations are randomly chosen each iteration.

    Parameters
    ----------
    param_grid: dict
        Parameter configurations.
        Example::
            param_grid = {"k": [50, 100], "eta": [1, 2, 3]}

    Returns
    -------
    params: iterator
        One particular combination of parameters.

    """
    param_history = ParamHistory()

    while True:
        param = _sample_parameters(param_grid)

        # Only yield unique parameter combinations
        if param in param_history:
            continue
        else:
            param_history.add(param)
            yield _remove_unused_params(param)


def _scalars_into_lists(param_grid):
    """
    For a param_grid with scalars (instead of lists or callables), transform scalars into lists of size one.

    Parameters
    ----------
    param_grid: dict
        Parameter configurations.
        Example::
            param_grid = {"k": [50, 100], "eta": lambda: np.random.choice([1, 2, 3]}
    """
    for k, v in param_grid.items():
        if not (callable(v) or isinstance(v, Iterable)) or type(v) is str:
            param_grid[k] = [v]
        elif type(v) is dict:
            _scalars_into_lists(v)


def select_best_model_ranking(model_class, X_train, X_valid, X_test, param_grid, max_combinations=None,
                              param_grid_random_seed=0, use_filter=True, early_stopping=False,
                              early_stopping_params=None, use_test_for_selection=False, entities_subset=None,
                              corrupt_side='s,o', use_default_protocol=False, retrain_best_model=False, verbose=False):
    """Model selection routine for embedding models via either grid search or random search.

    For grid search, pass a fixed ``param_grid`` and leave ``max_combinations`` as `None`
    so that all combinations will be explored.

    For random search, delimit ``max_combinations`` to your computational budget
    and optionally set some parameters to be callables instead of a list (see the documentation for ``param_grid``).

    .. note::
        Random search is more efficient than grid search as the number of parameters grows :cite:`bergstra2012random`.
        It is also a strong baseline against more advanced methods such as
        Bayesian optimization :cite:`li2018hyperband`.

    The function also retrains the best performing model on the concatenation of training and validation sets.

    Note we generate negatives at runtime according to the strategy described in :cite:`bordes2013translating`.

    .. note::
        By default, model selection is done with raw MRR for better runtime performance (``use_filter=False``).

    Parameters
    ----------
    model_class : class
        The class of the EmbeddingModel to evaluate (TransE, DistMult, ComplEx, etc).
    X_train : ndarray, shape [n, 3]
        An array of training triples.
    X_valid : ndarray, shape [n, 3]
        An array of validation triples.
    X_test : ndarray, shape [n, 3]
        An array of test triples.
    param_grid : dict
        A grid of hyperparameters to use in model selection. The routine will train a model for each combination
        of these hyperparameters.

        Parameters can be either callables or lists.
        If callable, it must take no parameters and return a constant value.
        If any parameter is a callable, ``max_combinations`` must be set to some value.

        For example, the learning rate could either be ``"lr": [0.1, 0.01]``
        or ``"lr": lambda: np.random.uniform(0.01, 0.1)``.
    max_combinations: int
        Maximum number of combinations to explore.
        By default (None) all combinations will be explored,
        which makes it incompatible with random parameters for random search.
    param_grid_random_seed: int
        Random seed for the parameters that are callables and random.
    use_filter : bool
        If True, will use the entire input dataset X to compute filtered MRR (default: True).
    early_stopping: bool
        Flag to enable early stopping (default:False).

        If set to ``True``, the training loop adopts the following early stopping heuristic:

        - The model will be trained regardless of early stopping for ``burn_in`` epochs.
        - Every ``check_interval`` epochs the method will compute the metric specified in ``criteria``.

        If such metric decreases for ``stop_interval`` checks, we stop training early.

        Note the metric is computed on ``x_valid``. This is usually a validation set that you held out.

        Also, because ``criteria`` is a ranking metric, it requires generating negatives.
        Entities used to generate corruptions can be specified, as long as the side(s) of a triple to corrupt.
        The method supports filtered metrics, by passing an array of positives to ``x_filter``. This will be used to
        filter the negatives generated on the fly (i.e. the corruptions).

        .. note::

            Keep in mind the early stopping criteria may introduce a certain overhead
            (caused by the metric computation).
            The goal is to strike a good trade-off between such overhead and saving training epochs.

            A common approach is to use MRR unfiltered: ::

                early_stopping_params={x_valid=X['valid'], 'criteria': 'mrr'}

            Note the size of validation set also contributes to such overhead.
            In most cases a smaller validation set would be enough.

    early_stopping_params: dict
        Dictionary of parameters for early stopping.

        The following keys are supported:

            * x_valid: ndarray, shape [n, 3] : Validation set to be used for early stopping. Uses X['valid'] by default.

            * criteria: criteria for early stopping ``hits10``, ``hits3``, ``hits1`` or ``mrr``. (default)

            * x_filter: ndarray, shape [n, 3] : Filter to be used(no filter by default)

            * burn_in: Number of epochs to pass before kicking in early stopping(default: 100)

            * check_interval: Early stopping interval after burn-in(default:10)

            * stop_interval: Stop if criteria is performing worse over n consecutive checks (default: 3)

    use_test_for_selection:bool
        Use test set for model selection. If False, uses validation set (default: False).
    entities_subset: array-like
        List of entities to use for corruptions. If None, will generate corruptions
        using all distinct entities (default: None).
    corrupt_side: string
        Specifies which side to corrupt the entities:
        ``s`` is to corrupt only subject.
        ``o`` is to corrupt only object.
        ``s+o`` is to corrupt both subject and object.
        ``s,o`` is to corrupt both subject and object but ranks are computed separately (default).
    use_default_protocol: bool
        Flag to indicate whether to evaluate head and tail corruptions separately(default:False).
        If this is set to true, it will ignore corrupt_side argument and corrupt both head
        and tail separately and rank triples i.e. corrupt_side='s,o' mode.
    retrain_best_model: bool
        Flag to indicate whether best model should be re-trained at the end with the validation set used in the search.
        Default: False.
    verbose : bool
        Verbose mode for the model selection procedure (which is independent of the verbose mode in the model fit).

        Verbose mode includes display of the progress bar, logging info for each iteration,
        evaluation information, and exception details.

        If you need verbosity inside the model training itself, change the verbose parameter within the ``param_grid``.

    Returns
    -------
    best_model : EmbeddingModel
        The best trained embedding model obtained in model selection.

    best_params : dict
        The hyperparameters of the best embedding model `best_model`.

    best_mrr_train : float
        The MRR (unfiltered) of the best model computed over the validation set in the model selection loop.

    ranks_test : ndarray, shape [n] or [n,2] depending on the value of corrupt_side.
        An array of ranks of test triples.
        When ``corrupt_side='s,o'`` the function returns [n,2]. The first column represents the rank against
        subject corruptions and the second column represents the rank against object corruptions.
        In other cases, it returns [n] i.e. rank against the specified corruptions.

    mrr_test : float
        The MRR (filtered) of the best model, retrained on the concatenation of training and validation sets,
        computed over the test set.

    experimental_history: list of dict
        A list containing all the intermediate experimental results:
        the model parameters and the corresponding validation metrics.

    Examples
    --------
    >>> from ampligraph.datasets import load_wn18
    >>> from ampligraph.latent_features import ComplEx
    >>> from ampligraph.evaluation import select_best_model_ranking
    >>> import numpy as np
    >>>
    >>> X = load_wn18()
    >>>
    >>> model_class = ComplEx
    >>> param_grid = {
    >>>     "batches_count": [50],
    >>>     "seed": 0,
    >>>     "epochs": [100],
    >>>     "k": [100, 200],
    >>>     "eta": [5, 10, 15],
    >>>     "loss": ["pairwise", "nll"],
    >>>     "loss_params": {
    >>>         "margin": [2]
    >>>     },
    >>>     "embedding_model_params": {
    >>>     },
    >>>     "regularizer": ["LP", None],
    >>>     "regularizer_params": {
    >>>         "p": [1, 3],
    >>>         "lambda": [1e-4, 1e-5]
    >>>     },
    >>>     "optimizer": ["adagrad", "adam"],
    >>>     "optimizer_params": {
    >>>         "lr": lambda: np.random.uniform(0.0001, 0.01)
    >>>     },
    >>>     "verbose": False
    >>> }
    >>> select_best_model_ranking(model_class, X['train'], X['valid'], X['test'],
    >>>                           param_grid,
    >>>                           max_combinations=100,
    >>>                           use_filter=True,
    >>>                           verbose=True,
    >>>                           early_stopping=True)

    """
    logger.debug('Starting gridsearch over hyperparameters. {}'.format(param_grid))
    if use_default_protocol:
        logger.warning('DeprecationWarning: use_default_protocol will be removed in future. \
                        Please use corrupt_side argument instead.')
        corrupt_side = 's,o'

    if early_stopping_params is None:
        early_stopping_params = {}

    # Verify missing parameters for the model class (default values will be used)
    undeclared_args = set(model_class.__init__.__code__.co_varnames[1:]) - set(param_grid.keys())
    if len(undeclared_args) != 0:
        logger.debug("The following arguments were not defined in the parameter grid"
                     " and thus the default values will be used: {}".format(', '.join(undeclared_args)))

    param_grid["model_name"] = model_class.name
    _scalars_into_lists(param_grid)

    if max_combinations is not None:
        np.random.seed(param_grid_random_seed)
        model_params_combinations = islice(_next_hyperparam_random(param_grid), max_combinations)
    else:
        model_params_combinations = _next_hyperparam(param_grid)

    best_mrr_train = 0
    best_model = None
    best_params = None

    if early_stopping:
        try:
            early_stopping_params['x_valid']
        except KeyError:
            logger.debug('Early stopping enable but no x_valid parameter set. Setting x_valid to {}'.format(X_valid))
            early_stopping_params['x_valid'] = X_valid

    if use_filter:
        X_filter = np.concatenate((X_train, X_valid, X_test))
    else:
        X_filter = None

    if use_test_for_selection:
        selection_dataset = X_test
    else:
        selection_dataset = X_valid

    experimental_history = []

    def evaluation(ranks):
        mrr = mrr_score(ranks)
        mr = mr_score(ranks)
        hits_1 = hits_at_n_score(ranks, n=1)
        hits_3 = hits_at_n_score(ranks, n=3)
        hits_10 = hits_at_n_score(ranks, n=10)
        return mrr, mr, hits_1, hits_3, hits_10

    for model_params in tqdm(model_params_combinations, total=max_combinations, disable=(not verbose)):
        current_result = {
            "model_name": model_params["model_name"],
            "model_params": model_params
        }
        del model_params["model_name"]
        try:
            model = model_class(**model_params)
            model.fit(X_train, early_stopping, early_stopping_params)
            ranks = evaluate_performance(selection_dataset, model=model,
                                         filter_triples=X_filter, verbose=verbose,
                                         entities_subset=entities_subset,
                                         use_default_protocol=use_default_protocol,
                                         corrupt_side=corrupt_side)

            curr_mrr, mr, hits_1, hits_3, hits_10 = evaluation(ranks)

            current_result["results"] = {
                "mrr": curr_mrr,
                "mr": mr,
                "hits_1": hits_1,
                "hits_3": hits_3,
                "hits_10": hits_10
            }

            info = 'mr: {} mrr: {} hits 1: {} hits 3: {} hits 10: {}, model: {}, params: {}'.format(
                mr, curr_mrr, hits_1, hits_3, hits_10, type(model).__name__, model_params
            )

            logger.debug(info)
            if verbose:
                logger.info(info)

            if curr_mrr > best_mrr_train:
                best_mrr_train = curr_mrr
                best_model = model
                best_params = model_params
        except Exception as e:
            current_result["results"] = {
                "exception": str(e)
            }

            if verbose:
                logger.error('Exception occurred for parameters:{}'.format(model_params))
                logger.error(str(e))
            else:
                pass
        experimental_history.append(current_result)

    if best_model is not None:
        if retrain_best_model:
            best_model.fit(np.concatenate((X_train, X_valid)), early_stopping, early_stopping_params)

        ranks_test = evaluate_performance(X_test, model=best_model,
                                          filter_triples=X_filter, verbose=verbose,
                                          entities_subset=entities_subset,
                                          use_default_protocol=use_default_protocol,
                                          corrupt_side=corrupt_side)

        test_mrr, test_mr, test_hits_1, test_hits_3, test_hits_10 = evaluation(ranks_test)

        info = \
            'Best model test results: mr: {} mrr: {} hits 1: {} hits 3: {} hits 10: {}, model: {}, params: {}'.format(
                test_mrr, test_mr, test_hits_1, test_hits_3, test_hits_10, type(best_model).__name__, best_params
            )

        logger.debug(info)
        if verbose:
            logger.info(info)

        test_evaluation = {
            "mrr": test_mrr,
            "mr": test_mr,
            "hits_1": test_hits_1,
            "hits_3": test_hits_3,
            "hits_10": test_hits_10
        }
    else:
        ranks_test = []

        test_evaluation = {
            "mrr": np.nan,
            "mr": np.nan,
            "hits_1": np.nan,
            "hits_3": np.nan,
            "hits_10": np.nan
        }

    return best_model, best_params, best_mrr_train, ranks_test, test_evaluation, experimental_history
