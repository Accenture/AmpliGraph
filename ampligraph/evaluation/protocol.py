# Copyright 2019 The AmpliGraph Authors. All Rights Reserved.
#
# This file is Licensed under the Apache License, Version 2.0.
# A copy of the Licence is available in LICENCE, or at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
import numpy as np
from tqdm import tqdm

from ..evaluation import mrr_score, hits_at_n_score, mr_score
import itertools
import tensorflow as tf
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def train_test_split_no_unseen(X, test_size=100, seed=0, allow_duplication=False):
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

    Returns
    -------
    X_train : ndarray, size[n, 3]
        The training set
    X_test : ndarray, size[n, 3]
        The test set

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
    >>> X_train_valid, X_test = train_test_split_no_unseen(X, test_size=2)
    >>> X_train, X_valid = train_test_split_no_unseen(X_train_valid, test_size=2)
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
    while idx_test.shape[0] < test_size:
        i = rnd.randint(len(X))
        if dict_subs[X[i, 0]] > 1 and dict_objs[X[i, 2]] > 1 and dict_rels[X[i,1]] > 1:
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
                                "Change seed values, or set test_size to a smaller value.")
            else:
                raise Exception("Cannot create a test split of the desired size. "
                                "Some entities will not occur in both training and test set. "
                                "Set allow_duplication=True, or "
                                "change seed values, or set test_size to a smaller value.")

    logger.debug('Completed random search.')
    
    idx = np.arange(len(X))
    idx_train = np.setdiff1d(idx, idx_test)
    logger.debug('Train test split completed.')

    return X[idx_train, :], X[idx_test, :]


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
        The relation-to-internal-id associations
    ent_to_idx: dict
        The entity-to-internal-id associations.

    """
    logger.debug('Creating mappings for entities and relations.')
    unique_ent = np.unique(np.concatenate((X[:, 0], X[:, 2])))
    unique_rel = np.unique(X[:, 1])
    return _create_unique_mappings(unique_ent, unique_rel)


def generate_corruptions_for_eval(X, entities_for_corruption, corrupt_side='s+o', table_entity_lookup_left=None,
                                  table_entity_lookup_right=None, table_reln_lookup=None):
    """Generate corruptions for evaluation.

        Create corruptions (subject and object) for a given triple x, in compliance with the
        local closed world assumption (LCWA), as described in :cite:`nickel2016review`.
        
        .. note::
            For filtering the corruptions, we adopt a hashing-based strategy to handle the set difference problem.
            This strategy is as described below:

            * We compute unique entities and relations in our dataset.

            * We assign unique prime numbers for entities (unique for subject and object separately) and for relations
              and create three separate hash tables. (these hash maps are input to this function)

            * For each triple in filter_triples, we get the prime numbers associated with subject, relation
              and object by mapping to their respective hash tables. We then compute the **prime product for the
              filter triple**. We store this triple product.

            * Since the numbers assigned to subjects, relations and objects are unique, their prime product is also
              unique. i.e. a triple :math:`(a, b, c)` would have a different product compared to triple :math:`(c, b, a)`
              as :math:`a, c` of subject have different primes compared to :math:`a, c` of object.

            * While generating corruptions for evaluation, we hash the triple's entities and relations and get
              the associated prime number and compute the **prime product for the corrupted triple**.

            * If this product is present in the products stored for the filter set, then we remove the corresponding
              corrupted triple (as it is a duplicate i.e. the corruption triple is present in filter_triples)

            * Using this approach we generate filtered corruptions for evaluation.

            **Execution Time:** This method takes ~20 minutes on FB15K using ComplEx
            (Intel Xeon Gold 6142, 64 GB Ubuntu 16.04 box, Tesla V100 16GB)

    Parameters
    ----------
    X : Tensor, shape [1, 3]
        Currently, a single positive triples that will be used to create corruptions.
    entities_for_corruption : Tensor
        All the entity IDs which are to be used for generation of corruptions
    corrupt_side: string
        Specifies which side of the triple to corrupt:

        - 's': corrupt only subject.
        - 'o': corrupt only object
        - 's+o': corrupt both subject and object
    table_entity_lookup_left : tf.HashTable
        Hash table of subject entities mapped to unique prime numbers
    table_entity_lookup_right : tf.HashTable
        Hash table of object entities mapped to unique prime numbers
    table_reln_lookup : tf.HashTable
        Hash table of relations mapped to unique prime numbers

    Returns
    -------

    out : Tensor, shape [n, 3]
        An array of corruptions for the triples for x.
        
    out_prime : Tensor, shape [n, 3]
        An array of product of prime numbers associated with corruption triples or None 
        based on filtered or non filtered version.

    """

    logger.debug('Generating corruptions for evaluation.')

    logger.debug('Getting repeating subjects.')
    if corrupt_side not in ['s+o', 's', 'o']:
        msg = 'Invalid argument value for corruption side passed for evaluation'
        logger.error(msg)
        raise ValueError(msg)

    if corrupt_side in ['s+o', 'o']:  # object is corrupted - so we need subjects as it is
        repeated_subjs = tf.keras.backend.repeat(
            tf.slice(X,
                     [0, 0],  # subj
                     [tf.shape(X)[0], 1])
            , tf.shape(entities_for_corruption)[0])
        repeated_subjs = tf.squeeze(repeated_subjs, 2)

    logger.debug('Getting repeating object.')
    if corrupt_side in ['s+o', 's']:  # subject is corrupted - so we need objects as it is
        repeated_objs = tf.keras.backend.repeat(
            tf.slice(X,
                     [0, 2],  # Obj
                     [tf.shape(X)[0], 1])
            , tf.shape(entities_for_corruption)[0])
        repeated_objs = tf.squeeze(repeated_objs, 2)

    logger.debug('Getting repeating relationships.')
    repeated_relns = tf.keras.backend.repeat(
        tf.slice(X,
                 [0, 1],  # reln
                 [tf.shape(X)[0], 1])
        , tf.shape(entities_for_corruption)[0])
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
    out_prime = tf.constant([])

    logger.debug('Creating prime numbers associated with corruptions.')
    if table_entity_lookup_left != None and table_entity_lookup_right != None and table_reln_lookup != None:

        if corrupt_side in ['s+o', 'o']:
            prime_subj = tf.squeeze(table_entity_lookup_left.lookup(repeated_subjs))
            prime_ent_right = tf.squeeze(table_entity_lookup_right.lookup(rep_ent))

        if corrupt_side in ['s+o', 's']:
            prime_obj = tf.squeeze(table_entity_lookup_right.lookup(repeated_objs))
            prime_ent_left = tf.squeeze(table_entity_lookup_left.lookup(rep_ent))

        prime_reln = tf.squeeze(table_reln_lookup.lookup(repeated_relns))

        if corrupt_side == 's+o':
            out_prime = tf.concat([prime_subj * prime_reln * prime_ent_right,
                                   prime_ent_left * prime_reln * prime_obj], 0)

        elif corrupt_side == 'o':
            out_prime = prime_subj * prime_reln * prime_ent_right
        else:
            out_prime = prime_ent_left * prime_reln * prime_obj

    logger.debug('Returning corruptions for evaluation.')
    return out, out_prime


def generate_corruptions_for_fit(X, entities_list=None, eta=1, corrupt_side='s+o', entities_size=0, rnd=None):
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
            You can enable such behaviour be setting ``entities_size==-1``. In such case, if ``entities_list=None``
            all entities from the *current batch* will be used to generate corruptions.

    Parameters
    ----------
    X : Tensor, shape [n, 3]
        An array of positive triples that will be used to create corruptions.
    entities_list : list
        List of entities to be used for generating corruptions. (default:None).
        if ``entities_list=None``, all entities will be used to generate corruptions (default behaviour).
    eta : int
        The number of corruptions per triple that must be generated.
    corrupt_side: string
        Specifies which side of the triple to corrupt:

        - 's': corrupt only subject.
        - 'o': corrupt only object
        - 's+o': corrupt both subject and object
    entities_size: int
        Size of entities to be used while generating corruptions. It assumes entity id's start from 0 and are
        continuous. (default: 0).
        When processing large knowledge graphs, it may be useful to generate corruptions only using entities from
        a single batch.
        This also brings the benefit of creating more meaningful negatives, as entities used to corrupt are
        sourced locally.
        The function can be configured to generate corruptions *only* using the entities from the current batch.
        You can enable such behaviour be setting ``entities_size==-1``. In such case, if ``entities_list=None``
        all entities from the *current batch* will be used to generate corruptions.
    rnd: numpy.random.RandomState
        A random number generator.

    Returns
    -------

    out : Tensor, shape [n * eta, 3]
        An array of corruptions for a list of positive triples x. For each row in X the corresponding corruption
        indexes can be found at [index+i*n for i in range(eta)]

    """
    logger.debug('Generating corruptions for fit.')
    if corrupt_side not in ['s+o', 's', 'o']:
        msg = 'Invalid argument value {} for corruption side passed for evaluation.'.format(corrupt_side)
        logger.error(msg)
        raise ValueError(msg)

    dataset = tf.reshape(tf.tile(tf.reshape(X, [-1]), [eta]), [tf.shape(X)[0] * eta, 3])

    if corrupt_side == 's+o':
        keep_subj_mask = tf.tile(tf.cast(tf.random_uniform([tf.shape(X)[0]], 0, 2, dtype=tf.int32, seed=rnd), tf.bool),
                                 [eta])
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

        random_indices = tf.squeeze(tf.multinomial(tf.expand_dims(tf.zeros(tf.shape(entities_list)[0]), 0),
                                                   num_samples=tf.shape(dataset)[0],
                                                   seed=rnd))

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
    x_idx_s = np.vectorize(ent_to_idx.get)(X[:, 0])
    x_idx_p = np.vectorize(rel_to_idx.get)(X[:, 1])
    x_idx_o = np.vectorize(obj_to_idx.get)(X[:, 2])

    if None in x_idx_s or None in x_idx_o:
        msg = 'Input triples include one or more entities not present in the training set. ' \
              'Please filter X using evaluation.filter_unseen_entities(), or retrain the model on a training set ' \
              'that includes all the desired distinct entities.'
        logger.error(msg)
        raise ValueError(msg)

    if None in x_idx_p:
        msg = 'Input triples include one or more relation type not present in the training set. ' \
              'Please filter all relation in X that do not occur in the training test. ' \
              'or retrain the model on a training set that includes all the desired relation types.'
        logger.error(msg)
        raise ValueError(msg)

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


def evaluate_performance(X, model, filter_triples=None, verbose=False, strict=True, rank_against_ent=None,
                         corrupt_side='s+o', use_default_protocol=True):
    """Evaluate the performance of an embedding model.

        Run the relational learning evaluation protocol defined in :cite:`bordes2013translating`.

        It computes the rank of each positive triple against a number of negatives generated on the fly.
        Such negatives are compliant with the local closed world assumption (LCWA),
        as described in :cite:`nickel2016review`. In practice, that means only one side of the triple is corrupted
        (i.e. either the subject or the object).

        .. note::
            When *filtered* mode is enabled (i.e. `filtered_triples` is not ``None``),
            to speed up the procedure, we adopt a hashing-based strategy to handle the set difference problem.
            This strategy is as described below:

            * We compute unique entities and relations in our dataset.

            * We assign unique prime numbers for entities (unique for subject and object separately) and for relations
              and create three separate hash tables.

            * For each triple in ``filter_triples``, we get the prime numbers associated with subject, relation
              and object by mapping to their respective hash tables. We then compute the **prime product for the
              filter triple**. We store this triple product.

            * Since the numbers assigned to subjects, relations and objects are unique, their prime product is also
              unique. i.e. a triple :math:`(a, b, c)` would have a different product compared to triple :math:`(c, b, a)`
              as :math:`a, c` of subject have different primes compared to :math:`a, c` of object.

            * While generating corruptions for evaluation, we hash the triple's entities and relations and get
              the associated prime number and compute the **prime product for the corrupted triple**.

            * If this product is present in the products stored for the filter set, then we remove the corresponding
              corrupted triple (as it is a duplicate i.e. the corruption triple is present in ``filter_triples``)

            * Using this approach we generate filtered corruptions for evaluation.

            **Execution Time:** This method takes ~20 minutes on FB15K using ComplEx
            (Intel Xeon Gold 6142, 64 GB Ubuntu 16.04 box, Tesla V100 16GB)

        .. hint::
            When ``rank_against_ent=None``, the method will use all distinct entities in the knowledge graph ``X``
            to generate negatives to rank against. If ``X`` includes more than 2.5 million unique
            entities and relations, the method will return a runtime error.
            To solve the problem, it is recommended to pass the desired entities to use to generate corruptions
            to ``rank_against_ent``. Besides, trying to rank a positive against an extremely large number of negatives
            may be overkilling. As a reference, the popular FB15k-237 dataset has ~15k distinct entities. The evaluation
            protocol ranks each positives against 15k corruptions per side.

    Parameters
    ----------
    X : ndarray, shape [n, 3]
        An array of test triples.
    model : EmbeddingModel
        A knowledge graph embedding model
    filter_triples : ndarray of shape [n, 3] or None
        The triples used to filter negatives.
    verbose : bool
        Verbose mode
    strict : bool
        Strict mode. If True then any unseen entity will cause a RuntimeError.
        If False then triples containing unseen entities will be filtered out.
    rank_against_ent: array-like
        List of entities to use for corruptions. If None, will generate corruptions
        using all distinct entities. Default is None.
    corrupt_side: string
        Specifies which side of the triple to corrupt:

        - 's': corrupt only subject.
        - 'o': corrupt only object
        - 's+o': corrupt both subject and object. The same behaviour is obtained with ``use_default_protocol=True``.

        .. note::
            If ``corrupt_side='s+o'`` the function will return 2*n ranks.
            If ``corrupt_side='s'`` or ``corrupt_side='o'``, it will return n ranks, where n is the
            number of statements in X.
            The first n elements of ranks are obtained against subject corruptions. From n+1 until 2n ranks are obtained
            against object corruptions.

    use_default_protocol: bool
        Flag to indicate whether to use the standard protocol used in literature defined in
        :cite:`bordes2013translating` (default: True).
        If set to ``True`` it is equivalent to ``corrupt_side='s+o'``.
        This corresponds to the evaluation protcol used in literature, where head and tail corruptions
        are evaluated separately.

        .. note::
            When ``use_default_protocol=True`` the function will return 2*n ranks.
            The first n elements of ranks are obtained against subject corruptions. From n+1 until 2n ranks are obtained
            against object corruptions.
    Returns
    -------
    ranks : ndarray, shape [n] or [2*n]
        An array of ranks of positive test triples.
        When ``use_default_protocol=True`` or ``corrupt_side='s+o'``, the function returns 2*n ranks instead of n.
        In that case the first n elements of ranks are obtained against subject corruptions. From n+1 until 2n ranks
        are obtained against object corruptions.

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
    >>> filter = np.concatenate((X['train'], X['valid'], X['test']))
    >>> ranks = evaluate_performance(X['test'][:5], model=model,
                                     filter_triples=filter,
                                     corrupt_side='s+o',
                                     use_default_protocol=False)
    >>> ranks
    [1, 582, 543, 6, 31]
    >>> mrr_score(ranks)
    0.24049691297347323
    >>> hits_at_n_score(ranks, n=10)
    0.4
    """

    logger.debug('Evaluating the performance of the embedding model.')
    X_test = filter_unseen_entities(X, model, verbose=verbose, strict=strict)

    X_test = to_idx(X_test, ent_to_idx=model.ent_to_idx, rel_to_idx=model.rel_to_idx)

    if filter_triples is not None:
        logger.debug('Getting filtered triples.')
        filter_triples = to_idx(filter_triples, ent_to_idx=model.ent_to_idx, rel_to_idx=model.rel_to_idx)
        
    eval_dict = {}
    eval_dict['default_protocol'] = False
    
    if use_default_protocol:
        corrupt_side = 's+o'
        eval_dict['default_protocol'] = True

    if rank_against_ent is not None:
        idx_entities = np.asarray([idx for uri, idx in model.ent_to_idx.items() if uri in rank_against_ent])
        eval_dict['corruption_entities'] = idx_entities

    ranks = []

    logger.debug('Evaluating the test set by corrupting side : {}'.format(corrupt_side))
    eval_dict['corrupt_side'] = corrupt_side
    if filter_triples is not None:
        model.set_filter_for_eval(filter_triples)
    logger.debug('Configuring evaluation protocol.')
    model.configure_evaluation_protocol(eval_dict)
    logger.debug('Making predictions.')
    for i in tqdm(range(X_test.shape[0]), disable=(not verbose)):
        _, rank = model.predict(X_test[i], from_idx=True, get_ranks=True)
        if use_default_protocol :
            ranks.extend(list(rank)) 
            continue
            
        ranks.append(rank)
            
    model.end_evaluation()
    logger.debug('Ending Evaluation')

    logger.debug('Returning ranks of positive test triples obtained by corrupting {}.'.format(corrupt_side))
    return ranks


def filter_unseen_entities(X, model, verbose=False, strict=True):
    """Filter unseen entities in the test set.

    Parameters
    ----------
    X : ndarray, shape [n, 3]
        An array of test triples.
    model : ampligraph.latent_features.EmbeddingModel
        A knowledge graph embedding model
    verbose : bool
        Verbose mode
    strict : bool
        Strict mode. If True then any unseen entity will cause a RuntimeError.
        If False then triples containing unseen entities will be filtered out.

    Returns
    -------
    filtered X : ndarray, shape [n, 3]
        An array of test triples containing no unseen entities.
    """

    logger.debug('Finding entities in test set that are not previously seen by model')
    ent_seen = np.unique(list(model.ent_to_idx.keys()))
    ent_test = np.unique(X[:, [0, 2]].ravel())
    ent_unseen = np.setdiff1d(ent_test, ent_seen, assume_unique=True)

    if ent_unseen.size == 0:
        logger.debug('No unseen entities found.')
        return X
    else:
        logger.debug('Unseen entities found.')
        if strict:
            msg = 'Unseen entities found in test set, please remove or run evaluate_performance() with strict=False.'
            logger.error(msg)
            raise RuntimeError(msg)
        else:
            # Get row-wise mask of triples containing unseen entities
            mask_unseen = np.isin(X, ent_unseen).any(axis=1)

            msg = 'Removing {} triples containing unseen entities. '.format(np.sum(mask_unseen))
            if verbose:
                logger.debug(msg)
            logger.debug(msg)
            return X[~mask_unseen]


def yield_all_permutations(registry, category_type, category_type_params):
    """Yields all the permutation of category type with their respective hyperparams

    Parameters
    ----------
    registry: dictionary
        registry of the category type
    category_type: string
        category type values
    category_type_params: list
        category type hyperparams

    Returns
    -------
    name: str
        Specific name of the category
    present_params: list
        Names of hyperparameters of the category
    val: list
        Values of the respective hyperparams
    """
    for name in category_type:
        present_params = []
        present_params_vals = []
        if name is not None:
            for param in registry[name].external_params:
                try:
                    present_params_vals.append(category_type_params[param])
                    present_params.append(param)
                except KeyError as e:
                    logger.debug('Key not found {}'.format(e))
                    pass
        for val in itertools.product(*present_params_vals):
            yield name, present_params, val


def gridsearch_next_hyperparam(model_name, in_dict):
    """Performs grid search on hyperparams

    Parameters
    ----------
    model_name: string
        name of the embedding model
    in_dict: dictionary
        dictionary of all the parameters and the list of values to be searched

    Returns:
    out_dict: dict
        Dictionary containing an instance of model hyperparameters.
    """

    from ..latent_features import LOSS_REGISTRY, REGULARIZER_REGISTRY, MODEL_REGISTRY
    logger.debug('Starting gridsearch over hyperparameters. {}'.format(in_dict))
    try:
        verbose = in_dict["verbose"]
    except KeyError:
        logger.debug('Verbose key not found. Setting to False.')
        verbose = False

    try:
        seed = in_dict["seed"]
    except KeyError:
        logger.debug('Seed key not found. Setting to -1.')
        seed = -1

    try:
        for batch_count in in_dict["batches_count"]:
            for epochs in in_dict["epochs"]:
                for k in in_dict["k"]:
                    for eta in in_dict["eta"]:
                        for reg_type, reg_params, reg_param_values in \
                                yield_all_permutations(REGULARIZER_REGISTRY, in_dict["regularizer"],
                                                       in_dict["regularizer_params"]):
                            for optimizer_type in in_dict["optimizer"]:
                                for optimizer_lr in in_dict["optimizer_params"]["lr"]:
                                    for loss_type, loss_params, loss_param_values in \
                                            yield_all_permutations(LOSS_REGISTRY, in_dict["loss"],
                                                                   in_dict["loss_params"]):
                                        for model_type, model_params, model_param_values in \
                                                yield_all_permutations(MODEL_REGISTRY, [model_name],
                                                                       in_dict["embedding_model_params"]):
                                            out_dict = {
                                                "batches_count": batch_count,
                                                "epochs": epochs,
                                                "k": k,
                                                "eta": eta,
                                                "loss": loss_type,
                                                "loss_params": {},
                                                "embedding_model_params": {},
                                                "regularizer": reg_type,
                                                "regularizer_params": {},
                                                "optimizer": optimizer_type,
                                                "optimizer_params": {
                                                    "lr": optimizer_lr
                                                },
                                                "verbose": verbose
                                            }

                                            if seed >= 0:
                                                out_dict["seed"] = seed
                                            # TODO - Revise this, use dict comprehension instead of for loops
                                            for idx in range(len(loss_params)):
                                                out_dict["loss_params"][loss_params[idx]] = loss_param_values[idx]
                                            for idx in range(len(reg_params)):
                                                out_dict["regularizer_params"][reg_params[idx]] = reg_param_values[idx]
                                            for idx in range(len(model_params)):
                                                out_dict["embedding_model_params"][model_params[idx]] = \
                                                    model_param_values[idx]

                                            yield (out_dict)
    except KeyError as e:
        logger.debug('Hyperparameters are missing from the input dictionary: {}'.format(e))
        print('One or more of the hyperparameters was not passed:')
        print(str(e))


def select_best_model_ranking(model_class, X, param_grid, use_filter=False, early_stopping=False,
                              early_stopping_params={}, use_test_for_selection=True, rank_against_ent=None,
                              corrupt_side='s+o', use_default_protocol=False, verbose=False):
    """Model selection routine for embedding models.

        .. note::
            By default, model selection is done with raw MRR for better runtime performance (``use_filter=False``).

        The function also retrains the best performing model on the concatenation of training and validation sets.

        Note we generate negatives at runtime according to the strategy described in ::cite:`bordes2013translating`).

    Parameters
    ----------
    model_class : class
        The class of the EmbeddingModel to evaluate (TransE, DistMult, ComplEx, etc).
    X : dict
        A dictionary of triples to use in model selection. Must include three keys: `train`, `val`, `test`.
        Values are ndarray of shape [n, 3]..
    param_grid : dict
        A grid of hyperparameters to use in model selection. The routine will train a model for each combination
        of these hyperparameters.
    use_filter : bool
        If True, will use the entire input dataset X to compute filtered MRR
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

            x_valid: ndarray, shape [n, 3] : Validation set to be used for early stopping. Uses X['valid'] by default.

            criteria: criteria for early stopping ``hits10``, ``hits3``, ``hits1`` or ``mrr``. (default)

            x_filter: ndarray, shape [n, 3] : Filter to be used(no filter by default)

            burn_in: Number of epochs to pass before kicking in early stopping(default: 100)

            check_interval: Early stopping interval after burn-in(default:10)

            stop_interval: Stop if criteria is performing worse over n consecutive checks (default: 3)

    use_test_for_selection:bool
        Use test set for model selection. If False, uses validation set. Default(True)
    rank_against_ent: array-like
        List of entities to use for corruptions. If None, will generate corruptions
        using all distinct entities. Default is None.
    corrupt_side: string
        Specifies which side to corrupt the entities.
        ``s`` is to corrupt only subject.
        ``o`` is to corrupt only object
        ``s+o`` is to corrupt both subject and object
    use_default_protocol: bool
        Flag to indicate whether to evaluate head and tail corruptions separately(default:False).
        If this is set to true, it will ignore corrupt_side argument and corrupt both head and tail separately and rank triples.
    verbose : bool
        Verbose mode during evaluation of trained model

    Returns
    -------
    best_model : EmbeddingModel
        The best trained embedding model obtained in model selection.

    best_params : dict
        The hyperparameters of the best embedding model `best_model`.

    best_mrr_train : float
        The MRR (unfiltered) of the best model computed over the validation set in the model selection loop.

    ranks_test : ndarray, shape [n]
        The ranks of each triple in the test set X['test].

    mrr_test : float
        The MRR (filtered) of the best model, retrained on the concatenation of training and validation sets,
        computed over the test set.

    Examples
    --------
    >>> from ampligraph.datasets import load_wn18
    >>> from ampligraph.latent_features import ComplEx
    >>> from ampligraph.evaluation import select_best_model_ranking
    >>>
    >>> X = load_wn18()
    >>> model_class = ComplEx
    >>> param_grid = {
    >>>                     "batches_count": [50],
    >>>                     "seed": 0,
    >>>                     "epochs": [4000],
    >>>                     "k": [100, 200],
    >>>                     "eta": [5,10,15],
    >>>                     "loss": ["pairwise", "nll"],
    >>>                     "loss_params": {
    >>>                         "margin": [2]
    >>>                     },
    >>>                     "embedding_model_params": {
    >>> 
    >>>                     },
    >>>                     "regularizer": ["LP", None],
    >>>                     "regularizer_params": {
    >>>                         "p": [1, 3],
    >>>                         "lambda": [1e-4, 1e-5]
    >>>                     },
    >>>                     "optimizer": ["adagrad", "adam"],
    >>>                     "optimizer_params":{
    >>>                         "lr": [0.01, 0.001, 0.0001]
    >>>                     },
    >>>                     "verbose": false
    >>>                 }
    >>> select_best_model_ranking(model_class, X, param_grid, use_filter=True, verbose=True, early_stopping=True)

    """
    hyperparams_list_keys = ["batches_count", "epochs", "k", "eta", "loss", "regularizer", "optimizer"]
    hyperparams_dict_keys = ["loss_params", "embedding_model_params", "regularizer_params", "optimizer_params"]

    for key in hyperparams_list_keys:
        if key not in param_grid.keys() or param_grid[key] == []:
            logger.debug('Hyperparameter key {} is missing.'.format(key))
            raise ValueError('Please pass values for key {}'.format(key))

    for key in hyperparams_dict_keys:
        if key not in param_grid.keys():
            logger.debug('Hyperparameter key {} is missing, replacing with empty dictionary.'.format(key))
            param_grid[key] = {}

    # this would be extended later to take multiple params for optimizers(currently only lr supported)
    try:
        lr = param_grid["optimizer_params"]["lr"]
    except KeyError:
        logger.debug('Hypermater key {} is missing'.format(key))
        raise ValueError('Please pass values for optimizer parameter - lr')

    model_params_combinations = gridsearch_next_hyperparam(model_class.name, param_grid)

    best_mrr_train = 0
    best_model = None
    best_params = None

    if early_stopping:
        try:
            early_stopping_params['x_valid']
        except KeyError:
            logger.debug('Early stopping enable but no x_valid parameter set. Setting x_valid to {}'.format(X['valid']))
            early_stopping_params['x_valid'] = X['valid']

    if use_filter:
        X_filter = np.concatenate((X['train'], X['valid'], X['test']))
    else:
        X_filter = None

    if use_test_for_selection:
        selection_dataset = X['test']
    else:
        selection_dataset = X['valid']

    for model_params in tqdm(model_params_combinations, disable=(not verbose)):
        try:
            model = model_class(**model_params)
            model.fit(X['train'], early_stopping, early_stopping_params)

            ranks = evaluate_performance(selection_dataset, model=model,
                                         filter_triples=X_filter, verbose=verbose,
                                         rank_against_ent=rank_against_ent,
                                         use_default_protocol=use_default_protocol,
                                         corrupt_side=corrupt_side)

            curr_mrr = mrr_score(ranks)
            mr = mr_score(ranks)
            hits_1 = hits_at_n_score(ranks, n=1)
            hits_3 = hits_at_n_score(ranks, n=3)
            hits_10 = hits_at_n_score(ranks, n=10)
            info = 'mr:{} mrr: {} hits 1: {} hits 3: {} hits 10: {}, model: {}, params: {}'.format(mr, curr_mrr, hits_1,
                                                                                                   hits_3, hits_10,
                                                                                                   type(model).__name__,
                                                                                                   model_params)
            logger.debug(info)
            if verbose:
                logger.info(info)

            if curr_mrr > best_mrr_train:
                best_mrr_train = curr_mrr
                best_model = model
                best_params = model_params
        except Exception as e:
            if verbose:
                logger.error('Exception occured for parameters:{}'.format(model_params))
                logger.error(str(e))
            else:
                pass
    
    ranks_test =[]
    mrr_test = 0
    if best_model is not None:
        # Retraining
        best_model.fit(np.concatenate((X['train'], X['valid'])))

        ranks_test = evaluate_performance(X['test'], model=best_model,
                                          filter_triples=X_filter, verbose=verbose,
                                          rank_against_ent=rank_against_ent,
                                          use_default_protocol=use_default_protocol,
                                          corrupt_side=corrupt_side)

        mrr_test = mrr_score(ranks_test)

    return best_model, best_params, best_mrr_train, ranks_test, mrr_test
