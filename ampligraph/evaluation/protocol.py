import numpy as np
from tqdm import tqdm
from ..evaluation import rank_score, mrr_score
import os
from joblib import Parallel, delayed
import itertools
import tensorflow as tf


def train_test_split_no_unseen(X, test_size=5000, seed=0):
    """Split into train and test sets.

     Test set contains only entities and relations which also occur
     in the training set.

    Parameters
    ----------
    X : ndarray, size[n, 3]
        The dataset to split.
    test_size : int, float
        If int, the number of triples in the test set. If float, the percentage of total triples.
    seed : int
        A random seed used to split the dataset.

    Returns
    -------
    X_train : ndarray, size[n, 3]
        The training set
    X_test : ndarray, size[n, 3]
        The test set

    """

    if type(test_size) is float:
        test_size = int(len(X) * test_size)

    rnd = np.random.RandomState(seed)

    subs, subs_cnt = np.unique(X[:, 0], return_counts=True)
    objs, objs_cnt = np.unique(X[:, 2], return_counts=True)
    rels, rels_cnt = np.unique(X[:, 1], return_counts=True)
    dict_subs = dict(zip(subs, subs_cnt))
    dict_objs = dict(zip(objs, objs_cnt))
    dict_rels = dict(zip(rels, rels_cnt))

    idx_test = []
    while len(idx_test) < test_size:
        i = rnd.randint(len(X))
        if dict_subs[X[i, 0]] > 1 and dict_objs[X[i, 2]] > 1 and dict_rels[X[i, 1]] > 1:
            dict_subs[X[i, 0]] -= 1
            dict_objs[X[i, 2]] -= 1
            dict_rels[X[i, 1]] -= 1
            idx_test.append(i)

    idx = np.arange(len(X))
    idx_train = np.setdiff1d(idx, idx_test)
    return X[idx_train, :], X[idx_test, :]


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
    unique_ent = np.unique(np.concatenate((X[:, 0], X[:, 2])))
    unique_rel = np.unique(X[:, 1])
    ent_count = len(unique_ent)
    rel_count = len(unique_rel)
    rel_to_idx = dict(zip(unique_rel, range(rel_count)))
    ent_to_idx = dict(zip(unique_ent, range(ent_count)))
    return rel_to_idx, ent_to_idx

def create_mappings_entity_with_schema(X, S):
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
    unique_ent = np.unique(np.concatenate((X[:, 0], X[:, 2], S[:, 0])))
    unique_rel = np.unique(X[:, 1])
    ent_count = len(unique_ent)
    rel_count = len(unique_rel)
    rel_to_idx = dict(zip(unique_rel, range(rel_count)))
    ent_to_idx = dict(zip(unique_ent, range(ent_count)))
    return rel_to_idx, ent_to_idx

def create_mappings_schema(S):
    """Create string-IDs mappings for classes and relations of the schema.

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
    unique_class = np.unique(S[:,2])
    unique_rel = np.unique(S[:,1])
    class_count = len(unique_class)
    rel_count = len(unique_rel)
    rel_to_idx = dict(zip(unique_rel, range(rel_count)))
    class_to_idx = dict(zip(unique_class, range(class_count)))
    return rel_to_idx, class_to_idx


def generate_corruptions_for_eval(x, idx_entities=None, filter=None, side='s+o'):
    """Generate corruptions for evaluation.

    Create all possible corruptions (subject and object) for a given triple x, in compliance with the LCWA.

    Parameters
    ----------
    x : ndarray, shape [1, 3]
        A single triple (as internal ID format).
    idx_entities: ndarray, shape [n, 1]
        The array of internal entity IDs.
    side : string
        Which side of the triple to corrupt: 's' will corrupt only the subject, 'o' will corrupt the object only,
        's+o' will create a list both subject and object corruptions (default).

    Returns
    -------
    x_neg : ndarray, shape [n, 3]
        All possible negatives corruptions or the triple x (subject and object corruptions).


    """

    if 's' in side:
        e_s = idx_entities[np.where(idx_entities != x[0])]
        x_n_s = np.stack((e_s, np.repeat(x[1], len(e_s)), np.repeat(x[2], len(e_s))), axis=1)
        raw = x_n_s

    if 'o' in side:
        e_o = idx_entities[np.where(idx_entities != x[2])]
        x_n_o = np.stack((np.repeat(x[0], len(e_o)), np.repeat(x[1], len(e_o)), e_o), axis=1)
        raw = x_n_o

    if side == 'o+s' or side == 's+o':
        raw = np.vstack((x_n_s, x_n_o))

    if filter is None:
        return raw
    else:
        raw_rows = raw.view([('', raw.dtype)] * raw.shape[1])
        filter_rows = filter.view([('', filter.dtype)] * filter.shape[1])
        return np.setdiff1d(raw_rows, filter_rows, assume_unique=True).view(raw.dtype).reshape(-1, raw.shape[1])


def generate_corruptions_for_fit(X, all_entities, eta=1, rnd=None):
    """Generate corruptions for training.

        Creates corrupted triples for each statement in an array of statements.

        Strategy as per ::cite:`trouillon2016complex`.

        .. note::
            Collisions are not checked. 
            Too computationally expensive (see ::cite:`trouillon2016complex`).

    Parameters
    ----------
    X : Tensor, shape [n, 3]
        An array of positive triples that will be used to create corruptions.
    all_entities : dict
        The entity-tointernal-IDs mappings
    eta : int
        The number of corruptions per triple that must be generated.
    rnd: numpy.random.RandomState
        A random number generator.

    Returns
    -------

    out : Tensor, shape [n * eta, 3]
        An array of corruptions for a list of positive triples x.

    """
    # idea: to make [1, 0, 2] becomes [1, 0, 3]
    # first random value 3
    # random position for 3 (s or o)
    # [1,0,3] = [1, 0, 2] * [1, 1, 0] + [0, 0, 3]
    # random entities with condition of s must different from o in X corr

    #Generate mask to replace subject entities in corruptions:
    #Uniformly decide where to replace subj entities in the batch of corruptions
    replace_subj = tf.greater(tf.random.uniform([ tf.shape(X)[0], eta, 1], 
                                                    seed=rnd), 0.5)
    #If not subject, replace object
    replace_obj = tf.logical_not(replace_subj)
    
    #Sample entity indices uniformly with which the subject must be replaced
    uniform_idx = tf.random.uniform([eta * tf.shape(X)[0]], 
                                        minval=0, 
                                        maxval = tf.shape(all_entities)[0],
                                        dtype=tf.int32, 
                                        seed=rnd)

    uniform_idx = tf.expand_dims(uniform_idx, 1)
    #Get the actual entitity
    random_sampled = tf.gather_nd(all_entities, uniform_idx)
    
    #Generate eta replacements for each subject 
    #(but replace only the ones where mask == True)
    #First repeat and create eta subject copies for each subject
    repeated_subjs = tf.keras.backend.repeat(
                                                tf.slice(X,
                                                    [0, 0], #subj
                                                    [tf.shape(X)[0],1])
                                            , eta)
    
    #based on the generated mask replace subject
    repeated_subjs = tf.where(replace_subj, 
                                tf.reshape(random_sampled,
                                            [ tf.shape(X)[0], eta, 1]), 
                                repeated_subjs)

    repeated_subjs = tf.squeeze(repeated_subjs, 2)
    
    #Sample entity indices uniformly with which the object must be replaced
    uniform_idx = tf.random.uniform([eta * tf.shape(X)[0]], 
                                    minval=0, 
                                    maxval = tf.shape(all_entities)[0], 
                                    seed=rnd, 
                                    dtype=tf.int32)

    uniform_idx = tf.expand_dims(uniform_idx, 1)
    #Get the actual entitity
    random_sampled = tf.gather_nd(all_entities, uniform_idx)
    
    #Generate eta replacements for each objects 
    #(but replace only the ones where mask == True)
    #First repeat and create eta object copies for each object
    repeated_objs = tf.keras.backend.repeat(
                                                tf.slice(X,
                                                        [0, 2], #Obj
                                                        [tf.shape(X)[0], 1])
                                            , eta)

    #based on the generated mask replace object
    repeated_objs = tf.where(replace_obj, 
                                tf.reshape(random_sampled,
                                                [ tf.shape(X)[0], eta, 1]), 
                                repeated_objs)

    repeated_objs = tf.squeeze(repeated_objs, 2)
    
    #Relations dont change while generating corruptions. 
    #So just repeat them eta times
    repeated_relns = tf.keras.backend.repeat(
                                                tf.slice(X,
                                                        [0, 1], #reln
                                                        [tf.shape(X)[0], 1])
                                            , eta)

    repeated_relns = tf.squeeze(repeated_relns, 2)
    
    #Stack the subject, relation, object
    out = tf.transpose(tf.stack([repeated_subjs, repeated_relns, repeated_objs] 
                                , 1),
                        [0, 2, 1])

    out = tf.reshape(out, [-1, tf.shape(X)[1]])
    return out           


def to_idx(X, ent_to_idx=None, rel_to_idx=None):
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

    x_idx_s = np.vectorize(ent_to_idx.get)(X[:, 0])
    x_idx_p = np.vectorize(rel_to_idx.get)(X[:, 1])
    x_idx_o = np.vectorize(ent_to_idx.get)(X[:, 2])

    return np.dstack([x_idx_s, x_idx_p, x_idx_o]).reshape((-1, 3))

def to_idx_schema(S, ent_to_idx=None, schema_class_to_idx=None, schema_rel_to_idx=None):
    """Convert schema statements (triples) into integer IDs.

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
        The ndarray of converted schema statements.
    """

    x_idx_ent = np.vectorize(ent_to_idx.get)(S[:, 0])
    x_idx_rel = np.vectorize(schema_rel_to_idx.get)(S[:, 1])
    x_idx_class = np.vectorize(schema_class_to_idx.get)(S[:, 2])

    return np.dstack([x_idx_ent, x_idx_rel, x_idx_class]).reshape((-1, 3))


def evaluate_performance(X, model, filter_triples=None, splits=10, side='s+o', corruption_entities=None, verbose=False):
    """Evaluate the performance of an embedding model.

        Run the relational learning evaluation protocol defined in Bordes TransE paper.

        It computes the mean reciprocal rank, by assessing the ranking of each positive triple against all
        possible negatives created in compliance with the local closed world assumption (LCWA).

    Parameters
    ----------
    X : ndarray, shape [n, 3]
        An array of test triples.
    model : ampligraph.latent_features.EmbeddingModel
        A knowledge graph embedding model
    filter_triples : ndarray of shape [n, 3] or None
        The triples used to filter negatives.
    splits : int
        The splits in which evaluate test data.
    side : string
        Which side of the triple to corrupt: 's' will corrupt only the subject, 'o' will corrupt the object only,
        's+o' will create a list both subject and object corruptions (default).
    corruption_entities : array-like
        List of entities to use for corruptions. If None, will generate corruptions
        using all distinct entities. Default is None.
    verbose : bool
        Verbose mode

    Returns
    -------
    ranks : ndarray, shape [n]
        An array of ranks of positive test triples.


    Examples
    --------
    >>> import numpy as np
    >>> from ampligraph.datasets import load_wn18
    >>> from ampligraph.latent_features import ComplEx
    >>> from ampligraph.evaluation import evaluate_performance
    >>>
    >>> X = load_wn18()
    >>> model = ComplEx(batches_count=10, seed=0, epochs=1, k=150, lr=.1, eta=10,
    >>>                 loss='pairwise', lambda_reg=0.01,
    >>>                 regularizer=None, optimizer='adagrad')
    >>> model.fit(np.concatenate((X['train'], X['valid'])))
    >>>
    >>> filter = np.concatenate((X['train'], X['valid'], X['test']))
    >>> ranks = evaluate_performance(X['test'][:5], model=model, filter_triples=filter)
    >>> ranks
    array([    2,     4,     1,     1, 28550], dtype=int32)
    >>> mrr_score(ranks)
    0.55000700525394053
    >>> hits_at_n_score(ranks, n=10)
    0.8
    """

    X = to_idx(X, ent_to_idx=model.ent_to_idx, rel_to_idx=model.rel_to_idx)

    if filter_triples is not None:
        filter_triples = to_idx(filter_triples, ent_to_idx=model.ent_to_idx, rel_to_idx=model.rel_to_idx)

    if corruption_entities is None:
        # conventional evaluation protocol: will use all entities in the KG to generate corruptions.
        idx_entities = np.asarray(list(model.ent_to_idx.values()))
    else:
        idx_entities = np.asarray([idx for uri, idx in model.ent_to_idx.items() if uri in corruption_entities])

    if 'unknown' in model.ent_to_idx:
        idx_entities = idx_entities[idx_entities != model.ent_to_idx['unknown']]

    ranks = np.zeros(len(X), dtype=np.int32)
    offset = 0

    cpu_cnt = os.cpu_count()

    # must be careful with memory footprint here. This is why we split in batches:
    pbar = tqdm(total=len(X), disable=not verbose, unit='triple')
    for X_batch in np.array_split(X, splits, axis=0):
        X_negs = Parallel(n_jobs=cpu_cnt)(delayed(generate_corruptions_for_eval)(x,
                                                                            filter=filter_triples,
                                                                            side=side,
                                                                            idx_entities=idx_entities) for x in X_batch)

        for i, x in enumerate(X_batch):
            y_pred = model.predict(np.vstack((x, X_negs[i])), from_idx=True)
            y_true = np.concatenate(([1], np.zeros(len(X_negs[i]), dtype=int)))
            ranks[offset + i] = rank_score(y_true, y_pred)
        offset += len(X_batch)

        pbar.update(len(X_batch))

    pbar.close()
    return ranks


def select_best_model_ranking(model_class, X, param_grid, filter_retrain=False, eval_splits=10,
                              corruption_entities=None, verbose=False):
    """Model selection routine for embedding models.

        .. note::
            Model selection done with raw MRR for better runtime performance.

        The function also retrains the best performing model on the concatenation of training and validation sets.

        Final evaluation on the test set is carried out by splitting the test sets, to keep memory consumption
        acceptable (note that we generate negatives at runtime according to the strategy described
        in ::cite:`bordes2013translating`).

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
    filter_retrain : bool
        If True, will use the entire input dataset X to compute filter MRR when retraining the model
        on the concatenation of training and validation sets.
    corruption_entities : array-like of shape [m]
        List of entities to use for corruptions. Useful to fit the evaluation protocol in memory when
        working with large KGs that include many distinct entities.
        If None, will generate corruptions using all distinct entities. Default is None.
    eval_splits : int
        The count of splits in which evaluate test data.
    verbose : bool
        Verbose mode

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
    >>> param_grid = {'batches_count': [10],
    >>>               'seed': [0],
    >>>               'epochs': [1],
    >>>               'k': [50, 150],
    >>>               'pairwise_margin': [1],
    >>>               'lr': [.1],
    >>>               'eta': [2],
    >>>               'loss': ['pairwise']}
    >>> select_best_model_ranking(model_class, X, param_grid, filter_retrain=True,
    >>>                           eval_splits=50, verbose=True)

    """

    model_params_combinations = (dict(zip(param_grid, x)) for x in itertools.product(*param_grid.values()))

    best_mrr_train = 0
    best_model = None
    best_params = None

    for model_params in model_params_combinations:
        model = model_class(**model_params)
        model.fit(X['train'])
        ranks = evaluate_performance(X['valid'], model=model, filter_triples=None,
                                     splits=eval_splits, corruption_entities=corruption_entities, verbose=verbose)
        curr_mrr = mrr_score(ranks)

        if curr_mrr > best_mrr_train:
            best_mrr_train = curr_mrr
            best_model = model
            best_params = model_params

    # Retraining

    if filter_retrain:
        X_filter = np.concatenate((X['train'], X['valid'], X['test']))
    else:
        X_filter = None

    best_model.fit(np.concatenate((X['train'], X['valid'])))
    ranks_test = evaluate_performance(X['test'], model=best_model, filter_triples=X_filter, splits=eval_splits,
                                      verbose=verbose)
    mrr_test = mrr_score(ranks_test)

    return best_model, best_params, best_mrr_train, ranks_test, mrr_test
























