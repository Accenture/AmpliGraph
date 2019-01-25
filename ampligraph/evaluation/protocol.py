import numpy as np
from tqdm import tqdm


from ..evaluation import rank_score, mrr_score, hits_at_n_score, mar_score
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


def generate_corruptions_for_eval(X, all_entities, table_entity_lookup_left=None, 
                                      table_entity_lookup_right=None, table_reln_lookup=None, rnd=None):
    """Generate corruptions for evaluation.

        Create all possible corruptions (subject and object) for a given triple x, in compliance with the LCWA.

    Parameters
    ----------
    X : Tensor, shape [1, 3]
        Currently, a single positive triples that will be used to create corruptions.
    all_entities : Tensor
        All the entity IDs
    table_entity_lookup_left : tf.HashTable
        Hash table of subject entities mapped to unique prime numbers
    table_entity_lookup_right : tf.HashTable
        Hash table of object entities mapped to unique prime numbers
    table_reln_lookup : tf.HashTable
        Hash table of relations mapped to unique prime numbers
    rnd: numpy.random.RandomState
        A random number generator.

    Returns
    -------

    out : Tensor, shape [n, 3]
        An array of corruptions for the triples for x.
        
    out_prime : Tensor, shape [n, 3]
        An array of product of prime numbers associated with corruption triples or None 
        based on filtered or non filtered version.

    """
    
    #get the subject entities
    repeated_subjs = tf.keras.backend.repeat(
                                                tf.slice(X,
                                                    [0, 0], #subj
                                                    [tf.shape(X)[0],1])
                                            , tf.shape(all_entities)[0])


    repeated_objs = tf.keras.backend.repeat(
                                                tf.slice(X,
                                                        [0, 2], #Obj
                                                        [tf.shape(X)[0], 1])
                                            , tf.shape(all_entities)[0])



    repeated_relns = tf.keras.backend.repeat(
                                                tf.slice(X,
                                                        [0, 1], #reln
                                                        [tf.shape(X)[0], 1])
                                            , tf.shape(all_entities)[0])

    rep_ent = tf.keras.backend.repeat(tf.expand_dims(all_entities,0), tf.shape(X)[0])


    repeated_subjs = tf.squeeze(repeated_subjs, 2)
    repeated_relns = tf.squeeze(repeated_relns, 2)
    repeated_objs = tf.squeeze(repeated_objs, 2)
    rep_ent = tf.squeeze(rep_ent, 0)
    stacked_out = tf.concat([tf.stack([repeated_subjs, repeated_relns, rep_ent], 1),
                        tf.stack([rep_ent, repeated_relns, repeated_objs], 1)],0)
    out = tf.reshape(tf.transpose(stacked_out , [0, 2, 1]),(-1,3))
    out_prime = tf.constant([])
    
    if table_entity_lookup_left!= None and table_entity_lookup_right!=None and table_reln_lookup != None:
        prime_subj = tf.squeeze(table_entity_lookup_left.lookup(repeated_subjs))
        prime_reln =tf.squeeze(table_reln_lookup.lookup(repeated_relns))
        prime_obj = tf.squeeze(table_entity_lookup_right.lookup(repeated_objs))
        prime_ent_left = tf.squeeze(table_entity_lookup_left.lookup(rep_ent))
        prime_ent_right = tf.squeeze(table_entity_lookup_right.lookup(rep_ent))
        out_prime = tf.concat([prime_subj * prime_reln * prime_ent_right, 
                               prime_ent_left * prime_reln * prime_obj],0)

    
    
    
    return out, out_prime


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
	dataset =  tf.reshape(tf.tile(tf.reshape(X,[-1]),[eta]),[tf.shape(X)[0]*eta,3])
	keep_subj_mask = tf.tile(tf.cast(tf.random_uniform([tf.shape(X)[0]], 0, 2, dtype=tf.int32, seed=rnd),tf.bool),[eta])
	keep_obj_mask = tf.logical_not(keep_subj_mask)
	keep_subj_mask = tf.cast(keep_subj_mask,tf.int32)
	keep_obj_mask = tf.cast(keep_obj_mask,tf.int32)

	replacements = tf.random_uniform([tf.shape(dataset)[0]],0,tf.shape(all_entities)[0], dtype=tf.int32, seed=rnd)

	subjects = tf.math.add(tf.math.multiply(keep_subj_mask,dataset[:,0]),tf.math.multiply(keep_obj_mask,replacements))
	relationships = dataset[:,1]
	objects = tf.math.add(tf.math.multiply(keep_obj_mask,dataset[:,2]),tf.math.multiply(keep_subj_mask,replacements))
	
	out = tf.transpose(tf.stack([subjects,relationships,objects]))
	
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


def evaluate_performance(X, model, filter_triples=None, verbose=False):
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
    X_test = to_idx(X, ent_to_idx=model.ent_to_idx, rel_to_idx=model.rel_to_idx)

    if filter_triples is not None:
        filter_triples = to_idx(filter_triples, ent_to_idx=model.ent_to_idx, rel_to_idx=model.rel_to_idx)
        model.set_filter_for_eval(filter_triples)
    
    ranks = []
    for i in range(X_test.shape[0]):
        y_pred, rank = model.predict(X_test[i], from_idx=True)
        ranks.append(rank)
    
    model.end_evaluation()
    

    return ranks

def yield_all_permutations(registry, category_type, category_type_params):
    """Yields all the permutation of category type with their respective hyperparams
    
    Parameters
    ----------
    registry: registry of the category type
    category_type: category type values
    category_type_params: category type hyperparams

    Returns:
    name: specific name of the category
    present_params: names of hyperparameters of the category
    val: values of the respective hyperparams
    """
    for name in category_type:
        present_params = []
        present_params_vals = []
        for param in registry[name].external_params:
            try:
                present_params_vals.append(category_type_params[param])
                present_params.append(param)
            except KeyError:
                pass
        for val in itertools.product(*present_params_vals):
            yield name, present_params, val

def gridsearch_next_hyperparam(model_name, in_dict):
    """Performs grid search on hyperparams
    
    Parameters
    ----------
    model_name: name of the embedding model
    in_dict: dictionary of all the parameters and the list of values to be searched

    Returns:
    out_dict: dictionary containing an instance of model hypermeters
    """
    from ..latent_features import LOSS_REGISTRY, REGULARIZER_REGISTRY, MODEL_REGISTRY
    for batch_count in in_dict["batches_count"]:
        for epochs in in_dict["epochs"]:
            for k in in_dict["k"]:
                for eta in in_dict["eta"]:
                    for reg_type, reg_params, reg_param_values in \
                        yield_all_permutations(REGULARIZER_REGISTRY, in_dict["regularizer"], in_dict["regularizer_params"]):
                        for optimizer_type in in_dict["optimizer"]:
                            for optimizer_lr in in_dict["optimizer_params"]["lr"]:
                                for loss_type, loss_params, loss_param_values in \
                                    yield_all_permutations(LOSS_REGISTRY, in_dict["loss"], in_dict["loss_params"]):
                                    for model_type, model_params, model_param_values in \
                                        yield_all_permutations(MODEL_REGISTRY, [model_name], in_dict["embedding_model_params"]):
                                        
                                        try:
                                            verbose = in_dict["verbose"]
                                        except KeyError:
                                            verbose = False
                                            
                                        try:
                                            seed = in_dict["seed"]
                                        except KeyError:
                                            seed = -1
                                            
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
                                            "optimizer_params":{
                                                "lr": optimizer_lr
                                                },
                                            "verbose": verbose
                                            }
                                
                                        if seed >= 0:
                                            out_dict["seed"] = seed
                                            
                                        for idx in range(len(loss_params)):
                                            out_dict["loss_params"][loss_params[idx]] = loss_param_values[idx]
                                        for idx in range(len(reg_params)):
                                            out_dict["regularizer_params"][reg_params[idx]] = reg_param_values[idx]
                                        for idx in range(len(model_params)):
                                            out_dict["embedding_model_params"][model_params[idx]] = model_param_values[idx] 
                                            
                                        yield (out_dict)
                                        



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
    hyperparams_list_keys = ["batches_count", "epochs", "k", "eta", "loss", "regularizer", "optimizer"]
    hyperparams_dict_keys = ["loss_params", "embedding_model_params",  "regularizer_params",  "optimizer_params"]
    
    for key in hyperparams_list_keys:
        if key not in param_grid.keys() or param_grid[key]==[]:
            raise ValueError('Please pass values for key ' + key)
            
    for key in hyperparams_dict_keys:
        if key not in param_grid.keys():
            param_grid[key] = {}
    
    #this would be extended later to take multiple params for optimizers(currently only lr supported)
    try:
        lr = param_grid["optimizer_params"]["lr"]
    except KeyError:
        raise ValueError('Please pass values for optimizer parameter - lr')
    
    model_params_combinations = gridsearch_next_hyperparam(model_class.name, param_grid)
    
    best_mrr_train = 0
    best_model = None
    best_params = None
    for model_params in model_params_combinations:
        model = model_class(**model_params)
        model.fit(X['train'])
        ranks = evaluate_performance(X['valid'], model=model, filter_triples=None, verbose=verbose)
        curr_mrr = mrr_score(ranks)
        mr = mar_score(ranks)
        hits_1 = hits_at_n_score(ranks, n=1)
        hits_3 = hits_at_n_score(ranks, n=3)
        hits_10 = hits_at_n_score(ranks, n=10)
        if verbose:
            print("mr:{0} mrr: {1} hits 1: {2} hits 3: {3} hits 10: {4}, model: {5}, params: {6}".format(mr, curr_mrr, hits_1, hits_3, hits_10, type(model).__name__, model_params))

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
    ranks_test = evaluate_performance(X['test'], model=best_model, filter_triples=X_filter, verbose=verbose)
    mrr_test = mrr_score(ranks_test)

    return best_model, best_params, best_mrr_train, ranks_test, mrr_test
























