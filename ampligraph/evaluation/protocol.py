# Copyright 2019-2023 The AmpliGraph Authors. All Rights Reserved.
#
# This file is Licensed under the Apache License, Version 2.0.
# A copy of the Licence is available in LICENCE, or at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#

import logging
from collections.abc import Iterable
from itertools import islice, product

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..evaluation import hits_at_n_score, mr_score, mrr_score

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

TOO_MANY_ENTITIES_TH = 50000


def train_test_split_no_unseen(
    X,
    test_size=100,
    seed=0,
    allow_duplication=False,
    filtered_test_predicates=None,
):
    """Split into train and test sets.

    This function carves out a test set that contains only entities
    and relations which also occur in the training set.

    This is an improved version which is much faster - since this does not sample like in the earlier approach but
    rather shuffles indices and gets the test set of required size by selecting from the shuffled indices only triples
    which do not disconnect entities/relations.

    Parameters
    ----------
    X : ndarray, shape (n, 3)
        The dataset to split.
    test_size : int, float
        If `int`, the number of triples in the test set.
        If `float`, the percentage of total triples.
    seed : int
        A random seed used to split the dataset.
    allow_duplication: bool
        Flag to indicate if the test set can contain duplicated triples.
    filtered_test_predicates: None, list
        If `None`, all predicate types will be considered for the test set.
        If `list`, only the predicate types in the list will be considered for
        the test set.

    Returns
    -------
    X_train : ndarray, shape (n, 3)
        The training set.
    X_test : ndarray, shape (n, 3)
        The test set.

    Example
    -------
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

    np.random.seed(seed)
    if filtered_test_predicates:
        candidate_idx = np.isin(X[:, 1], filtered_test_predicates)
        X_test_candidates = X[candidate_idx]
        X_train = X[~candidate_idx]
    else:
        X_train = None
        X_test_candidates = X

    if isinstance(test_size, float):
        test_size = int(len(X_test_candidates) * test_size)

    entities, entity_cnt = np.unique(
        np.concatenate([X_test_candidates[:, 0], X_test_candidates[:, 2]]),
        return_counts=True,
    )
    rels, rels_cnt = np.unique(X_test_candidates[:, 1], return_counts=True)
    dict_entities = dict(zip(entities, entity_cnt))
    dict_rels = dict(zip(rels, rels_cnt))
    idx_test = []
    idx_train = []

    all_indices_shuffled = np.random.permutation(
        np.arange(X_test_candidates.shape[0])
    )

    for i, idx in enumerate(all_indices_shuffled):
        test_triple = X_test_candidates[idx]
        # reduce the entity and rel count
        dict_entities[test_triple[0]] = dict_entities[test_triple[0]] - 1
        dict_rels[test_triple[1]] = dict_rels[test_triple[1]] - 1
        dict_entities[test_triple[2]] = dict_entities[test_triple[2]] - 1

        # test if the counts are > 0
        if (
            dict_entities[test_triple[0]] > 0
            and dict_rels[test_triple[1]] > 0
            and dict_entities[test_triple[2]] > 0
        ):
            # Can safetly add the triple to test set
            idx_test.append(idx)
            if len(idx_test) == test_size:
                # Since we found the requested test set of given size
                # add all the remaining indices of candidates to training set
                idx_train.extend(list(all_indices_shuffled[i + 1:]))

                # break out of the loop
                break

        else:
            # since removing this triple results in unseen entities, add it to
            # training
            dict_entities[test_triple[0]] = dict_entities[test_triple[0]] + 1
            dict_rels[test_triple[1]] = dict_rels[test_triple[1]] + 1
            dict_entities[test_triple[2]] = dict_entities[test_triple[2]] + 1
            idx_train.append(idx)

    if len(idx_test) != test_size:
        # if we cannot get the test set of required size that means we cannot get unique triples
        # in the test set without creating unseen entities
        if allow_duplication:
            # if duplication is allowed, randomly choose from the existing test
            # set and create duplicates
            duplicate_idx = np.random.choice(
                idx_test, size=(test_size - len(idx_test))
            ).tolist()
            idx_test.extend(list(duplicate_idx))
        else:
            # throw an exception since we cannot get unique triples in the test set without creating
            # unseen entities
            raise Exception(
                "Cannot create a test split of the desired size. "
                "Some entities will not occur in both training and test set. "
                "Set allow_duplication=True,"
                "remove filter on test predicates or "
                "set test_size to a smaller value."
            )

    if X_train is None:
        X_train = X_test_candidates[idx_train]
    else:
        X_train_subset = X_test_candidates[idx_train]
        X_train = np.concatenate([X_train, X_train_subset])
    X_test = X_test_candidates[idx_test]

    X_train = np.random.permutation(X_train)
    X_test = np.random.permutation(X_test)

    return X_train, X_test


def filter_unseen_entities(X, model, verbose=False):
    """Filter unseen entities in the test set.

    Parameters
    ----------
    X : ndarray, shape (n, 3)
        An array of test triples.
    model : ampligraph.latent_features.EmbeddingModel
        A knowledge graph embedding model.
    verbose : bool
        Verbose mode.

    Returns
    -------
    filtered X : ndarray, shape (n, 3)
        An array of test triples containing no unseen entities.
    """
    logger.debug(
        "Finding entities in the dataset that are not previously seen by model"
    )
    ent_seen = np.unique(list(model.ent_to_idx.keys()))
    df = pd.DataFrame(X, columns=["s", "p", "o"])
    filtered_df = df[df.s.isin(ent_seen) & df.o.isin(ent_seen)]
    n_removed_ents = df.shape[0] - filtered_df.shape[0]
    if n_removed_ents > 0:
        msg = "Removing {} triples containing unseen entities. ".format(
            n_removed_ents
        )
        if verbose:
            logger.info(msg)
        logger.debug(msg)
        return filtered_df.values
    return X


def _flatten_nested_keys(dictionary):
    """
    Flatten the nested values of a dictionary into tuple keys.

    E.g., {"a": {"b": [1], "c": [2]}} becomes {("a", "b"): [1], ("a", "c"): [2]}
    """
    # Find the parameters that are nested dictionaries
    nested_keys = {k for k, v in dictionary.items() if isinstance(v, dict)}
    # Flatten them into tuples
    flattened_nested_keys = {
        (nk, k): dictionary[nk][k]
        for nk in nested_keys
        for k in dictionary[nk]
    }
    # Get original dictionary without the nested keys
    dictionary_without_nested_keys = {
        k: v for k, v in dictionary.items() if k not in nested_keys
    }
    # Return merged dicts
    return {**dictionary_without_nested_keys, **flattened_nested_keys}


def _unflatten_nested_keys(dictionary):
    """
    Unflatten the nested values of a dictionary based on the keys that are tuples.

    E.g., {("a", "b"): [1], ("a", "c"): [2]} becomes {"a": {"b": [1], "c": [2]}}
    """
    # Find the parameters that are nested dictionaries
    nested_keys = {k[0] for k in dictionary if isinstance(k, tuple)}
    # Select the parameters which were originally nested and unflatten them
    nested_dict = {
        nk: {k[1]: v for k, v in dictionary.items() if k[0] == nk}
        for nk in nested_keys
    }
    # Get original dictionary without the nested keys
    dictionary_without_nested_keys = {
        k: v for k, v in dictionary.items() if not isinstance(k, tuple)
    }
    # Return merged dicts
    return {**dictionary_without_nested_keys, **nested_dict}


def _get_param_hash(param):
    """Get the hash of a param dictionary.

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
    hash : str
        Hash of the param dictionary.
    """
    # Remove parameters that are not used by particular configurations
    # For example, if the regularization is None, there is no need for the
    # regularization lambda
    flattened_params = _flatten_nested_keys(_unflatten_nested_keys(param))

    return hash(frozenset(flattened_params.items()))


class ParamHistory(object):
    """
    Used to evaluate whether a particular parameter configuration has already been previously seen or not.

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

    # Flatten nested dictionaries so we can apply itertools.product to get all
    # possible parameter combinations
    flattened_param_grid = _flatten_nested_keys(param_grid)

    for values in product(*flattened_param_grid.values()):
        # Get one single parameter combination as a flattened dictionary
        param = dict(zip(flattened_param_grid.keys(), values))

        # Only yield unique parameter combinations
        if param in param_history:
            continue
        else:
            param_history.add(param)
            # Yields nested configuration (unflattened) without useless
            # parameters
            yield _unflatten_nested_keys(param)


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
    params: dict
        Return dictionary containing sampled parameters.

    """
    param = {}
    for k, v in param_grid.items():
        if callable(v):
            param[k] = v()
        elif isinstance(v, dict):
            param[k] = _sample_parameters(v)
        elif isinstance(v, Iterable) and not isinstance(v, str):
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
            yield param


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
        if not (callable(v) or isinstance(v, Iterable)) or isinstance(v, str):
            param_grid[k] = [v]
        elif isinstance(v, dict):
            _scalars_into_lists(v)


def select_best_model_ranking(
    model_class,
    X_train,
    X_valid,
    X_test,
    param_grid,
    max_combinations=None,
    param_grid_random_seed=0,
    use_filter=True,
    early_stopping=False,
    early_stopping_params=None,
    use_test_for_selection=False,
    entities_subset=None,
    corrupt_side="s,o",
    focusE=False,
    focusE_params={},
    retrain_best_model=False,
    verbose=False,
):
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
    model_class : str
        The class of the EmbeddingModel to evaluate (`'TransE'`, `'DistMult'`, `'ComplEx'`, etc).
    X_train : ndarray, shape (n, 3)
        An array of training triples.
    X_valid : ndarray, shape (n, 3)
        An array of validation triples.
    X_test : ndarray, shape (n, 3)
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
        By default (`None`) all combinations will be explored,
        which makes it incompatible with random parameters for random search.
    param_grid_random_seed: int
        Random seed for the parameters that are callables and random.
    use_filter : bool
        If `True`, it will use the entire input dataset `X` to compute filtered MRR (default: `True`).
    early_stopping: bool
        Flag to enable early stopping (default: `False`).

        If set to `True`, the training loop adopts the following early stopping heuristic:

        - The model will be trained regardless of early stopping for ``burn_in`` epochs.
        - Every ``check_interval`` epochs the method will compute the metric specified in ``criteria``.

        If such metric decreases for ``stop_interval`` checks, we stop training early.

        Note the metric is computed on ``X_valid``. This is usually a validation set that you held out.

        Also, since ``criteria`` is a ranking metric, it requires generating negatives.
        Entities used to generate corruptions can be specified as the side(s) of a triple to corrupt.
        The method supports filtered metrics, by passing an array of positives to ``x_filter``. This will be used to
        filter the negatives generated on the fly (i.e., the corruptions).

        .. note::

            Keep in mind the early stopping criteria may introduce a certain overhead
            (caused by the metric computation).
            The goal is to strike a good trade-off between such overhead and saving training epochs.

            A common approach is to use MRR unfiltered: ::

                early_stopping_params={x_valid=X['valid'], 'criteria': 'mrr'}

            Note that the size of validation set also contributes to such overhead.
            In most cases a smaller validation set would be enough.

    early_stopping_params: dict
        Dictionary of parameters for early stopping.

        The following keys are supported:
            * x_valid: ndarray, shape (n, 3) - Validation set to be used for early stopping (default: `X['valid']`).
            * criteria: Criteria for early stopping ``hits10``, ``hits3``, ``hits1`` or ``mrr`` (default: `"mrr"`).
            * x_filter: ndarray, shape (n, 3) - Filter to be used (default: `None`).
            * burn_in: Number of epochs to pass before kicking in early stopping (default: 100).
            * check_interval: Early stopping interval after burn-in (default: 10).
            * stop_interval: Stop if criteria is performing worse over `n` consecutive checks (default: 3).

    focusE: bool
        Whether to use the focusE layer (default: `False`). If `True`, make sure you pass the weights as an additional
        column concatenated after the training triples.
    focusE_params: dict
        Dictionary of parameters if focusE is activated.
    use_test_for_selection:bool
        Use test set for model selection. If `False`, uses validation set (default: `False`).
    entities_subset: array-like
        List of entities to use for corruptions. If `None`, will generate corruptions
        using all distinct entities (default: `None`).
    corrupt_side: str
        Specifies which side to corrupt the entities:
        `"s"` to corrupt only subject.
        `"o"` to corrupt only object.
        `"s+o"` to corrupt both subject and object.
        `"s,o"` to corrupt both subject and object but ranks are computed separately (default).
    retrain_best_model: bool
        Flag to indicate whether best model should be re-trained at the end with the validation set used in the search
        (default: `False`).
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

    ranks_test : ndarray, shape (n) or (n,2)
        An array of ranks of test triples.
        When ``corrupt_side='s,o'`` the function returns an array of shape (n,2). The first column represents the
        rank against subject corruptions and the second column represents the rank against object corruptions.
        In other cases, it returns an array of size (n), i.e., rank against the specified corruptions.

    mrr_test : float
        The MRR (filtered) of the best model, retrained on the concatenation of training and validation sets,
        computed over the test set.

    experimental_history: list of dict
        A list containing all the intermediate experimental results:
        the model parameters and the corresponding validation metrics.

    Example
    -------
    >>> from ampligraph.datasets import load_wn18
    >>> from ampligraph.evaluation import select_best_model_ranking
    >>> import numpy as np
    >>>
    >>> X = load_wn18()
    >>> model_class = 'ComplEx'
    >>> param_grid = {
    >>>                "batches_count": [50],
    >>>                "seed": 0,
    >>>                "epochs": [4000],
    >>>                "k": [100, 200],
    >>>                "eta": [5,10,15],
    >>>                "loss": ["pairwise", "nll"],
    >>>                "loss_params": {
    >>>                    "margin": [2]
    >>>                },
    >>>                "embedding_model_params": {},
    >>>                "regularizer": ["LP", None],
    >>>                "regularizer_params": {
    >>>                    "p": [1, 3],
    >>>                    "lambda": [1e-4, 1e-5]
    >>>                },
    >>>                "optimizer": ["adagrad", "adam"],
    >>>                "optimizer_params":{
    >>>                    "lr": lambda: np.random.uniform(0.0001, 0.01)
    >>>                },
    >>>                "verbose": False
    >>>               }
    >>> select_best_model_ranking(model_class, X['train'], X['valid'], X['test'], param_grid,
    >>>                           max_combinations=100, use_filter=True, verbose=True,
    >>>                           early_stopping=True)

    """
    from importlib import import_module

    from ..compat import evaluate_performance

    compat_module = import_module("ampligraph.compat")
    model_class = getattr(compat_module, model_class)

    logger.debug(
        "Starting gridsearch over hyperparameters. {}".format(param_grid)
    )

    if early_stopping_params is None:
        early_stopping_params = {}

    # Verify missing parameters for the model class (default values will be
    # used)
    undeclared_args = set(model_class.__init__.__code__.co_varnames[1:]) - set(
        param_grid.keys()
    )
    if len(undeclared_args) != 0:
        logger.debug(
            "The following arguments were not defined in the parameter grid"
            " and thus the default values will be used: {}".format(
                ", ".join(undeclared_args)
            )
        )

    param_grid["model_name"] = model_class.name
    _scalars_into_lists(param_grid)

    if max_combinations is not None:
        np.random.seed(param_grid_random_seed)
        model_params_combinations = islice(
            _next_hyperparam_random(param_grid), max_combinations
        )
    else:
        model_params_combinations = _next_hyperparam(param_grid)
        max_combinations = 1
        for param in param_grid.values():
            if isinstance(param, list):
                max_combinations *= len(param)
            elif isinstance(param, dict):
                try:
                    max_combinations *= int(
                        np.prod(
                            [
                                len(el)
                                for el in param.values()
                                if isinstance(el, list)
                            ]
                        )
                    )
                except Exception as e:
                    logger.debug("Exception " + e)

    best_mrr_train = 0
    best_model = None
    best_params = None

    if early_stopping:
        try:
            early_stopping_params["x_valid"]
        except KeyError:
            logger.debug(
                "Early stopping enable but no x_valid parameter set. Setting x_valid to {}".format(
                    X_valid
                )
            )
            early_stopping_params["x_valid"] = X_valid

    focusE_numeric_edge_values = None
    if focusE:
        assert isinstance(X_train, np.ndarray) and X_train.shape[1] > 3, (
            "Weights are missing! Concatenate them to X_train"
            "in order to use FocusE!"
        )
        focusE_numeric_edge_values = X_train[:, 3:]
        param_grid["embedding_model_params"] = {
            **param_grid["embedding_model_params"],
            **focusE_params,
        }

    if use_filter:
        X_filter = {"train": X_train, "valid": X_valid, "test": X_test}
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

    print("Grid search initialized successfully, training starting!")
    print("Maximum number of combinations: ", max_combinations)
    for model_params in tqdm(
        model_params_combinations, total=max_combinations
    ):
        print()
        current_result = {
            "model_name": model_params["model_name"],
            "model_params": model_params,
        }
        del model_params["model_name"]
        try:
            model = model_class(**model_params)
            model.fit(
                X_train,
                early_stopping,
                early_stopping_params,
                focusE_numeric_edge_values=focusE_numeric_edge_values,
                verbose=verbose,
            )

            ranks = evaluate_performance(
                selection_dataset,
                model=model,
                filter_triples=X_filter,
                verbose=verbose,
                entities_subset=entities_subset,
                corrupt_side=corrupt_side,
            )

            curr_mrr, mr, hits_1, hits_3, hits_10 = evaluation(ranks)

            current_result["results"] = {
                "mrr": curr_mrr,
                "mr": mr,
                "hits_1": hits_1,
                "hits_3": hits_3,
                "hits_10": hits_10,
            }

            info = "mr: {} mrr: {} hits 1: {} hits 3: {} hits 10: {}, model: {}, params: {}".format(
                mr,
                curr_mrr,
                hits_1,
                hits_3,
                hits_10,
                type(model).__name__,
                model_params,
            )

            logger.debug(info)
            if verbose:
                logger.info(info)

            if curr_mrr > best_mrr_train:
                best_mrr_train = curr_mrr
                best_model = model
                best_params = model_params
        except Exception as e:
            current_result["results"] = {"exception": str(e)}

            if verbose:
                logger.error(
                    "Exception occurred for parameters:{}".format(model_params)
                )
                logger.error(str(e))
            else:
                pass
        experimental_history.append(current_result)
        print("Combination tried, on to the next one!")
    if best_model is not None:
        if retrain_best_model:
            if focusE:
                assert (
                    isinstance(X_valid, np.ndarray) and X_valid.shape[1] > 3
                ), (
                    "Validation set is used as training"
                    "data for retraining the best model,"
                    "but weights are missing."
                    "Concatenate them to X_valid!"
                )
                focusE_numeric_edge_values = np.concatenate(
                    [focusE_numeric_edge_values, X_valid[:, 3:]], axis=0
                )
            best_model.fit(
                np.concatenate((X_train, X_valid)),
                early_stopping,
                early_stopping_params,
                focusE_numeric_edge_values=focusE_numeric_edge_values,
            )

        ranks_test = evaluate_performance(
            X_test,
            model=best_model,
            filter_triples=X_filter,
            verbose=verbose,
            entities_subset=entities_subset,
            corrupt_side=corrupt_side,
        )

        test_mrr, test_mr, test_hits_1, test_hits_3, test_hits_10 = evaluation(
            ranks_test
        )

        info = "Best model test results: mr: {} mrr: {} hits 1: {} hits 3: {} hits 10: {}, model: {}, params: {}".format(
            test_mrr,
            test_mr,
            test_hits_1,
            test_hits_3,
            test_hits_10,
            type(best_model).__name__,
            best_params,
        )

        logger.debug(info)
        if verbose:
            logger.info(info)

        test_evaluation = {
            "mrr": test_mrr,
            "mr": test_mr,
            "hits_1": test_hits_1,
            "hits_3": test_hits_3,
            "hits_10": test_hits_10,
        }
    else:
        ranks_test = []

        test_evaluation = {
            "mrr": np.nan,
            "mr": np.nan,
            "hits_1": np.nan,
            "hits_3": np.nan,
            "hits_10": np.nan,
        }

    return (
        best_model,
        best_params,
        best_mrr_train,
        ranks_test,
        test_evaluation,
        experimental_history,
    )
