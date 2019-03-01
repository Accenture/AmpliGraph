import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import normalize
from itertools import chain

"""This module contains learning-to-rank metrics to evaluate the performance of neural graph embedding models."""


def hits_at_n_score(ranks, n):
    """Hits@n metric.

    .. math::

        Hits@N = \sum_{i = 1}^{|Q|} 1 \, if rank_{(s, p, o)_i} \leq N

    Parameters
    ----------
    ranks: ndarray, shape [n]
        Input ranks of n positive statements.
    n: int
        The maximum rank considered to accept a positive.

    Returns
    -------
    hits_n_score: float
        The Hits@n score

    Examples
    --------
    >>> import numpy as np
    >>> from ampligraph.evaluation.metrics import hits_at_n_score
    >>> rankings = np.array([1, 12, 6, 2])
    >>> hits_at_n_score(rankings, n=3)
    0.5


    """

    if isinstance(ranks, list):
        ranks = np.asarray(ranks)

    return np.sum(ranks <= n) / len(ranks)


def mrr_score(ranks):
    """Mean Reciprocal Rank (MRR).

    .. math::

        MRR = \\frac{1}{|Q|}\sum_{i = 1}^{|Q|}\\frac{1}{rank_{(s, p, o)_i}}



    Parameters
    ----------
    ranks: ndarray, shape [n]
        Input ranks of n positive statements.

    Returns
    -------
    hits_n_score: float
        The MRR score

    Examples
    --------
    >>> import numpy as np
    >>> from ampligraph.evaluation.metrics import mrr_score
    >>> rankings = np.array([1, 12, 6, 2])
    >>> mrr_score(rankings)
    0.4375

    """
    if isinstance(ranks, list):
        ranks = np.asarray(ranks)

    return np.sum(1/ranks)/len(ranks)


def rank_score(y_true, y_pred, pos_lab=1):
    """Rank score metric.

        The rank of a positive element against a list of negatives.

    .. math::

        rank_{(s, p, o)_i}

    Parameters
    ----------
    y_true : ndarray, shape [n]
        An array of binary labels. The array only contains one positive.
    y_pred : ndarray, shape [n]
        An array of scores, for the positive element and the n-1 negatives.
    pos_lab : int
        The value of the positive label (default = 1)


    Returns
    -------
    rank : int
        The rank of the positive element against the negatives.


    Examples
    --------
    >>> from ampligraph.evaluation.metrics import rank_score
    >>> y_pred = np.array([.434, .65, .21, .84])
    >>> y_true = np.array([0, 0, 1, 0])
    >>> rank_score(y_true, y_pred)
    4

    """

    idx = np.argsort(y_pred)[::-1]
    y_ord = y_true[idx]
    rank = np.where(y_ord == pos_lab)[0][0] + 1
    return rank


def quality_loss_mse(original_model, subset_model, triple_list, norm=None):
    """ Mean squared error metric to measure the quality loss between two EmbeddingModels.
        
        Parameters
        ----------
        original_model : EmbeddingModel
            An embedding model trained on a graph G
        subset_model : EmbeddingModel
            An embedding model trained on a subset of G [triple_list]
        triple_list : np.ndarray, shape [n, 3]
           An array-like of triples [subject, predicate, object], the training set for subset_model.
           Used to evaluate the quality loss between embeddings of original_model and subset_model.
        norm : str or None
            If set to `l2`, will normalize the embeddings with L2 norm before computing the mean squared error.

        Returns
        -------
        mse : float
            The mean squared error between original_model and subset_model for embeddings of all entities and relations in triple list.
    """
    entities = list(chain.from_iterable([[triple[0], triple[2]] for triple in triple_list]))
    relations = [triple[1] for triple in triple_list]
    orig_embeds = np.vstack([original_model.get_embeddings(entities), original_model.get_embeddings(relations, type="relation")])
    subset_embeds = np.vstack([subset_model.get_embeddings(entities), subset_model.get_embeddings(relations, type="relation")])

    # TEC-1838
    if norm == 'l2':
        orig_embeds = normalize(orig_embeds, norm=norm)
        subset_embeds = normalize(subset_embeds, norm=norm)
    elif norm is not None and norm != 'l2':
        raise ValueError('Normalization not supported: %s' % norm)

    mse = mean_squared_error(orig_embeds, subset_embeds)
    return(mse)


def mar_score(ranks):
    """ Mean Average Rank score.

        Examples
        --------
        >>> from ampligraph.temporal.evaluation_function import mar_score
        >>> ranks= [5, 3, 4, 10, 1]
        >>> mar_score(ranks)
        4.6
    """

    if isinstance(ranks, list):
        ranks = np.asarray(ranks)
    return np.sum(ranks)/len(ranks)
