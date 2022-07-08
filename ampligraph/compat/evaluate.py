import logging
import numpy as np
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def evaluate_performance(X, model, filter_triples=None, verbose=False, filter_unseen=True, entities_subset=None,
                         corrupt_side='s,o', use_default_protocol=False, batch_size=100):
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

    use_default_protocol: bool
        Flag to indicate whether to use the standard protocol used in literature defined in
        :cite:`bordes2013translating` (default: False).
        If set to `True`, ``corrupt_side`` will be set to `'s,o'`.
        This corresponds to the evaluation protocol used in literature, where head and tail corruptions
        are evaluated separately, i.e. in corrupt_side='s,o' mode
    batch_size: int
        batch size to use for evaluation

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

    
    if use_default_protocol:
        logger.warning('DeprecationWarning: use_default_protocol will be removed in future. '
                       'Please use corrupt_side argument instead.')
        corrupt_side = 's,o'

    logger.debug('Evaluating the performance of the embedding model.')
    assert corrupt_side in ['s', 'o', 's+o', 's,o'], 'Invalid value for corrupt_side.'
    
    return model.evaluate(x=X,
                     batch_size=batch_size,
                     verbose=verbose,
                     use_filter={'train': filter_triples},
                     corrupt_side=corrupt_side,
                     entities_subset=entities_subset,
                     callbacks=None)
    

