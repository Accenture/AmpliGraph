from . import EmbeddingModel

@register_model("RandomBaseline")
class RandomBaseline(EmbeddingModel):
    """Random baseline

    A dummy model that assigns a pseudo-random score included between 0 and 1,
    drawn from a uniform distribution.

    The model is useful whenever you need to compare the performance of
    another model on a custom knowledge graph, and no other baseline is available.

    .. note:: Although the model still requires invoking the ``fit()`` method,
        no actual training will be carried out.

    Examples
    --------
    >>> import numpy as np
    >>> from ampligraph.latent_features import RandomBaseline
    >>> model = RandomBaseline()
    >>> X = np.array([['a', 'y', 'b'],
    >>>               ['b', 'y', 'a'],
    >>>               ['a', 'y', 'c'],
    >>>               ['c', 'y', 'a'],
    >>>               ['a', 'y', 'd'],
    >>>               ['c', 'y', 'd'],
    >>>               ['b', 'y', 'c'],
    >>>               ['f', 'y', 'e']])
    >>> model.fit(X)
    >>> model.predict(np.array([['f', 'y', 'e'], ['b', 'y', 'd']]))
    [0.5488135039273248, 0.7151893663724195]
    """

    def __init__(self, seed=DEFAULT_SEED):
        """Initialize the model
        
        Parameters
        ----------
        seed : int
            The seed used by the internal random numbers generator.
            
        """
        self.seed = seed
        self.is_fitted = False
        self.rnd = check_random_state(self.seed)
        self.eval_config = {}

    def _fn(e_s, e_p, e_o):
        pass

    def get_embeddings(self, entities, type='entity'):
        """Get the embeddings of entities or relations.

        .. Note ::
            Use :meth:`ampligraph.utils.create_tensorboard_visualizations` to visualize the embeddings with TensorBoard.

        Parameters
        ----------
        entities : array-like, dtype=int, shape=[n]
            The entities (or relations) of interest. Element of the vector must be the original string literals, and
            not internal IDs.
        type : string
            If 'entity', will consider input as KG entities. If `relation`, they will be treated as KG predicates.

        Returns
        -------
        embeddings : None
            Returns None as this model does not have any embeddings.
            While scoring, it creates a random score for a triplet.

        """
        return None

    def fit(self, X):
        """Train the random model

        Parameters
        ----------
        X : ndarray, shape [n, 3]
            The training triples
        """
        self.rel_to_idx, self.ent_to_idx = create_mappings(X)
        self.is_fitted = True

    def end_evaluation(self):
        """End the evaluation
        """
        self.is_filtered = False
        self.eval_config = {}

    def predict(self, X, from_idx=False, get_ranks=False):
        """Assign random scores to candidate triples and then ranks them

        Parameters
        ----------
        X : ndarray, shape [n, 3]
            The triples to score.
        from_idx : bool
            If True, will skip conversion to internal IDs. (default: False).
        get_ranks : bool
            Flag to compute ranks by scoring against corruptions (default: False).
            
        Returns
        -------
        scores : ndarray, shape [n]
            The predicted scores for input triples X.
            
        ranks : ndarray, shape [n]
            Rank of the triple

        """
        if X.ndim == 1:
            X = np.expand_dims(X, 0)

        positive_scores = self.rnd.uniform(low=0, high=1, size=len(X)).tolist()
        if get_ranks:
            corruption_entities = self.eval_config.get('corruption_entities', DEFAULT_CORRUPTION_ENTITIES)
            if corruption_entities == "all":
                corruption_length = len(self.ent_to_idx)
            else:
                corruption_length = len(corruption_entities)

            corrupt_side = self.eval_config.get('corrupt_side', DEFAULT_CORRUPT_SIDE_EVAL)
            if corrupt_side == 's+o':
                # since we are corrupting both subject and object
                corruption_length *= 2
                # to account for the positive that we are testing
                corruption_length -= 2
            else:
                # to account for the positive that we are testing
                corruption_length -= 1
            ranks = []
            for i in range(len(X)):
                rank = np.sum(self.rnd.uniform(low=0, high=1, size=corruption_length) >= positive_scores[i]) + 1
                ranks.append(rank)

            return positive_scores, ranks

        return positive_scores


