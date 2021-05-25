# Copyright 2019-2021 The AmpliGraph Authors. All Rights Reserved.
#
# This file is Licensed under the Apache License, Version 2.0.
# A copy of the Licence is available in LICENCE, or at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
from .ComplEx import ComplEx, register_model
from ampligraph.latent_features import constants as constants
from ampligraph.latent_features.initializers import DEFAULT_XAVIER_IS_UNIFORM


@register_model("HolE", ["negative_corruption_entities"])
class HolE(ComplEx):
    r"""Holographic Embeddings

    The HolE model :cite:`nickel2016holographic` as re-defined by Hayashi et al. :cite:`HayashiS17`:

    .. math::

        f_{HolE}= \frac{2}{n} \, f_{ComplEx}

    Examples
    --------
    >>> import numpy as np
    >>> from ampligraph.latent_features import HolE
    >>> model = HolE(batches_count=1, seed=555, epochs=100, k=10, eta=5,
    >>>             loss='pairwise', loss_params={'margin':1},
    >>>             regularizer='LP', regularizer_params={'lambda':0.1})
    >>>
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
    [[0.009254738], [0.00023370088]]
   """
    def __init__(self,
                 k=constants.DEFAULT_EMBEDDING_SIZE,
                 eta=constants.DEFAULT_ETA,
                 epochs=constants.DEFAULT_EPOCH,
                 batches_count=constants.DEFAULT_BATCH_COUNT,
                 seed=constants.DEFAULT_SEED,
                 embedding_model_params={'negative_corruption_entities': constants.DEFAULT_CORRUPTION_ENTITIES,
                                         'corrupt_sides': constants.DEFAULT_CORRUPT_SIDE_TRAIN},
                 optimizer=constants.DEFAULT_OPTIM,
                 optimizer_params={'lr': constants.DEFAULT_LR},
                 loss=constants.DEFAULT_LOSS,
                 loss_params={},
                 regularizer=constants.DEFAULT_REGULARIZER,
                 regularizer_params={},
                 initializer=constants.DEFAULT_INITIALIZER,
                 initializer_params={'uniform': DEFAULT_XAVIER_IS_UNIFORM},
                 verbose=constants.DEFAULT_VERBOSE):
        """Initialize an EmbeddingModel

        Also creates a new Tensorflow session for training.

        Parameters
        ----------
        k : int
            Embedding space dimensionality
        eta : int
            The number of negatives that must be generated at runtime during training for each positive.
        epochs : int
            The iterations of the training loop.
        batches_count : int
            The number of batches in which the training set must be split during the training loop.
        seed : int
            The seed used by the internal random numbers generator.
        embedding_model_params : dict
            HolE-specific hyperparams:

            - **negative_corruption_entities** - Entities to be used for generation of corruptions while training.
              It can take the following values :
              ``all`` (default: all entities),
              ``batch`` (entities present in each batch),
              list of entities
              or an int (which indicates how many entities that should be used for corruption generation).
            - **corrupt_sides** : Specifies how to generate corruptions for training.
              Takes values `s`, `o`, `s+o` or any combination passed as a list.
            - **'non_linearity'**: can be one of the following values ``linear``, ``softplus``, ``sigmoid``, ``tanh``
            - **'stop_epoch'**: specifies how long to decay (linearly) the numeric values from 1 to original value 
            until it reachs original value.
            - **'structural_wt'**: structural influence hyperparameter [0, 1] that modulates the influence of graph 
            topology. 
            - **'normalize_numeric_values'**: normalize the numeric values, such that they are scaled between [0, 1]

            The last 4 parameters are related to FocusE layers.

        optimizer : string
            The optimizer used to minimize the loss function. Choose between 'sgd',
            'adagrad', 'adam', 'momentum'.

        optimizer_params : dict
            Arguments specific to the optimizer, passed as a dictionary.

            Supported keys:

            - **'lr'** (float): learning rate (used by all the optimizers). Default: 0.1.
            - **'momentum'** (float): learning momentum (only used when ``optimizer=momentum``). Default: 0.9.

            Example: ``optimizer_params={'lr': 0.01}``

        loss : string
            The type of loss function to use during training.

            - ``pairwise``  the model will use pairwise margin-based loss function.
            - ``nll`` the model will use negative loss likelihood.
            - ``absolute_margin`` the model will use absolute margin likelihood.
            - ``self_adversarial`` the model will use adversarial sampling loss function.
            - ``multiclass_nll`` the model will use multiclass nll loss.
              Switch to multiclass loss defined in :cite:`chen2015` by passing
              'corrupt_sides' as ['s','o'] to embedding_model_params.
              To use loss defined in :cite:`kadlecBK17` pass 'corrupt_sides' as 'o' to embedding_model_params.

        loss_params : dict
            Dictionary of loss-specific hyperparameters. See :ref:`loss functions <loss>`
            documentation for additional details.

            Example: ``optimizer_params={'lr': 0.01}`` if ``loss='pairwise'``.

        regularizer : string
            The regularization strategy to use with the loss function.

            - ``None``: the model will not use any regularizer (default)
            - 'LP': the model will use L1, L2 or L3 based on the value of ``regularizer_params['p']`` (see below).

        regularizer_params : dict
            Dictionary of regularizer-specific hyperparameters. See the :ref:`regularizers <ref-reg>`
            documentation for additional details.

            Example: ``regularizer_params={'lambda': 1e-5, 'p': 2}`` if ``regularizer='LP'``.

        initializer : string
            The type of initializer to use.

            - ``normal``: The embeddings will be initialized from a normal distribution
            - ``uniform``: The embeddings will be initialized from a uniform distribution
            - ``xavier``: The embeddings will be initialized using xavier strategy (default)

        initializer_params : dict
            Dictionary of initializer-specific hyperparameters. See the
            :ref:`initializer <ref-init>`
            documentation for additional details.

            Example: ``initializer_params={'mean': 0, 'std': 0.001}`` if ``initializer='normal'``.

        verbose : bool
            Verbose mode.
        """
        super().__init__(k=k, eta=eta, epochs=epochs, batches_count=batches_count, seed=seed,
                         embedding_model_params=embedding_model_params,
                         optimizer=optimizer, optimizer_params=optimizer_params,
                         loss=loss, loss_params=loss_params,
                         regularizer=regularizer, regularizer_params=regularizer_params,
                         initializer=initializer, initializer_params=initializer_params,
                         verbose=verbose)
        self.internal_k = self.k * 2

    def _fn(self, e_s, e_p, e_o):
        """The Hole scoring function.

        The function implements the scoring function as defined by
        .. math::

            f_{HolE}= 2 / n * f_{ComplEx}

        Additional details for equivalence of the models available in :cite:`HayashiS17`.

        Parameters
        ----------
        e_s : Tensor, shape [n]
            The embeddings of a list of subjects.
        e_p : Tensor, shape [n]
            The embeddings of a list of predicates.
        e_o : Tensor, shape [n]
            The embeddings of a list of objects.

        Returns
        -------
        score : TensorFlow operation
            The operation corresponding to the HolE scoring function.

        """
        return (2 / self.k) * (super()._fn(e_s, e_p, e_o))

    def fit(self, X, early_stopping=False, early_stopping_params={}, focusE_numeric_edge_values=None,
            tensorboard_logs_path=None):
        """Train a HolE model.

        The model is trained on a training set X using the training protocol
        described in :cite:`nickel2016holographic`.

        Parameters
        ----------
        X : ndarray, shape [n, 3]
            The training triples
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

        early_stopping_params: dictionary
            Dictionary of hyperparameters for the early stopping heuristics.

            The following string keys are supported:

                - **'x_valid'**: ndarray, shape [n, 3] : Validation set to be used for early stopping.
                - **'criteria'**: string : criteria for early stopping 'hits10', 'hits3', 'hits1' or 'mrr'(default).
                - **'x_filter'**: ndarray, shape [n, 3] : Positive triples to use as filter if a 'filtered'
                  early stopping criteria is desired (i.e. filtered-MRR if 'criteria':'mrr').
                  Note this will affect training time (no filter by default).
                - **'burn_in'**: int : Number of epochs to pass before kicking in early stopping (default: 100).
                - **check_interval'**: int : Early stopping interval after burn-in (default:10).
                - **'stop_interval'**: int : Stop if criteria is performing worse over n consecutive checks (default: 3)
                - **'corruption_entities'**: List of entities to be used for corruptions.
                  If 'all', it uses all entities (default: 'all')
                - **'corrupt_side'**: Specifies which side to corrupt. 's', 'o', 's+o' (default)

                Example: ``early_stopping_params={x_valid=X['valid'], 'criteria': 'mrr'}``
        
        focusE_numeric_edge_values: nd array (n, 1)
            Numeric values associated with links. 
            Semantically, the numeric value can signify importance, uncertainity, significance, confidence, etc.
            If the numeric value is unknown pass a NaN weight. The model will uniformly randomly assign a numeric value.
            One can also think about assigning numeric values by looking at the distribution of it per predicate.
 
        tensorboard_logs_path: str or None
            Path to store tensorboard logs, e.g. average training loss tracking per epoch (default: ``None`` indicating
            no logs will be collected). When provided it will create a folder under provided path and save tensorboard 
            files there. To then view the loss in the terminal run: ``tensorboard --logdir <tensorboard_logs_path>``.
           
        """
        super().fit(X, early_stopping, early_stopping_params, focusE_numeric_edge_values,
                    tensorboard_logs_path=tensorboard_logs_path)

    def predict(self, X, from_idx=False):
        __doc__ = super().predict.__doc__  # NOQA
        return super().predict(X, from_idx=from_idx)

    def calibrate(self, X_pos, X_neg=None, positive_base_rate=None, batches_count=100, epochs=50):
        __doc__ = super().calibrate.__doc__ # NOQA
        super().calibrate(X_pos, X_neg, positive_base_rate, batches_count, epochs)

    def predict_proba(self, X):
        __doc__ = super().predict_proba.__doc__ # NOQA
        return super().predict_proba(X)
