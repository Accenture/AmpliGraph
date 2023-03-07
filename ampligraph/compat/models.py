# Copyright 2019-2023 The AmpliGraph Authors. All Rights Reserved.
#
# This file is Licensed under the Apache License, Version 2.0.
# A copy of the Licence is available in LICENCE, or at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#

import numpy as np
import tensorflow as tf

from ampligraph.latent_features.loss_functions import get as get_loss
from ampligraph.latent_features.models import ScoringBasedEmbeddingModel
from ampligraph.latent_features.optimizers import get as get_optimizer
from ampligraph.latent_features.regularizers import get as get_regularizer

BACK_COMPAT_MODELS = {}


def register_compatibility(name):
    def insert_in_registry(class_handle):
        BACK_COMPAT_MODELS[name] = class_handle
        class_handle.name = name
        return class_handle

    return insert_in_registry


class ScoringModelBase:
    def __init__(
        self,
        k=100,
        eta=2,
        epochs=100,
        batches_count=100,
        seed=0,
        embedding_model_params={
            "corrupt_sides": ["s,o"],
            "negative_corruption_entities": "all",
            "norm": 1,
            "normalize_ent_emb": False,
        },
        optimizer="adam",
        optimizer_params={"lr": 0.0005},
        loss="nll",
        loss_params={},
        regularizer=None,
        regularizer_params={},
        initializer="xavier",
        initializer_params={"uniform": False},
        verbose=False,
        model=None,
    ):
        """Initialize the model class.

        Parameters
        ----------
        k : int
            Embedding space dimensionality.
        eta : int
            The number of negatives that must be generated at runtime during
            training for each positive.
        epochs : int
            The iterations of the training loop.
        batches_count : int
            The number of batches in which the training set must be split
            during the training loop.
        seed : int
            The seed used by the internal random numbers generator.
        embedding_model_params : dict
            Model-specific hyperparams, passed to the model as a dictionary.
            Refer to model-specific documentation for details.
        optimizer : str
            The optimizer used to minimize the loss function. Choose between
            'sgd', 'adagrad', 'adam', 'momentum'.
        optimizer_params : dict
            Arguments specific to the optimizer, passed as a dictionary.
            Supported keys:

            - **'lr'** (float): learning rate (used by all the optimizers).
            Default: 0.1.
            - **'momentum'** (float): learning momentum
            (only used when ``optimizer=momentum``). Default: 0.9.

        loss : str
            The type of loss function to use during training.

            - `"pairwise"`  the model will use pairwise margin-based
            loss function.
            - `"nll"` the model will use negative loss likelihood.
            - `"absolute_margin"` the model will use absolute
            margin likelihood.
            - `"self_adversarial"` the model will use adversarial sampling
            loss function.
            - `"multiclass_nll"` the model will use multiclass nll loss.
            Switch to multiclass loss defined in \
            :cite:`chen2015` by passing ``"corrupt_side"`` as `["s","o"]` to
            ``embedding_model_params``. To use loss defined in\
                    :cite:`kadlecBK17` pass ``"corrupt_side"``\
                    as `"o"` to embedding_model_params.

        loss_params : dict
            Dictionary of loss-specific hyperparameters.
        regularizer : str
            The regularization strategy to use with the loss function.

            - `None`: the model will not use any regularizer (default)
            - `LP`: the model will use :math:`L^1, L^2` or :math:`L^3`
            regularization based on the value of
            ``regularizer_params['p']`` in the ``regularizer_params``.

        regularizer_params : dict
            Dictionary of regularizer-specific hyperparameters.
        initializer : str
            The type of initializer to use.

            - `"normal"`: The embeddings will be initialized from a normal
            distribution
            - `"uniform"`: The embeddings will be initialized from a uniform
            distribution
            - `"xavier"`: The embeddings will be initialized using xavier
            strategy (default)

        initializer_params : dict
            Dictionary of initializer-specific hyperparameters.
        verbose : bool
            Verbose mode.
        """
        if model is not None:
            self.model_name = model.scoring_type
        else:
            self.k = k
            self.eta = eta
            self.seed = seed

            self.batches_count = batches_count

            self.epochs = epochs
            self.embedding_model_params = embedding_model_params
            self.optimizer = optimizer
            self.optimizer_params = optimizer_params
            self.loss = loss
            self.loss_params = loss_params
            self.initializer = initializer
            self.initializer_params = initializer_params
            self.regularizer = regularizer
            self.regularizer_params = regularizer_params
            self.verbose = verbose

        self.model = model
        self.is_backward = True

    def _get_optimizer(self, optimizer, optim_params):
        """Get the optimizer from tf.keras.optimizers."""
        learning_rate = optim_params.get("lr", 0.001)
        del optim_params["lr"]

        if optimizer == "adam":
            optim = tf.keras.optimizers.Adam(
                learning_rate=learning_rate, **optim_params
            )
            status = True
        elif optimizer == "adagrad":
            optim = tf.keras.optimizers.Adagrad(
                learning_rate=learning_rate, **optim_params
            )
            status = True
        elif optimizer == "sgd":
            optim = tf.keras.optimizers.SGD(
                learning_rate=learning_rate, **optim_params
            )
            status = True
        else:
            optim = get_optimizer(optimizer)
            status = False

        optim_params["lr"] = learning_rate
        return optim, status

    def is_fit(self):
        """Flag whether the model has been fitted or not."""
        return self.model.is_fit()

    def _get_initializer(self, initializer, initializer_params):
        """Get the initializers among tf.keras.initializers."""
        if initializer == "xavier":
            if initializer_params["uniform"]:
                return tf.keras.initializers.GlorotUniform(seed=self.seed)
            else:
                return tf.keras.initializers.GlorotNormal(seed=self.seed)
        elif initializer == "uniform":
            return tf.keras.initializers.RandomUniform(
                minval=initializer_params.get("low", -0.05),
                maxval=initializer_params.get("high", 0.05),
                seed=self.seed,
            )
        elif initializer == "normal":
            return tf.keras.initializers.RandomNormal(
                mean=initializer_params.get("mean", 0.0),
                stddev=initializer_params.get("std", 0.05),
                seed=self.seed,
            )
        elif initializer == "constant":
            entity_init = initializer_params.get("entity", None)
            rel_init = initializer_params.get("relation", None)
            assert (
                entity_init is not None
            ), "Please pass the `entity` initializer value"
            assert (
                rel_init is not None
            ), "Please pass the `relation` initializer value"
            return [
                tf.constant_initializer(entity_init),
                tf.constant_initializer(rel_init),
            ]
        else:
            return tf.keras.initializers.get(initializer)

    def fit(
        self,
        X,
        early_stopping=False,
        early_stopping_params={},
        focusE_numeric_edge_values=None,
        tensorboard_logs_path=None,
        callbacks=None,
        verbose=False,
    ):
        """Train the model (with optional early stopping).

        The model is trained on a training set ``X`` using the training
        protocol described in :cite:`trouillon2016complex`.

        Parameters
        ----------
        X : ndarray, shape (n, 3) or str or GraphDataLoader or
            AbstractGraphPartitioner Data OR Filename of the data
            file OR Data Handle to be used for training.
        early_stopping: bool
            Flag to enable early stopping (default:`False`)
        early_stopping_params: dict
            Dictionary of hyperparameters for the early stopping heuristics.
            The following string keys are supported:

                - **"x_valid"** (ndarray, shape (n, 3) or str or
                GraphDataLoader or AbstractGraphPartitioner) - Numpy \
                array of validation triples OR handle of Dataset adapter
                which would help retrieve data.
                - **"criteria"** (str) - Criteria for early stopping
                `'hits10'`, `'hits3'`, `'hits1'` \
                or `'mrr'` (default: `'mrr'`).
                - **"x_filter"** (ndarray, shape (n, 3)) - Positive triples
                to use as filter if a `'filtered'` early \
                stopping criteria is desired (i.e., filtered-MRR if
                ``'criteria':'mrr'``). Note this will affect training time
                (no filter by default). If the filter has already been set in
                the adapter, pass `True`.
                - **"burn_in"** (int) - Number of epochs to pass before
                  kicking in early stopping (default: 100).
                - **"check_interval"** (int) - Early stopping interval after
                  burn-in (default:10).
                - **"stop_interval"** (int) - Stop if criteria is performing
                  worse over n consecutive checks (default: 3).
                - **"corruption_entities"** (list) - List of entities to be
                  used for corruptions. If `'all'`, it uses all entities
                  (default: `'all'`).
                - **"corrupt_side"** (str) - Specifies which side to corrupt:
                `'s'`, `'o'`, `'s+o'`, `'s,o'` \
                (default: `'s,o'`).

        focusE_numeric_edge_values: ndarray, shape (n, 1)
            Numeric values associated with links in the training set.
            Semantically, the numeric value can signify importance,
            uncertainity, significance, confidence, etc. If the numeric value
            is unknown pass a NaN weight. The model will uniformly randomly
            assign a numeric value. One can also think about assigning
            numeric values by looking at the distribution of it per predicate.
            .. warning:: In the compatible version, this option only supports
            data passed as np.array.
        tensorboard_logs_path: str or None
            Path to store tensorboard logs, e.g., average training loss
            tracking per epoch (default: `None` indicating no logs will be
            collected). When provided it will create a folder under provided
            path and save tensorboard files there. To then view the loss in
            the terminal run: ``tensorboard --logdir <tensorboard_logs_path>``.
        """
        self.model = ScoringBasedEmbeddingModel(
            self.eta, self.k, scoring_type=self.model_name, seed=self.seed
        )
        if callbacks is None:
            callbacks = []
        if tensorboard_logs_path is not None:
            tensorboard_callback = tf.keras.callbacks.TensorBoard(
                log_dir=tensorboard_logs_path
            )
            callbacks.append(tensorboard_callback)

        regularizer = self.regularizer
        if regularizer is not None:
            regularizer = get_regularizer(regularizer,
                                          self.regularizer_params)

        initializer = self.initializer
        if initializer is not None:
            initializer = self._get_initializer(
                initializer, self.initializer_params
            )

        loss = get_loss(self.loss, self.loss_params)
        optimizer, is_back_compat_optim = self._get_optimizer(
            self.optimizer, self.optimizer_params
        )

        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            entity_relation_initializer=initializer,
            entity_relation_regularizer=regularizer,
        )
        if not is_back_compat_optim:
            tf.keras.backend.set_value(
                self.model.optimizer.learning_rate,
                self.optimizer_params.get("lr", 0.001),
            )

        if len(early_stopping_params) != 0:
            checkpoint = tf.keras.callbacks.EarlyStopping(
                monitor="val_{}".format(
                    early_stopping_params.get("criteria", "mrr")
                ),
                min_delta=0,
                patience=early_stopping_params.get("stop_interval", 10),
                verbose=self.verbose,
                mode="max",
                restore_best_weights=True,
            )
            callbacks.append(checkpoint)

        x_filter = early_stopping_params.get("x_filter", None)

        if isinstance(x_filter, np.ndarray) or isinstance(x_filter, list):
            x_filter = {"test": x_filter}
        elif x_filter is None or not x_filter:
            x_filter = False
        elif isinstance(x_filter, dict):
            pass
        else:
            raise ValueError("Incorrect type for x_filter")

        focusE = False
        params_focusE = {}
        if focusE_numeric_edge_values is not None:
            if isinstance(
                focusE_numeric_edge_values, np.ndarray
            ) and isinstance(X, np.ndarray):
                focusE = True
                X = np.concatenate([X, focusE_numeric_edge_values], axis=1)
                params_focusE = {
                    "non_linearity": self.embedding_model_params.get(
                        "non_linearity", "linear"
                    ),
                    "stop_epoch": self.embedding_model_params.get(
                        "stop_epoch", 251
                    ),
                    "structural_wt": self.embedding_model_params.get(
                        "structural_wt", 0.001
                    ),
                }
            else:
                msg = (
                    "Either X or focusE_numeric_edge_values are not\
                    np.array, so focusE is not supported. "
                    "Try using Ampligraph 2 or Ampligraph 1.x APIs!"
                )
                raise ValueError(msg)

        self.model.fit(
            X,
            batch_size=np.ceil(X.shape[0] / self.batches_count),
            epochs=self.epochs,
            validation_freq=early_stopping_params.get("check_interval", 10),
            validation_burn_in=early_stopping_params.get("burn_in", 25),
            validation_batch_size=early_stopping_params.get("batch_size",
                                                            100),
            validation_data=early_stopping_params.get("x_valid", None),
            validation_filter=x_filter,
            validation_entities_subset=early_stopping_params.get(
                "corruption_entities", None
            ),
            callbacks=callbacks,
            verbose=verbose,
            focusE=focusE,
            focusE_params=params_focusE,
        )
        self.data_shape = self.model.data_shape

    def get_indexes(self, X, type_of="t", order="raw2ind"):
        """Converts given data to indexes or to raw data (according to order).

        It works for both triples (``type_of='t'``), entities
        (``type_of='e'``), and relations (``type_of='r'``).

        Parameters
        ----------
        X: np.array
            Data to be indexed.
        type_of: str
            One of `['e', 't', 'r']` to specify which type of data is to be
            indexed or converted to raw data.
        order: str
            One of `['raw2ind', 'ind2raw']` to specify whether to convert raw
            data to indexes or vice versa.

        Returns
        -------
        Y: np.array
            Indexed data or raw data.
        """
        return self.model.get_indexes(X, type_of, order)

    def get_count(self, concept_type="e"):
        """Returns the count of entities and relations that were present
        during training.

        Parameters
        ----------
        concept_type: str
            Indicates whether to count entities (``concept_type='e'``) or
            relations (``concept_type='r'``). Default: ``concept_type='e'``.

        Returns
        -------
        count: int
            Count of the entities or relations.
        """
        if concept_type == "entity" or concept_type == "e":
            return self.model.get_count("e")
        elif concept_type == "relation" or concept_type == "r":
            return self.model.get_count("r")
        else:
            raise ValueError("Invalid value for concept_type!")

    def get_embeddings(self, entities, embedding_type="entity"):
        """Get the embeddings of entities or relations.

        .. Note ::

            Use :meth:`ampligraph.utils.create_tensorboard_visualizations` to
            visualize the embeddings with TensorBoard.

        Parameters
        ----------
        entities : array-like, dtype=int, shape=[n]
            The entities (or relations) of interest. Elements of the vector
            must be the original string literals, and
            not internal IDs.
        embedding_type : str
            If `'e'` or `'entities'`, ``entities`` argument will be
            considered as a list of knowledge graph entities (i.e. nodes).
            If set to `'r'` or `'relation'`, they will be treated as relation
            types instead (i.e. predicates).

        Returns
        -------
        embeddings : ndarray, shape [n, k]
            An array of k-dimensional embeddings.
        """
        if embedding_type == "entity" or embedding_type == "e":
            return self.model.get_embeddings(entities, "e")
        elif embedding_type == "relation" or embedding_type == "r":
            return self.model.get_embeddings(entities, "r")
        else:
            raise ValueError("Invalid value for embedding_type!")

    def get_hyperparameter_dict(self):
        """Returns hyperparameters of the model.

        Returns
        -------
        hyperparam_dict : dict
            Dictionary of hyperparameters that were used for training.
        """
        ent_idx = np.arange(self.model.data_indexer.get_entities_count())
        rel_idx = np.arange(self.model.data_indexer.get_relations_count())
        ent_values_raw = self.model.data_indexer.get_indexes(
            ent_idx, "e", "ind2raw"
        )
        rel_values_raw = self.model.data_indexer.get_indexes(
            rel_idx, "r", "ind2raw"
        )
        return dict(zip(ent_values_raw, ent_idx)), dict(
            zip(rel_values_raw, rel_idx)
        )

    def predict(self, X):
        """
        Predict the scores of triples using a trained embedding model.

        The function returns raw scores generated by the model.

        Parameters
        ----------
        X : ndarray, shape (n, 3)
            The triples to score.

        Returns
        -------
        scores_predict : ndarray, shape (n)
            The predicted scores for input triples.
        """
        return self.model.predict(X)

    def calibrate(
        self,
        X_pos,
        X_neg=None,
        positive_base_rate=None,
        batches_count=100,
        epochs=50,
    ):
        """Calibrate predictions.

        The method implements the heuristics described in :cite:`calibration`,
        using Platt scaling :cite:`platt1999probabilistic`.

        The calibrated predictions can be obtained with :meth:`predict_proba`
        after calibration is done.

        Parameters
        ----------
        X_pos : ndarray, shape (n, 3)
            Numpy array of positive triples.
        X_neg : ndarray, shape (n, 3)
            Numpy array of negative triples.
            If `None`, the negative triples are generated via corruptions
            and the user must provide a positive base rate instead.
        positive_base_rate: float
            Base rate of positive statements.
            For example, if we assume there is a fifty-fifty chance of any
            query to be true, the base rate would be 50%. If ``X_neg`` is
            provided and this is `None`, the relative sizes of ``X_pos``
            and ``X_neg`` will be used to determine the base rate.
            For example, if we have 50 positive triples and 200 negative
            triples, the positive base rate will be assumed to be
            50/(50+200) = 1/5 = 0.2. This must be a value between 0 and 1.
        batches_count: int
            Number of batches to complete one epoch of the Platt
            scaling training. Only applies when ``X_neg`` is  `None`.
        epochs: int
            Number of epochs used to train the Platt scaling model.
            Only applies when ``X_neg`` is  `None`.

        """
        batch_size = int(np.ceil(X_pos.shape[0] / batches_count))
        return self.model.calibrate(
            X_pos, X_neg, positive_base_rate, batch_size, epochs
        )

    def predict_proba(self, X):
        """
        Predicts probabilities using the Platt scaling model
        (after calibration).

        Model must be calibrated beforehand with the ``calibrate`` method.

        Parameters
        ----------
        X: ndarray, shape (n, 3)
            Numpy array of triples to be evaluated.

        Returns
        -------
        scores: np.array, shape (n, )
            Calibrated scores for the input triples.
        """
        return self.model.predict_proba(X)

    def evaluate(
        self,
        x=None,
        batch_size=32,
        verbose=True,
        use_filter=False,
        corrupt_side="s,o",
        entities_subset=None,
        callbacks=None,
    ):
        """
        Evaluate the inputs against corruptions and return ranks.

        Parameters
        ----------
        x: np.array, shape (n,3) or str or GraphDataLoader or
           AbstractGraphPartitioner Data OR Filename of the data file
           OR Data Handle to be used for training.
        batch_size: int
            Batch size to use during training.
            May be overridden if ``x`` is `GraphDataLoader` or
            `AbstractGraphPartitioner` instance
        verbose: bool
            Verbosity mode.
        use_filter: bool or dict
            Whether to use a filter of not. If a dictionary is specified, the
            data in the dict is concatenated and used as filter.
        corrupt_side: str
            Which side to corrupt of a triple to corrupt. It can be the
            subject (``corrupt_size="s"``), the object (``corrupt_size="o"``),
            the subject and the object (``corrupt_size="s+o"`` or
            ``corrupt_size="s,o"``) (default:`"s,o"`).
        entities_subset: list or np.array
            Subset of entities to be used for generating corruptions.
        callbacks: list of keras.callbacks.Callback instances
            List of callbacks to apply during evaluation.

        Returns
        -------
        rank: np.array, shape (n, number of corrupted sides)
            Ranking of test triples against subject corruptions and/or
            object corruptions.
        """

        return self.model.evaluate(
            x,
            batch_size=batch_size,
            verbose=verbose,
            use_filter=use_filter,
            corrupt_side=corrupt_side,
            entities_subset=entities_subset,
            callbacks=callbacks,
        )


@register_compatibility("TransE")
class TransE(ScoringModelBase):
    """Class wrapping around the ScoringBasedEmbeddingModel with the TransE
    scoring function."""

    def __init__(
        self,
        k=100,
        eta=2,
        epochs=100,
        batches_count=100,
        seed=0,
        embedding_model_params={
            "corrupt_sides": ["s,o"],
            "negative_corruption_entities": "all",
            "norm": 1,
            "normalize_ent_emb": False,
        },
        optimizer="adam",
        optimizer_params={"lr": 0.0005},
        loss="nll",
        loss_params={},
        regularizer=None,
        regularizer_params={},
        initializer="xavier",
        initializer_params={"uniform": False},
        verbose=False,
        model=None,
    ):
        """Initialize the ScoringBasedEmbeddingModel with the TransE
        scoring function."""
        super().__init__(
            k,
            eta,
            epochs,
            batches_count,
            seed,
            embedding_model_params,
            optimizer,
            optimizer_params,
            loss,
            loss_params,
            regularizer,
            regularizer_params,
            initializer,
            initializer_params,
            verbose,
            model,
        )

        self.model_name = "TransE"


@register_compatibility("DistMult")
class DistMult(ScoringModelBase):
    """Class wrapping around the ScoringBasedEmbeddingModel with the DistMult
    scoring function."""

    def __init__(
        self,
        k=100,
        eta=2,
        epochs=100,
        batches_count=100,
        seed=0,
        embedding_model_params={
            "corrupt_sides": ["s,o"],
            "negative_corruption_entities": "all",
            "norm": 1,
            "normalize_ent_emb": False,
        },
        optimizer="adam",
        optimizer_params={"lr": 0.0005},
        loss="nll",
        loss_params={},
        regularizer=None,
        regularizer_params={},
        initializer="xavier",
        initializer_params={"uniform": False},
        verbose=False,
        model=None,
    ):
        """Initialize the ScoringBasedEmbeddingModel with the DistMult
        scoring function."""
        super().__init__(
            k,
            eta,
            epochs,
            batches_count,
            seed,
            embedding_model_params,
            optimizer,
            optimizer_params,
            loss,
            loss_params,
            regularizer,
            regularizer_params,
            initializer,
            initializer_params,
            verbose,
            model,
        )

        self.model_name = "DistMult"


@register_compatibility("ComplEx")
class ComplEx(ScoringModelBase):
    """Class wrapping around the ScoringBasedEmbeddingModel with the ComplEx
    scoring function."""

    def __init__(
        self,
        k=100,
        eta=2,
        epochs=100,
        batches_count=100,
        seed=0,
        embedding_model_params={
            "corrupt_sides": ["s,o"],
            "negative_corruption_entities": "all",
            "norm": 1,
            "normalize_ent_emb": False,
        },
        optimizer="adam",
        optimizer_params={"lr": 0.0005},
        loss="nll",
        loss_params={},
        regularizer=None,
        regularizer_params={},
        initializer="xavier",
        initializer_params={"uniform": False},
        verbose=False,
        model=None,
    ):
        """Initialize the ScoringBasedEmbeddingModel with the ComplEx
        scoring function."""
        super().__init__(
            k,
            eta,
            epochs,
            batches_count,
            seed,
            embedding_model_params,
            optimizer,
            optimizer_params,
            loss,
            loss_params,
            regularizer,
            regularizer_params,
            initializer,
            initializer_params,
            verbose,
            model,
        )

        self.model_name = "ComplEx"


@register_compatibility("HolE")
class HolE(ScoringModelBase):
    """Class wrapping around the ScoringBasedEmbeddingModel with the HolE
    scoring function."""

    def __init__(
        self,
        k=100,
        eta=2,
        epochs=100,
        batches_count=100,
        seed=0,
        embedding_model_params={
            "corrupt_sides": ["s,o"],
            "negative_corruption_entities": "all",
            "norm": 1,
            "normalize_ent_emb": False,
        },
        optimizer="adam",
        optimizer_params={"lr": 0.0005},
        loss="nll",
        loss_params={},
        regularizer=None,
        regularizer_params={},
        initializer="xavier",
        initializer_params={"uniform": False},
        verbose=False,
        model=None,
    ):
        """Initialize the ScoringBasedEmbeddingModel with the HolE
        scoring function."""
        super().__init__(
            k,
            eta,
            epochs,
            batches_count,
            seed,
            embedding_model_params,
            optimizer,
            optimizer_params,
            loss,
            loss_params,
            regularizer,
            regularizer_params,
            initializer,
            initializer_params,
            verbose,
            model,
        )

        self.model_name = "HolE"
