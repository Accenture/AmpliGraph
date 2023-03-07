# Copyright 2019-2023 The AmpliGraph Authors. All Rights Reserved.
#
# This file is Licensed under the Apache License, Version 2.0.
# A copy of the Licence is available in LICENCE, or at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
import tensorflow as tf
import copy
import shelve
import pickle
import numpy as np
import os
import tempfile
import logging

from ampligraph.evaluation.metrics import mrr_score, hits_at_n_score, mr_score
from ampligraph.datasets import data_adapter
from ampligraph.datasets.partitioned_data_manager import PartitionDataManager
from ampligraph.latent_features.layers.scoring.AbstractScoringLayer import (
    SCORING_LAYER_REGISTRY,
)
from ampligraph.latent_features.layers.encoding import EmbeddingLookupLayer
from ampligraph.latent_features.layers.calibration import CalibrationLayer
from ampligraph.latent_features.layers.corruption_generation import (
    CorruptionGenerationLayerTrain,
)
from ampligraph.datasets.data_indexer import DataIndexer
from ampligraph.latent_features import optimizers
from ampligraph.latent_features import loss_functions
from ampligraph.evaluation import train_test_split_no_unseen
from tensorflow.python.keras import callbacks as callbacks_module
from tensorflow.python.keras.engine import training_utils
from tensorflow.python.eager import def_function
from tensorflow.python.keras import metrics as metrics_mod
from tensorflow.python.keras.engine import compile_utils

tf.config.set_soft_device_placement(False)
tf.debugging.set_log_device_placement(False)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class ScoringBasedEmbeddingModel(tf.keras.Model):
    """Class for handling KGE models which follows the ranking based protocol.

    Example
    -------
    >>> # create model and compile using user defined optimizer settings and
    >>> # user defined settings of an existing loss
    >>> from ampligraph.datasets import load_fb15k_237
    >>> from ampligraph.latent_features import ScoringBasedEmbeddingModel
    >>> from ampligraph.latent_features.loss_functions import SelfAdversarialLoss
    >>> import tensorflow as tf
    >>> X = load_fb15k_237()
    >>> loss = SelfAdversarialLoss({'margin': 0.1, 'alpha': 5, 'reduction': 'sum'})
    >>> model = ScoringBasedEmbeddingModel(eta=5,
    >>>                                    k=300,
    >>>                                    scoring_type='ComplEx',
    >>>                                    seed=0)
    >>> model.compile(optimizer='adam', loss=loss)
    >>> model.fit(X['train'],
    >>>           batch_size=10000,
    >>>           epochs=5)
    Epoch 1/5
    29/29 [==============================] - 3s 87ms/step - loss: 13496.5752
    Epoch 2/5
    29/29 [==============================] - 1s 36ms/step - loss: 13488.8682
    Epoch 3/5
    29/29 [==============================] - 1s 35ms/step - loss: 13436.2725
    Epoch 4/5
    29/29 [==============================] - 1s 35ms/step - loss: 13259.0840
    Epoch 5/5
    29/29 [==============================] - 1s 34ms/step - loss: 12977.0117
    """

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def get_config(self):
        """Get the configuration hyper-parameters of the scoring based embedding model."""
        config = super(ScoringBasedEmbeddingModel, self).get_config()
        config.update(
            {
                "eta": self.eta,
                "k": self.k,
                "scoring_type": self.scoring_type,
                "seed": self.seed,
                "max_ent_size": self.encoding_layer._max_ent_size_internal,
                "max_rel_size": self.encoding_layer._max_rel_size_internal,
            }
        )

        return config

    def __init__(
        self,
        eta,
        k,
        scoring_type="DistMult",
        seed=0,
        max_ent_size=None,
        max_rel_size=None,
    ):
        """
        Initializes the scoring based embedding model using the user specified scoring function.

        Parameters
        ----------
        eta: int
            Num of negatives to use during training per triple.
        k: int
            Embedding size.
        scoring_type: str
            Name of the scoring layer to use.

            - ``TransE``  Translating embedding scoring function will be used
            - ``DistMult`` DistMult embedding scoring function will be used
            - ``ComplEx`` ComplEx embedding scoring function will be used
            - ``HolE`` Holograph embedding scoring function will be used

        seed: int
            Random seed.
        max_ent_size: int
            Maximum number of entities that can occur in any partition (default: `None`).
        max_rel_size: int
            Maximum number of relations that can occur in any partition (default: `None`).
        """
        super(ScoringBasedEmbeddingModel, self).__init__()
        # set the random seed
        tf.random.set_seed(seed)
        np.random.seed(seed)

        self.max_ent_size = max_ent_size
        self.max_rel_size = max_rel_size

        self.eta = eta
        self.scoring_type = scoring_type

        # get the scoring layer
        self.scoring_layer = SCORING_LAYER_REGISTRY[scoring_type](k)
        # get the actual k depending on scoring layer
        # Ex: complex model uses k embeddings for real and k for img side.
        # so internally it has 2*k, whereas transE uses k.
        self.k = k
        self.internal_k = self.scoring_layer.internal_k

        # create the corruption generation layer - generates eta corruptions
        # during training
        self.corruption_layer = CorruptionGenerationLayerTrain()

        # If it is single partition, assume that you have max_ent_size unique entities
        # This would change if we use partitions: based on which partition is in memory
        # this attribute is used by the corruption_layer to sample eta
        # corruptions
        self.num_ents = self.max_ent_size

        # Create the embedding lookup layer.
        # size of entity emb is max_ent_size * k and relation emb is
        # max_rel_size * k
        self.encoding_layer = EmbeddingLookupLayer(
            self.internal_k, self.max_ent_size, self.max_rel_size
        )

        # Flag to indicate whether the partitioned training is being done
        self.is_partitioned_training = False
        # Variable related to data indexing (entity to idx mapping)
        self.data_indexer = True

        # Flag to indicate whether to include FocusE layer
        self.use_focusE = False
        self.focusE_params = {}

        self.is_calibrated = False
        self.is_fitted = False
        self.is_backward = False

        self.seed = seed
        self.base_dir = tempfile.gettempdir()
        self.partitioner_metadata = {}

    def is_fit(self):
        """Check whether the model has been fitted already."""
        return self.is_fitted

    def compute_output_shape(self, inputShape):
        """Returns the output shape of the outputs of the call function.

        Parameters
        ----------
        input_shape: tuple
            Shape of inputs of call function.

        Returns
        -------
        output_shape: list of tuples
            List with the shape of outputs of call function for the input triples and the corruption scores.
        """
        # input triple score (batch_size, 1) and corruption score (batch_size *
        # eta, 1)
        return [(None, 1), (None, 1)]

    def partition_change_updates(self, num_ents, ent_emb, rel_emb):
        """Perform the changes that are required when the partition is modified during training.

        Parameters
        ----------
        num_ents: int
            Number of unique entities in the partition.
        ent_emb: array-like
            Entity embeddings that need to be trained for the partition
            (all triples of the partition will have embeddings in this matrix).
        rel_emb: array-like
            relation embeddings that need to be trained for the partition
            (all triples of the partition will have embeddings in this matrix).

        """
        # save the unique entities of the partition : will be used for
        # corruption generation
        self.num_ents = num_ents
        if self.encoding_layer.built:
            # update the trainable variable in the encoding layer
            self.encoding_layer.partition_change_updates(ent_emb, rel_emb)
        else:
            # if the encoding layer has not been built then store it as an initializer
            # this would be the case of during partitioned training (first
            # batch)
            self.encoding_layer.set_ent_rel_initial_value(ent_emb, rel_emb)

    def call(self, inputs, training=False):
        """
        Computes the scores of the triples and returns the corruption scores as well.

        Parameters
        ----------
        inputs: ndarray, shape (n, 3)
            Batch of input triples.

        Returns
        -------
        out: list
            List of input scores along with their corruptions.
        """
        # lookup embeddings of the inputs
        inp_emb = self.encoding_layer(inputs)
        # score the inputs
        inp_score = self.scoring_layer(inp_emb)
        # score the corruptions

        if training:
            # generate the corruptions for the input triples
            corruptions = self.corruption_layer(
                inputs, self.num_ents, self.eta
            )
            # lookup embeddings of the inputs
            corr_emb = self.encoding_layer(corruptions)
            corr_score = self.scoring_layer(corr_emb)

            return inp_score, corr_score

        else:
            return inp_score

    @tf.function(experimental_relax_shapes=True)
    def _get_ranks(
        self,
        inputs,
        ent_embs,
        start_id,
        end_id,
        filters,
        mapping_dict,
        corrupt_side="s,o",
        ranking_strategy="worst",
    ):
        """
        Evaluate the inputs against corruptions and return ranks.

        Parameters
        ----------
        inputs: array-like, shape (n, 3)
            Batch of input triples.
        ent_embs: array-like, shape (m, k)
            Slice of embedding matrix (corruptions).
        start_id: int
            Original id of the first row of embedding matrix (used during partitioned approach).
        end_id: int
            Original id of the last row of embedding matrix (used during partitioned approach).
        filters: list of lists
            Size of list is either 1 or 2 depending on ``corrupt_side``.
            Size of the internal list is equal to the size of the input triples.
            Each list contains an array of filters (i.e., True Positives) related to the specified side of the
            corresponding input triples.
        corrupt_side: str
            Which side to corrupt during evaluation.
        ranking_strategy: str
            Indicates how to break ties (default: `worst`, i.e., assigns the worst rank to the test triple).
            Can be one of the three types `"best"`, `"middle"`, `"worst"`.

        Returns
        -------
        rank: tf.Tensor, shape (n, num of sides being corrupted)
            Ranking against subject corruptions and object corruptions
            (corruptions defined by `ent_embs` matrix).
        """
        if not self.is_partitioned_training:
            inputs = [
                tf.nn.embedding_lookup(
                    self.encoding_layer.ent_emb, inputs[:, 0]
                ),
                tf.nn.embedding_lookup(
                    self.encoding_layer.rel_emb, inputs[:, 1]
                ),
                tf.nn.embedding_lookup(
                    self.encoding_layer.ent_emb, inputs[:, 2]
                ),
            ]

        return self.scoring_layer.get_ranks(
            inputs,
            ent_embs,
            start_id,
            end_id,
            filters,
            mapping_dict,
            corrupt_side,
            ranking_strategy,
        )

    def build(self, input_shape):
        """Override the build function of the Model class.

        It is called on the first call to ``__call__``.
        With this function we set some internal parameters of the encoding layers (needed to build that layers
        themselves) based on the input data supplied by the user while calling the `~ScoringBasedEmbeddingModel.fit` method.
        """
        # set the max number of the entities that will be trained per partition
        # in case of non-partitioned training, it is equal to the total number
        # of entities of the dataset
        self.encoding_layer.max_ent_size = self.max_ent_size
        # set the max number of relations being trained just like above
        self.encoding_layer.max_rel_size = self.max_rel_size
        self.num_ents = self.max_ent_size
        self.built = True

    def compute_focusE_weights(self, weights, structure_weight):
        """Compute positive and negative weights to scale scores if ``use_focusE=True``.

        Parameters
        ----------
        weights: array-like, shape (n, m)
            Batch of weights associated triples.
        strucuture_weight: float
            Structural influence assigned to the weights.

        Returns
        -------
        out: tuple of two tf.Tensors, (tf.Tensor(shape=(n, 1)), tf.Tensor(shape=(n * self.eta, 1)))
            Tuple where the first elements is a tensor containing the positive weights
            and the second is a tensor containing the negative weights.
        """

        # Weights computation
        weights = tf.reduce_mean(weights, 1)
        weights_pos = structure_weight + (1 - structure_weight) * (1 - weights)
        weights_neg = structure_weight + (1 - structure_weight) * (
            tf.reshape(
                tf.tile(weights, [self.eta]), [tf.shape(weights)[0] * self.eta]
            )
        )

        return weights_pos, weights_neg

    def train_step(self, data):
        """
        Training step.

        Parameters
        ----------
        data: array-like, shape (n, m)
            Batch of input triples (true positives) with weights associated if m>3.

        Returns
        -------
        out: dict
            Dictionary of metrics computed on the outputs (e.g., loss).
        """
        if self.data_shape > 3:
            triples = data[0]
            if self.data_handler._adapter.use_filter:
                weights = data[2]
            else:
                weights = data[1]
        else:
            triples = data
        with tf.GradientTape() as tape:
            # get the model predictions
            score_pos, score_neg = self(tf.cast(triples, tf.int32), training=1)
            # focusE layer
            if self.use_focusE:
                logger.debug("Using FocusE")
                non_linearity = self.focusE_params["non_linearity"]
                structure_weight = self.focusE_params["structural_wt"]

                weights_pos, weights_neg = self.compute_focusE_weights(
                    weights=weights, structure_weight=structure_weight
                )
                # Computation of scores
                score_pos = non_linearity(score_pos) * weights_pos
                score_neg = non_linearity(score_neg) * weights_neg

            # compute the loss
            loss = self.compiled_loss(
                score_pos,
                score_neg,
                self.eta,
                regularization_losses=self.losses,
            )
        try:
            # minimize the loss and update the trainable variables
            self.optimizer.minimize(
                loss,
                self.encoding_layer.ent_emb,
                self.encoding_layer.rel_emb,
                tape,
            )
        except ValueError as e:
            if self.scoring_layer.name == "Random":
                pass
            else:
                raise e

        return {m.name: m.result() for m in self.metrics}

    def make_train_function(self):
        """Similar to keras lib, this function returns the handle to the training step function.
        It processes one batch of data by iterating over the dataset iterator, it computes the loss and optimizes on it.

        Returns
        -------
        out: Function handle
              Handle to the training step function.
        """
        if self.train_function is not None:
            return self.train_function

        def train_function(iterator):
            """This is the function whose handle will be returned.

            Parameters
            ----------
            iterator: tf.data.Iterator
                Data iterator.

            Returns
            -------
            output: dict
              Return a dictionary containing values that will be passed to ``tf.keras.Callbacks.on_train_batch_end``.
            """
            data = next(iterator)
            output = self.train_step(data)
            return output

        if not self.run_eagerly and not self.is_partitioned_training:
            train_function = def_function.function(
                train_function, experimental_relax_shapes=True
            )

        self.train_function = train_function
        return self.train_function

    def get_focusE_params(self, dict_params={}):
        """Get parameters for focusE.

        Parameters
        ----------
        dict_params: dict
            The following hyper-params can be passed:

            - "non_linearity": can assume of the following values `"linear"`, `"softplus"`, `"sigmoid"`, `"tanh"`.
            - "stop_epoch": specifies how long to decay (linearly) the structural influence hyper-parameter \
            from 1 until it reaches its original value.
            - "structural_wt": structural influence hyperparameter [0, 1] that modulates the influence of graph \
            topology.

            If the respective key is missing: ``non_linearity="linear"``, ``stop_epoch=251`` and ``structural_wt=0.001``.

        Returns
        -------
        focusE_params : tuple
            A tuple containing three values: the non-linearity function (`str`), the `stop_epoch` (`int`) and the
            structure weight (`float`).

        """
        # Get the non-linearity function
        non_linearity = dict_params.get("non_linearity", "linear")
        if non_linearity == "linear":
            non_linearity = tf.identity
        elif non_linearity == "tanh":
            non_linearity = tf.tanh
        elif non_linearity == "sigmoid":
            non_linearity = tf.sigmoid
        elif non_linearity == "softplus":

            def non_linearity(x):
                return tf.math.log(1 + 9999 * tf.exp(x))

        else:
            raise ValueError("Invalid focusE non-linearity")

        # Get the stop_epoch for the decay
        stop_epoch = dict_params.get("stop_epoch", 251)
        msg = "Invalid value for focusE stop_epoch: expected a value >=0 but got {}".format(
            stop_epoch
        )
        assert stop_epoch >= 0, msg

        # Get structural_wt
        structure_weight = dict_params.get("structural_wt", 0.001)
        assert (structure_weight <= 1) and (
            structure_weight >= 0
        ), "Invalid focusE 'structural_wt' passed! It has to belong to [0,1]."

        # if stop_epoch == 0, fixed structure weights is used
        if stop_epoch > 0:
            # linear decay of numeric values
            structure_weight = tf.maximum(
                1 - self.current_epoch / stop_epoch, 0.001
            )

        return non_linearity, stop_epoch, structure_weight

    def update_focusE_params(self):
        """Update the structural weight after decay."""
        if self.focusE_params["stop_epoch"] > 0:
            stop_epoch = self.focusE_params["stop_epoch"]
            self.focusE_params["structural_wt"] = tf.maximum(
                1 - self.current_epoch / stop_epoch, 0.001
            )

    def fit(
        self,
        x=None,
        batch_size=1,
        epochs=1,
        verbose=True,
        callbacks=None,
        validation_split=0.0,
        validation_data=None,
        shuffle=True,
        initial_epoch=0,
        validation_batch_size=100,
        validation_corrupt_side="s,o",
        validation_freq=50,
        validation_burn_in=100,
        validation_filter=False,
        validation_entities_subset=None,
        partitioning_k=1,
        focusE=False,
        focusE_params={},
    ):
        """Fit the model on the provided data.

        Parameters
        ----------
        x: np.array, shape (n, 3), or str or GraphDataLoader or AbstractGraphPartitioner
            Data OR Filename of the data file OR Data Handle to be used for training.
        batch_size: int
            Batch size to use during training.
            May be overridden if **x** is a GraphDataLoader or AbstractGraphPartitioner instance.
        epochs: int
            Number of epochs to train (default: 1).
        verbose: bool
            Verbosity (default: `True`).
        callbacks: list of tf.keras.callbacks.Callback
            List of callbacks to be used during training (default: `None`).
        validation_split: float
            Validation split to carve out of **x** (default: 0.0) (currently supported only when **x** is a np.array).
        validation_data: np.array, shape (n, 3) or str or `GraphDataLoader` or `AbstractGraphPartitioner`
            Data OR Filename of the data file OR Data Handle to be used for validation.
        shuffle: bool
            Indicates whether to shuffle the data after every epoch during training (default: `True`).
        initial epoch: int
            Initial epoch number (default: 1).
        validation_batch_size: int
            Batch size to use during validation (default: 100).
            May be overridden if ``validation_data`` is `GraphDataLoader` or `AbstractGraphPartitioner` instance.
        validation_freq: int
            Indicates how often to validate (default: 50).
        validation_burn_in: int
            The burn-in time after which the validation kicks in.
        validation_filter: bool or dict
            Validation filter to be used.
        validation_entities_subset: list or np.array
            Subset of entities to be used for generating corruptions.

            .. Note ::

                One can perform early stopping using the tensorflow callback ``tf.keras.callbacks.EarlyStopping``
                as shown in the accompanying example below.

        focusE: bool
            Specify whether to include the FocusE layer (default: `False`).
            The FocusE layer :cite:`pai2021learning` allows to inject numeric edge attributes into the scoring layer
            of a traditional knowledge graph embedding architecture.
            Semantically, the numeric value can signify importance, uncertainity, significance, confidence...
            of a triple.

            .. Note ::

                In order to activate focusE, the training data must have shape (n, 4), where the first three columns
                store subject, predicate and object of triples, and the 4-th column stores the numerical edge value
                associated with each triple.

        focusE_params: dict
            If FocusE layer is included, specify its hyper-parameters.
            The following hyper-params can be passed:

            + `"non_linearity"`: can be one of the following values `"linear"`, `"softplus"`, `"sigmoid"`, `"tanh"`.
            + `"stop_epoch"`: specifies how long to decay (linearly) the numeric values from 1 to original value.
            + `"structural_wt"`: structural influence hyperparameter :math:`\\in [0, 1]` that modulates the influence of graph topology.

            If ``focusE==True`` and ``focusE_params==dict()``, then the default values are passed:
            ``non_linearity="linear"``, ``stop_epoch=251`` and ``structural_wt=0.001``.

        partitioning_k: int
            Num of partitions to use while training (default: 1, i.e., the data is not partitioned).
            May be overridden if ``x`` is an `AbstractGraphPartitioner` instance.

            .. Note ::

                This function is quite useful when the size of your dataset is extremely large and cannot fit in memory.
                Setting this to a number strictly larger than 1 will automatically partition the data using
                ``BucketGraphPartitioner``.
                Kindly checkout the tutorials for usage in Advanced mode.

        Returns
        -------
        history: History object
            Its `History.history` attribute is a record of training loss values, as well as validation loss
            and validation metrics values.

        Example
        -------
        >>> from ampligraph.datasets import load_fb15k_237
        >>> from ampligraph.latent_features import ScoringBasedEmbeddingModel
        >>> X = load_fb15k_237()
        >>> model = ScoringBasedEmbeddingModel(eta=5,
        >>>                                    k=300,
        >>>                                    scoring_type='ComplEx',
        >>>                                    seed=0)
        >>> model.compile(optimizer='adam', loss='nll')
        >>> model.fit(X['train'],
        >>>           batch_size=10000,
        >>>           epochs=5)
        Epoch 1/5
        29/29 [==============================] - 2s 71ms/step - loss: 67361.3047
        Epoch 2/5
        29/29 [==============================] - 1s 35ms/step - loss: 67318.6094
        Epoch 3/5
        29/29 [==============================] - 1s 37ms/step - loss: 67020.0703
        Epoch 4/5
        29/29 [==============================] - 1s 35ms/step - loss: 65867.3750
        Epoch 5/5
        29/29 [==============================] - 1s 35ms/step - loss: 63517.9062

        >>> # Early stopping example
        >>> from ampligraph.latent_features import ScoringBasedEmbeddingModel
        >>> from ampligraph.datasets import load_fb15k_237
        >>> dataset = load_fb15k_237()
        >>> model = ScoringBasedEmbeddingModel(eta=1,
        >>>                                    k=10,
        >>>                                    scoring_type='TransE')
        >>> model.compile(optimizer='adam', loss='multiclass_nll')
        >>> import tensorflow as tf
        >>> early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_mrr",            # which metrics to monitor
        >>>                                               patience=3,                   # If the monitored metric doesnt improve for these many checks the model early stops
        >>>                                               verbose=1,                    # verbosity
        >>>                                               mode="max",                   # how to compare the monitored metrics; "max" means higher is better
        >>>                                               restore_best_weights=True)    # restore the weights with best value
        >>> # the early stopping instance needs to be passed as callback to fit function
        >>> model.fit(dataset['train'],
        >>>           batch_size=10000,
        >>>           epochs=5,
        >>>           validation_freq=1,                       # validation frequency
        >>>           validation_batch_size=100,               # validation batch size
        >>>           validation_burn_in=3,                    # burn in time
        >>>           validation_corrupt_side='s,o',           # which side to corrupt
        >>>           validation_data=dataset['valid'][::100], # Validation data
        >>>           callbacks=[early_stop])                  # Pass the early stopping object as a callback
        Epoch 1/5
        29/29 [==============================] - 2s 82ms/step - loss: 6698.2188
        Epoch 2/5
        29/29 [==============================] - 1s 34ms/step - loss: 6648.8862
        Epoch 3/5
        3/3 [==============================] - 1s 446ms/steposs: 6652.895
        29/29 [==============================] - 2s 84ms/step - loss: 6590.2842 - val_mrr: 0.0811 -
        val_mr: 1776.4545 - val_hits@1: 0.0000e+00 - val_hits@10: 0.2301 - val_hits@100: 0.4148
        Epoch 4/5
        3/3 [==============================] - 0s 102ms/steposs: 6564.021
        29/29 [==============================] - 1s 47ms/step - loss: 6517.4517 - val_mrr: 0.0918 -
        val_mr: 1316.6335 - val_hits@1: 0.0000e+00 - val_hits@10: 0.2528 - val_hits@100: 0.4716
        Epoch 5/5
        3/3 [==============================] - 1s 177ms/steposs: 6468.798
        29/29 [==============================] - 2s 62ms/step - loss: 6431.8696 - val_mrr: 0.0901 -
        val_mr: 1074.8920 - val_hits@1: 0.0000e+00 - val_hits@10: 0.2386 - val_hits@100: 0.4773

        """
        # verifies if compile has been called before calling fit
        self._assert_compile_was_called()
        # focusE
        self.current_epoch = 0
        self.use_focusE = focusE

        # use train test unseen to split training set
        if validation_split:
            assert isinstance(
                x, np.ndarray
            ), "Validation split supported for numpy arrays only!"
            x, validation_data = train_test_split_no_unseen(
                x,
                test_size=validation_split,
                seed=self.seed,
                allow_duplication=False,
            )

        with training_utils.RespectCompiledTrainableState(self):
            # create data handler for the data
            self.data_handler = data_adapter.DataHandler(
                x,
                model=self,
                batch_size=batch_size,
                dataset_type="train",
                epochs=epochs,
                initial_epoch=initial_epoch,
                use_filter=False,
                # if model is already
                # trained use the old
                # indexer
                use_indexer=self.data_indexer,
                partitioning_k=partitioning_k,
            )

            self.partitioner_metadata = (
                self.data_handler.get_update_partitioner_metadata(
                    self.base_dir
                )
            )
            # get the mapping details
            self.data_indexer = self.data_handler.get_mapper()
            # get the maximum entities and relations that will be trained
            # (useful during partitioning)
            self.max_ent_size = self.data_handler._adapter.max_entities
            self.max_rel_size = self.data_handler._adapter.max_relations
            # Number of columns (i.e., only triples or also weights?)
            if isinstance(self.data_handler._adapter, PartitionDataManager):
                self.data_shape = (
                    self.data_handler._parent_adapter.backend.data_shape
                )
            else:
                self.data_shape = self.data_handler._adapter.backend.data_shape

            # FocusE
            if self.data_shape < 4:
                self.use_focusE = False
            else:
                if self.use_focusE:
                    assert isinstance(
                        focusE_params, dict
                    ), "focusE parameters need to be in a dict!"
                    # Define FocusE params
                    (
                        non_linearity,
                        stop_epoch,
                        structure_weight,
                    ) = self.get_focusE_params(focusE_params)
                    self.focusE_params = {
                        "non_linearity": non_linearity,
                        "stop_epoch": stop_epoch,
                        "structural_wt": structure_weight,
                    }
                else:
                    print(
                        "Data shape is {}: not only triples were given, but focusE is not active!".format(
                            self.data_shape
                        )
                    )

            # Container that configures and calls `tf.keras.Callback`s.
            if not isinstance(callbacks, callbacks_module.CallbackList):
                callbacks = callbacks_module.CallbackList(
                    callbacks,
                    add_history=True,
                    add_progbar=verbose != 0,
                    model=self,
                    verbose=verbose,
                    epochs=epochs,
                )

            # This variable is used by callbacks to stop training in case of
            # any error
            self.stop_training = False
            self.is_partitioned_training = self.data_handler.using_partitioning
            self.optimizer.set_partitioned_training(
                self.is_partitioned_training
            )

            # set some partition related params if it is partitioned training
            if self.is_partitioned_training:
                self.partitioner_k = self.data_handler._adapter.partitioner_k
                self.encoding_layer.max_ent_size = self.max_ent_size
                self.encoding_layer.max_rel_size = self.max_rel_size

            # make the train function that will be used to process each batch
            # of data
            train_function = self.make_train_function()
            # before training begins call this callback function
            callbacks.on_train_begin()

            if (
                isinstance(validation_entities_subset, str)
                and validation_entities_subset == "all"
            ):
                # if the subset is set to none, it will use all entities in the
                # graph for generating corruptions
                validation_entities_subset = None

            # enumerate over the data
            for epoch, iterator in self.data_handler.enumerate_epochs():
                # current epoch number
                self.current_epoch = epoch
                # before epoch begins call this callback function
                callbacks.on_epoch_begin(epoch)
                # Update focusE parameter
                if self.use_focusE:
                    self.update_focusE_params()
                # handle the stop iteration of data iterator in this scope
                with self.data_handler.catch_stop_iteration():
                    # iterate over the dataset
                    for step in self.data_handler.steps():
                        # before a batch is processed call this callback
                        # function
                        callbacks.on_train_batch_begin(step)

                        # process this batch
                        logs = train_function(iterator)
                        # after a batch is processed call this callback
                        # function
                        callbacks.on_train_batch_end(step, logs)

                # store the logs of the last batch of the epoch
                epoch_logs = copy.copy(logs)
                # if validation is enabled
                if (
                    epoch >= (validation_burn_in - 1)
                    and validation_data is not None
                    and self._should_eval(epoch, validation_freq)
                ):
                    if self.data_shape > 3 and validation_data.shape[1] == 3:
                        nan_weights = np.empty(validation_data.shape[0])
                        nan_weights.fill(np.nan)
                        validation_data = np.concatenate(
                            [validation_data, nan_weights], axis=1
                        )
                    # evaluate on the validation
                    ranks = self.evaluate(
                        validation_data,
                        batch_size=validation_batch_size or batch_size,
                        use_filter=validation_filter,
                        dataset_type="valid",
                        corrupt_side=validation_corrupt_side,
                        entities_subset=validation_entities_subset,
                    )
                    # compute all the metrics
                    val_logs = {
                        "val_mrr": mrr_score(ranks),
                        "val_mr": mr_score(ranks),
                        "val_hits@1": hits_at_n_score(ranks, 1),
                        "val_hits@10": hits_at_n_score(ranks, 10),
                        "val_hits@100": hits_at_n_score(ranks, 100),
                    }
                    # update the epoch logs with validation details
                    epoch_logs.update(val_logs)

                # after an epoch is completed, call this callback function
                callbacks.on_epoch_end(epoch, epoch_logs)
                if self.stop_training:
                    break

            # on training end call this method
            callbacks.on_train_end()
            self.is_fitted = True
            # all the training and validation logs are stored in the history
            # object by keras.Model
            return self.history

    def get_indexes(self, X, type_of="t", order="raw2ind"):
        """Converts given data to indexes or to raw data (according to ``order``).

         It works for ``X`` containing triples, entities, or relations.

        Parameters
        ----------
        X: np.array or list
            Data to be indexed.
        type_of: str
            Specifies whether to get indexes/raw data for triples (``type_of='t'``), entities (``type_of='e'``),
            or relations (``type_of='r'``).
        order: str
            Specifies whether to get indexes from raw data (``order='raw2ind'``) or
            raw data from indexes (``order='ind2raw'``).

        Returns
        -------
        Y: np.array
            Indexed data or raw data.

        Example
        -------
        >>> from ampligraph.latent_features import ScoringBasedEmbeddingModel
        >>> from ampligraph.datasets import load_fb15k_237
        >>> X = load_fb15k_237()
        >>> model = ScoringBasedEmbeddingModel(eta=5,
        >>>                                    k=300,
        >>>                                    scoring_type='ComplEx',
        >>>                                    seed=0)
        >>> model.compile(optimizer='adam', loss='nll')
        >>> model.fit(X['train'],
        >>>           batch_size=10000,
        >>>           epochs=5,
        >>>           verbose=False)
        >>> print(model.get_indexes(['/m/027rn', '/m/06v8s0'], 'e', 'raw2ind'))
        >>> print(model.get_indexes([3877, 0], 'e', 'ind2raw'))
        [0, 3877]
        ['/m/06v8s0', '/m/027rn']
        """
        return self.data_indexer.get_indexes(X, type_of, order)

    def get_count(self, concept_type="e"):
        """Returns the count of entities and relations that were present during training.

        Parameters
        ----------
        concept_type: str
            Indicates whether to count entities (``concept_type='e'``) or
            relations (``concept_type='r'``) (default: `'e'`).

        Returns
        -------
        count: int
            Count of the entities or relations.

        Example
        -------
        >>> from ampligraph.datasets import load_fb15k_237
        >>> from ampligraph.latent_features import ScoringBasedEmbeddingModel
        >>> X = load_fb15k_237()
        >>> model = ScoringBasedEmbeddingModel(eta=5,
        >>>                                    k=300,
        >>>                                    scoring_type='ComplEx',
        >>>                                    seed=0)
        >>> model.compile(optimizer='adam', loss='nll')
        >>> model.fit(X['train'],
        >>>           batch_size=10000,
        >>>           epochs=5,
        >>>           verbose=False)
        >>> print('Entities:', model.get_count('e'))
        >>> print('Relations:', model.get_count('r'))
        Entities: 14505
        Relations: 237
        """
        assert self.is_fitted, "Model is not fit on the data yet!"
        if concept_type == "e":
            return self.data_indexer.get_entities_count()
        elif concept_type == "r":
            return self.data_indexer.get_relations_count()
        else:
            raise ValueError("Invalid Concept Type (expected 'e' or 'r')")

    def get_train_embedding_matrix_size(self):
        """Returns the size of the embedding matrix used for training.

        This may not be same as (n, k) during partitioned training (where `n` is the number of triples in the
        whole training set).
        """
        assert self.is_fitted, "Model is not fit on the data yet!"
        return {
            "e": self.encoding_layer.ent_emb.shape,
            "r": self.encoding_layer.rel_emb.shape,
        }

    def save(
        self,
        filepath,
        overwrite=True,
        include_optimizer=True,
        save_format=None,
        signatures=None,
        options=None,
        save_traces=True,
    ):
        """Save the model."""
        super(ScoringBasedEmbeddingModel, self).save(
            filepath,
            overwrite,
            include_optimizer,
            save_format,
            signatures,
            options,
            save_traces,
        )
        self.save_metadata(filedir=filepath)

    def save_metadata(self, filepath=None, filedir=None):
        """Save metadata."""
        # store ampligraph specific metadata
        if filepath is not None:
            base_dir = os.path.dirname(filedir)
            base_dir = "." if base_dir == "" else base_dir
            filepath = os.path.basename(filepath)

        if filedir is not None:
            base_dir = filedir
            filepath = os.path.basename(filedir)

        with open(
            os.path.join(base_dir, filepath + "_metadata.ampkl"), "wb"
        ) as f:
            metadata = {
                "is_partitioned_training": self.is_partitioned_training,
                "max_ent_size": self.max_ent_size,
                "max_rel_size": self.max_rel_size,
                "eta": self.eta,
                "k": self.k,
                "is_fitted": self.is_fitted,
                "is_calibrated": self.is_calibrated,
                "is_backward": self.is_backward,
            }

            metadata.update(self.data_indexer.get_update_metadata(base_dir))
            if self.is_partitioned_training:
                self.partitioner_metadata = (
                    self.data_handler.get_update_partitioner_metadata(base_dir)
                )
                metadata.update(self.partitioner_metadata)

            if self.is_calibrated:
                metadata["calib_w"] = self.calibration_layer.calib_w.numpy()
                metadata["calib_b"] = self.calibration_layer.calib_b.numpy()
                metadata["pos_size"] = self.calibration_layer.pos_size
                metadata["neg_size"] = self.calibration_layer.neg_size
                metadata[
                    "positive_base_rate"
                ] = self.calibration_layer.positive_base_rate
            pickle.dump(metadata, f)

    def save_weights(self, filepath, overwrite=True):
        """Save the trainable weights.

         Use this function if the training process is complete and you want to
         use the model only for inference. Use :meth:`load_weights` to load the model weights back.

         .. Note ::
             If you want to be able of continuing the training, you can use the :meth:`ampligraph.utils.save_model`
             and :meth:`ampligraph.utils.restore_model`.These functions save and restore the entire state
             of the graph, which allows to continue the training from where it was stopped.

        Parameters
        ----------
        filepath: str
            Path to save the model.
        overwrite: bool
            Flag which indicates whether the model, if present, needs to be overwritten or not (default: `True`).
        """
        # TODO: verify other formats

        # call the base class method to save the weights
        if not self.is_partitioned_training:
            super(ScoringBasedEmbeddingModel, self).save_weights(
                filepath, overwrite
            )
        self.save_metadata(filepath)

    def build_full_model(self, batch_size=100):
        """This method is called while loading the weights to build the model."""
        self.build((batch_size, 3))
        for i in range(len(self.layers)):
            self.layers[i].build((batch_size, 3))
            self.layers[i].built = True

    def load_metadata(self, filepath=None, filedir=None):
        if filedir is not None:
            filepath = os.path.join(filedir, os.path.basename(filedir))

        with open(filepath + "_metadata.ampkl", "rb") as f:
            metadata = pickle.load(f)
            metadata["root_directory"] = os.path.dirname(filepath)
            metadata["root_directory"] = (
                "."
                if metadata["root_directory"] == ""
                else metadata["root_directory"]
            )
            self.base_dir = metadata["root_directory"]
            try:
                metadata["db_file"] = os.path.basename(metadata["db_file"])
            except KeyError:
                print("Saved model does not include a db file. Skipping.")

            self.data_indexer = DataIndexer([], **metadata)
            self.is_partitioned_training = metadata["is_partitioned_training"]
            self.max_ent_size = metadata["max_ent_size"]
            self.max_rel_size = metadata["max_rel_size"]
            self.is_fitted = metadata["is_fitted"]
            self.is_backward = metadata.get("is_backward", False)
            if self.is_partitioned_training:
                self.partitioner_k = metadata["partitioner_k"]
                self.partitioner_metadata = {}
                self.partitioner_metadata["ent_map_fname"] = metadata[
                    "ent_map_fname"
                ]
                self.partitioner_metadata["rel_map_fname"] = metadata[
                    "rel_map_fname"
                ]

            self.is_calibrated = metadata["is_calibrated"]
            if self.is_calibrated:
                self.calibration_layer = CalibrationLayer(
                    metadata["pos_size"],
                    metadata["neg_size"],
                    metadata["positive_base_rate"],
                    calib_w=metadata["calib_w"],
                    calib_b=metadata["calib_b"],
                )

    def load_weights(self, filepath):
        """Loads the model weights.

         Use this function if ``save_weights`` was used to save the model.

         .. Note ::
             If you want to continue training, you can use the :meth:`ampligraph.utils.save_model` and
             :meth:`ampligraph.utils.load_model`. These functions save the entire state of the graph
             which allows to continue the training from where it stopped.

        Parameters
        ----------
        filepath: str
            Path to save the model.
        """
        self.load_metadata(filepath)
        self.build_full_model()
        if not self.is_partitioned_training:
            super(ScoringBasedEmbeddingModel, self).load_weights(filepath)

    def compile(
        self,
        optimizer="adam",
        loss=None,
        entity_relation_initializer="glorot_uniform",
        entity_relation_regularizer=None,
        **kwargs
    ):
        """ Compile the model.

        Parameters
        ----------
        optimizer: str (name of optimizer) or optimizer instance
            The optimizer used to minimize the loss function. For pre-defined options, choose between
            `"sgd"`, `"adagrad"`, `"adam"`, `"rmsprop"`, etc.
            See `tf.keras.optimizers <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers>`_
            for up-to-date details.

            If a string is passed, then the default parameters of the optimizer will be used.

            If you want to use custom hyperparameters you need to create an instance of the optimizer and
            pass the instance to the compile function ::

                import tensorflow as tf
                adam_opt = tf.keras.optimizers.Adam(learning_rate=0.003)
                model.compile(loss='pairwise', optim=adam_opt)

        loss: str (name of objective function), objective function or `ampligraph.latent_features.loss_functions.Loss`

            If a string is passed, you can use one of the following losses which will be used with their
            default setting:

            - `"pairwise"`:  the model will use the pairwise margin-based loss function.
            - `"nll"`: the model will use the negative loss likelihood.
            - `"absolute_margin"`: the model will use the absolute margin likelihood.
            - `"self_adversarial"`: the model will use the adversarial sampling loss function.
            - `"multiclass_nll"`: the model will use the multiclass nll loss. ::

                model.compile(loss='absolute_margin', optim='adam')

            If you want to modify the default parameters of the loss function, you need to explictly create an instance
            of the loss with required hyperparameters and then pass this instance. ::

                from ampligraph.latent_features import AbsoluteMarginLoss
                ab_loss = AbsoluteMarginLoss(loss_params={'margin': 3})
                model.compile(loss=ab_loss, optim='adam')

            An objective function is any callable with the signature
            ``loss = fn(score_true, score_corr, eta)`` ::

                # Create a user defined loss function with the above signature
                def userLoss(scores_pos, scores_neg):
                    # user defined loss - takes in 2 params and returns loss
                    neg_exp = tf.exp(scores_neg)
                    pos_exp = tf.exp(scores_pos)
                    # Apply softmax to the scores
                    score = pos_exp / (tf.reduce_sum(neg_exp, axis=0) + pos_exp)
                    loss = -tf.math.log(score)
                    return loss
                # Pass this loss while compiling the model
                model.compile(loss=userLoss, optim='adam')

        entity_relation_initializer: str (name of initializer function), initializer function or \
        `tf.keras.initializers.Initializer` or list.

            Initializer of the entity and relation embeddings. This is either a single value or a list of size 2.
            If a single value is passed, then both the entities and relations will be initialized based on
            the same initializer; if a list, the first initializer will be used for entities and the second
            for relations.

            If a string is passed, then the default parameters will be used. Choose between
            `"random_normal"`, `"random_uniform"`, `"glorot_normal"`, `"he_normal"`, etc.

            See `tf.keras.initializers <https://www.tensorflow.org/api_docs/python/tf/keras/initializers>`_
            for up-to-date details. ::

                model.compile(loss='pairwise', optim='adam',
                              entity_relation_initializer='random_normal')

            If the user wants to use custom hyperparameters, then an instance of the
            ``tf.keras.initializers.Initializer`` needs to be passed. ::

                import tensorflow as tf
                init = tf.keras.initializers.RandomNormal(stddev=0.00003)
                model.compile(loss='pairwise', optim='adam',
                              entity_relation_initializer=init)

            If the user wants to define custom initializer it can be any callable with the signature `init = fn(shape)` ::

                def my_init(shape):
                    return tf.random.normal(shape)
                model.compile(loss='pairwise', optim='adam',
                              entity_relation_initializer=my_init)

        entity_relation_regularizer: str (name of regularizer function) or regularizer function or \
        `tf.keras.regularizers.Regularizer` instance or list
            Regularizer of entities and relations.
            If a single value is passed, then both the entities and relations will be regularized based on
            the same regularizer; if a list, the first regularizer will be used for entities and second
            for relations.

            If a string is passed, then the default parameters of the regularizers will be used. Choose between
            `"l1"`, `"l2"`, `"l1_l2"`, etc.

            See `tf.keras.regularizers <https://www.tensorflow.org/api_docs/python/tf/keras/regularizers>`_
            for up-to-date details. ::

                model.compile(loss='pairwise', optim='adam',
                              entity_relation_regularizer='l2')

            If the user wants to use custom hyperparameters, then an instance of the
            ``tf.keras.regularizers.Regularizer`` needs to be passed. ::

                import tensorflow as tf
                reg = tf.keras.regularizers.L1L2(l1=0.001, l2=0.1)
                model.compile(loss='pairwise', optim='adam',
                              entity_relation_regularizer=reg)

            If the user wants to define custom regularizer it can be any callable with signature
            ``reg = fn(weight_matrix)``. ::

                def my_reg(weight_mx):
                      return 0.01 * tf.math.reduce_sum(tf.math.abs(weight_mx))
                model.compile(loss='pairwise', optim='adam',
                              entity_relation_regularizer=my_reg)

        Example
        -------
        >>> from ampligraph.datasets import load_fb15k_237
        >>> from ampligraph.latent_features import ScoringBasedEmbeddingModel
        >>> X = load_fb15k_237()
        >>> model = ScoringBasedEmbeddingModel(eta=5,
        >>>                                    k=300,
        >>>                                    scoring_type='ComplEx',
        >>>                                    seed=0)
        >>> model.compile(optimizer='adam', loss='nll')
        >>> model.fit(X['train'],
        >>>           batch_size=10000,
        >>>           epochs=5)
        Epoch 1/5
        29/29 [==============================] - 2s 61ms/step - loss: 67361.3047
        Epoch 2/5
        29/29 [==============================] - 1s 35ms/step - loss: 67318.6094
        Epoch 3/5
        29/29 [==============================] - 1s 34ms/step - loss: 67020.0703
        Epoch 4/5
        29/29 [==============================] - 1s 34ms/step - loss: 65867.3750
        Epoch 5/5
        29/29 [==============================] - 1s 34ms/step - loss: 63517.9062

        """
        # get the optimizer
        self.optimizer = optimizers.get(optimizer)
        self._run_eagerly = kwargs.pop("run_eagerly", None)
        # reset the training/evaluate/predict function
        self._reset_compile_cache()

        # get the loss
        self.compiled_loss = loss_functions.get(loss)
        # Only metric supported during the training is mean Loss
        self.compiled_metrics = compile_utils.MetricsContainer(
            metrics_mod.Mean(name="loss"), None, None
        )  # Total loss.

        # set the initializer and regularizer of the embedding matrices in the
        # encoding layer
        self.encoding_layer.set_initializer(entity_relation_initializer)
        self.encoding_layer.set_regularizer(entity_relation_regularizer)
        self._is_compiled = True

    @property
    def metrics(self):
        """Returns all the metrics that will be computed during training."""
        metrics = []
        if self._is_compiled:
            if self.compiled_loss is not None:
                metrics += self.compiled_loss.metrics
        return metrics

    def get_emb_matrix_test(self, part_number=1, number_of_parts=1):
        """Get the embedding matrix during evaluation.

        Parameters
        ----------
        part number: int
            Specifies which part to return from the ``number_of_parts`` in which the entire embedding matrix is split.
        number_of_parts: int
            Total number of parts in which to split the embedding matrix.

        Returns
        -------
        emb_matrix: np.array, shape (n,k)
            Part of the embedding matrix corresponding to `part_number`.
        start_index: int
            Original entity index (data dict) of the first row of the `emb_matrix`.
        end_index: int
            Original entity index (data dict) of the last row of the `emb_matrix`.

        """
        if number_of_parts == 1:
            if self.entities_subset.shape[0] != 0:
                out = tf.nn.embedding_lookup(
                    self.encoding_layer.ent_emb, self.entities_subset
                )
            else:
                out = self.encoding_layer.ent_emb
            return out, 0, out.shape[0] - 1
        else:
            with shelve.open(
                self.partitioner_metadata["ent_map_fname"]
            ) as ent_partition:
                batch_size = int(
                    np.ceil(len(ent_partition.keys()) / number_of_parts)
                )
                indices = np.arange(
                    part_number * batch_size, (part_number + 1) * batch_size
                ).astype(str)
                emb_matrix = []
                for idx in indices:
                    try:
                        emb_matrix.append(ent_partition[idx])
                    except KeyError:
                        break
                return np.array(emb_matrix), int(indices[0]), int(indices[-1])

    def make_test_function(self):
        """Similar to keras lib, this function returns the handle to test step function.

        It processes one batch of data by iterating over the dataset iterator and computes the test metrics.

        Returns
        -------
        out: Function handle
              Handle to the test step function.
        """

        # if self.test_function is not None:
        #    return self.test_function

        def test_function(iterator):
            # total number of parts in which to split the embedding matrix
            # (default 1, i.e., use full matrix as it is)
            number_of_parts = 1

            # if it is partitioned training
            if self.is_partitioned_training:
                # split the emb matrix based on number of buckets
                number_of_parts = self.partitioner_k

            data = next(iterator)
            if self.use_filter:
                inputs, filters = data[0], data[1]
            else:
                if self.data_shape > 3:
                    inputs, filters = data[
                        0
                    ], tf.RaggedTensor.from_row_lengths([], [])
                else:
                    inputs, filters = data, tf.RaggedTensor.from_row_lengths(
                        [], []
                    )

            # compute the output shape based on the type of corruptions to be
            # used
            output_shape = 0
            if "s" in self.corrupt_side:
                output_shape += 1

            if "o" in self.corrupt_side:
                output_shape += 1

            # create an array to store the ranks based on output shape
            overall_rank = tf.zeros(
                (output_shape, tf.shape(inputs)[0]), dtype=np.int32
            )

            if self.is_partitioned_training:
                inputs = self.process_model_inputs_for_test(inputs)

            # run the loop based on number of parts in which the original emb
            # matrix was generated
            for j in range(number_of_parts):
                # get the embedding matrix along with entity ids of first and
                # last row of emb matrix
                emb_mat, start_ent_id, end_ent_id = self.get_emb_matrix_test(
                    j, number_of_parts
                )
                # compute the rank
                ranks = self._get_ranks(
                    inputs,
                    emb_mat,
                    start_ent_id,
                    end_ent_id,
                    filters,
                    self.mapping_dict,
                    self.corrupt_side,
                    self.ranking_strategy,
                )
                # store it in the output
                for i in tf.range(output_shape):
                    overall_rank = tf.tensor_scatter_nd_add(
                        overall_rank, [[i]], [ranks[i, :]]
                    )

            overall_rank = tf.transpose(
                tf.reshape(overall_rank, (output_shape, -1))
            )
            # if corruption type is s+o then add s and o ranks and return the
            # added ranks
            if self.corrupt_side == "s+o":
                # add the subject and object ranks
                overall_rank = tf.reduce_sum(overall_rank, 1)
                # return the added ranks
                return tf.reshape(overall_rank, (-1, 1))

            return overall_rank

        if not self.run_eagerly and not self.is_partitioned_training:
            test_function = def_function.function(
                test_function, experimental_relax_shapes=True
            )

        self.test_function = test_function

        return self.test_function

    def process_model_inputs_for_test(self, triples):
        """Return the processed triples.

        Parameters
        ----------
        triples: np.array
            Triples to be processed.

        Returns
        -------
        out_triples: np.array or list
            In regular (non partitioned) mode, the triples are returned as they are given in input.
            In case of partitioning, it returns the triple embeddings as a list of size 3, where each element
            is a np.array of subjects, predicates and objects embeddings.
        """
        if self.is_partitioned_training:
            np_triples = triples.numpy()
            sub_emb_out = []
            obj_emb_out = []
            rel_emb_out = []
            with shelve.open(
                self.partitioner_metadata["ent_map_fname"]
            ) as ent_emb:
                with shelve.open(
                    self.partitioner_metadata["rel_map_fname"]
                ) as rel_emb:
                    for triple in np_triples:
                        sub_emb_out.append(ent_emb[str(triple[0])])
                        rel_emb_out.append(rel_emb[str(triple[1])])
                        obj_emb_out.append(ent_emb[str(triple[2])])

            emb_out = [
                np.array(sub_emb_out),
                np.array(rel_emb_out),
                np.array(obj_emb_out),
            ]
            return emb_out
        else:
            return triples

    def evaluate(
        self,
        x=None,
        batch_size=32,
        verbose=True,
        use_filter=False,
        corrupt_side="s,o",
        entities_subset=None,
        ranking_strategy="worst",
        callbacks=None,
        dataset_type="test",
    ):
        """
        Evaluate the inputs against corruptions and return ranks.

        Parameters
        ----------
        x: np.array, shape (n,3) or str or GraphDataLoader or AbstractGraphPartitioner
            Data OR Filename of the data file OR Data Handle to be used for training.
        batch_size: int
            Batch size to use during training.
            May be overridden if ``x`` is `GraphDataLoader` or `AbstractGraphPartitioner` instance
        verbose: bool
            Verbosity mode.
        use_filter: bool or dict
            Whether to use a filter of not. If a dictionary is specified, the data in the dict is concatenated
            and used as filter.
        corrupt_side: str
            Which side to corrupt of a triple to corrupt. It can be the subject (``corrupt_size="s"``),
            the object (``corrupt_size="o"``), the subject and the object (``corrupt_size="s+o"`` or
            ``corrupt_size="s,o"``) (default:`"s,o"`).
        ranking_strategy: str
            Indicates how to break ties when a test triple gets the same rank of a corruption.
            Can be one of the three types: `"best"`, `"middle"`, `"worst"` (default: `"worst"`, i.e.,
            the worst rank is assigned to the test triple).
        entities_subset: list or np.array
            Subset of entities to be used for generating corruptions.
        callbacks: list of keras.callbacks.Callback instances
            List of callbacks to apply during evaluation.

        Returns
        -------
        rank: np.array, shape (n, number of corrupted sides)
            Ranking of test triples against subject corruptions and/or object corruptions.

        Example
        -------
        >>> from ampligraph.datasets import load_fb15k_237
        >>> from ampligraph.latent_features import ScoringBasedEmbeddingModel
        >>> from ampligraph.evaluation.metrics import mrr_score, hits_at_n_score, mr_score
        >>> X = load_fb15k_237()
        >>> model = ScoringBasedEmbeddingModel(eta=5,
        >>>                                    k=300,
        >>>                                    scoring_type='ComplEx',
        >>>                                    seed=0)
        >>> model.compile(optimizer='adam', loss='nll')
        >>> model.fit(X['train'],
        >>>           batch_size=10000,
        >>>           epochs=5)
        Epoch 1/5
        29/29 [==============================] - 2s 71ms/step - loss: 67361.3047
        Epoch 2/5
        29/29 [==============================] - 1s 35ms/step - loss: 67318.6094
        Epoch 3/5
        29/29 [==============================] - 1s 35ms/step - loss: 67020.0703
        Epoch 4/5
        29/29 [==============================] - 1s 33ms/step - loss: 65867.3750
        Epoch 5/5
        29/29 [==============================] - 1s 34ms/step - loss: 63517.9062
        >>> ranks = model.evaluate(X['test'],
        >>>                        batch_size=100,
        >>>                        corrupt_side='s,o',
        >>>                        use_filter={'train': X['train'],
        >>>                                    'valid': X['valid'],
        >>>                                    'test': X['test'])
        >>> mr_score(ranks), mrr_score(ranks), hits_at_n_score(ranks, 1), hits_at_n_score(ranks, 10), len(ranks)
        28 triples containing invalid keys skipped!
        9 triples containing invalid keys skipped!
        2045/2045 [==============================] - 149s 73ms/step
        (428.44671689989235,
         0.25761041025282316,
         0.1898179861043155,
         0.391965945787259,
         20438)
        """
        # get the test set handler
        self.data_handler_test = data_adapter.DataHandler(
            x,
            batch_size=batch_size,
            dataset_type=dataset_type,
            epochs=1,
            use_filter=use_filter,
            use_indexer=self.data_indexer,
        )

        assert corrupt_side in [
            "s",
            "o",
            "s,o",
            "s+o",
        ], "Invalid value for corrupt_side"
        assert ranking_strategy in [
            "best",
            "middle",
            "worst",
        ], "Invalid value for ranking_strategy"

        self.corrupt_side = corrupt_side
        self.ranking_strategy = ranking_strategy

        self.entities_subset = tf.constant([])
        self.mapping_dict = tf.lookup.experimental.DenseHashTable(
            tf.int32, tf.int32, -1, -1, -2
        )
        if entities_subset is not None:
            entities_subset = self.data_indexer.get_indexes(
                entities_subset, "e"
            )
            self.entities_subset = tf.constant(entities_subset, dtype=tf.int32)
            self.mapping_dict.insert(
                self.entities_subset, tf.range(self.entities_subset.shape[0])
            )

        # flag to indicate if we are using filter or not
        self.use_filter = (
            self.data_handler_test._parent_adapter.backend.use_filter
            or isinstance(
                self.data_handler_test._parent_adapter.backend.use_filter, dict
            )
        )

        # Container that configures and calls `tf.keras.Callback`s.
        if not isinstance(callbacks, callbacks_module.CallbackList):
            callbacks = callbacks_module.CallbackList(
                callbacks,
                add_history=True,
                add_progbar=verbose != 0,
                model=self,
                verbose=verbose,
                epochs=1,
                steps=self.data_handler_test.inferred_steps,
            )

        test_function = self.make_test_function()

        # before test begins call this callback function
        callbacks.on_test_begin()

        self.all_ranks = []

        # enumerate over the data
        for _, iterator in self.data_handler_test.enumerate_epochs():
            # handle the stop iteration of data iterator in this scope
            with self.data_handler_test.catch_stop_iteration():
                # iterate over the dataset
                for step in self.data_handler_test.steps():
                    # before a batch is processed call this callback function
                    callbacks.on_test_batch_begin(step)

                    # process this batch
                    overall_rank = test_function(iterator)
                    # increment the rank by 1 (ranks returned are from (0 -
                    # n-1) so increment by 1
                    overall_rank += 1
                    # save the ranks of the batch triples
                    self.all_ranks.append(overall_rank)
                    # after a batch is processed call this callback function
                    callbacks.on_test_batch_end(step)
        # on test end call this method
        callbacks.on_test_end()
        # return ranks
        return np.concatenate(self.all_ranks)

    def predict_step(self, inputs):
        """Returns the output of predict step on a batch of data."""
        if self.data_shape > 3 and isinstance(inputs, tuple):
            inputs = inputs[0]
        score_pos = self(inputs, False)
        return score_pos

    def predict_step_partitioning(self, inputs):
        """Returns the output of predict step on a batch of data."""
        score_pos = self.scoring_layer(inputs)
        return score_pos

    def make_predict_function(self):
        """Similar to keras lib, this function returns the handle to the predict step function.

        It processes one batch of data by iterating over the dataset iterator and computes the prediction outputs.

        Returns
        -------
        out: Function handle
              Handle to the predict function.
        """
        if self.predict_function is not None:
            return self.predict_function

        def predict_function(iterator):
            inputs = next(iterator)
            if self.is_partitioned_training:
                inputs = self.process_model_inputs_for_test(inputs)
                outputs = self.predict_step_partitioning(inputs)
            else:
                outputs = self.predict_step(inputs)
            return outputs

        if not self.run_eagerly and not self.is_partitioned_training:
            predict_function = def_function.function(
                predict_function, experimental_relax_shapes=True
            )

        self.predict_function = predict_function
        return self.predict_function

    def predict(self, x, batch_size=32, verbose=0, callbacks=None):
        """
        Compute scores of the input triples.

        Parameters
        -----------
        x: np.array, shape (n, 3) or str or GraphDataLoader or AbstractGraphPartitioner
            Data OR Filename of the data file OR Data Handle to be used for training.
        batch_size: int
            Batch size to use during training.
            May be overridden if ``x`` is `GraphDataLoader` or `AbstractGraphPartitioner` instance
        verbose: bool
            Verbosity mode.
        callbacks: list of keras.callbacks.Callback instances
            List of callbacks to apply during evaluation.

        Returns
        -------
        scores: np.array, shape (n, )
            Score of the input triples.

        Example
        -------
        >>> from ampligraph.latent_features import ScoringBasedEmbeddingModel
        >>> import numpy as np
        >>> from ampligraph.datasets import load_fb15k_237
        >>> X = load_fb15k_237()
        >>> model = ScoringBasedEmbeddingModel(eta=5,
        >>>                                    k=300,
        >>>                                    scoring_type='ComplEx',
        >>>                                    seed=0)
        >>> model.compile(optimizer='adam', loss='nll')
        >>> model.fit(X['train'],
        >>>           batch_size=10000,
        >>>           epochs=5)
        Epoch 1/5
        29/29 [==============================] - 7s 228ms/step - loss: 67361.2734
        Epoch 2/5
        29/29 [==============================] - 5s 184ms/step - loss: 67318.8203
        Epoch 3/5
        29/29 [==============================] - 5s 187ms/step - loss: 67021.1641
        Epoch 4/5
        29/29 [==============================] - 5s 188ms/step - loss: 65865.5547
        Epoch 5/5
        29/29 [==============================] - 5s 188ms/step - loss: 63510.2773

        >>> pred = model.predict(X['test'],
        >>>                      batch_size=100)
        >>> print(np.sort(pred))
        [-1.0868168  -0.46582496 -0.44715863 ...  3.2484274   3.3147712  3.326     ]

        """

        self.data_handler_test = data_adapter.DataHandler(
            x,
            batch_size=batch_size,
            dataset_type="test",
            epochs=1,
            use_filter=False,
            use_indexer=self.data_indexer,
        )

        if not isinstance(callbacks, callbacks_module.CallbackList):
            callbacks = callbacks_module.CallbackList(
                callbacks,
                add_history=True,
                add_progbar=verbose != 0,
                model=self,
                verbose=verbose,
                epochs=1,
                steps=self.data_handler_test.inferred_steps,
            )

        predict_function = self.make_predict_function()
        callbacks.on_predict_begin()
        outputs = []
        for _, iterator in self.data_handler_test.enumerate_epochs():
            with self.data_handler_test.catch_stop_iteration():
                for step in self.data_handler_test.steps():
                    callbacks.on_predict_batch_begin(step)
                    batch_outputs = predict_function(iterator)
                    outputs.append(batch_outputs)

                    callbacks.on_predict_batch_end(
                        step, {"outputs": batch_outputs}
                    )
        callbacks.on_predict_end()
        return np.concatenate(outputs)

    def make_calibrate_function(self):
        """Similar to keras lib, this function returns the handle to the calibrate step function.

        It processes one batch of data by iterating over the dataset iterator and computes the calibration
        of predictions.

        Returns
        -------
        out: Function handle
              Handle to the calibration function.
        """

        def calibrate_with_corruption(iterator):
            inputs = next(iterator)
            if self.data_shape > 3 and isinstance(inputs, tuple):
                inputs = inputs[0]
            if self.is_partitioned_training:
                inp_emb = self.process_model_inputs_for_test(inputs)
                inp_score = self.scoring_layer(inp_emb)

                corruptions = self.corruption_layer(inputs, self.num_ents, 1)
                corr_emb = self.encoding_layer(corruptions)
                corr_score = self.scoring_layer(corr_emb)
            else:
                inp_emb = self.encoding_layer(inputs)
                inp_score = self.scoring_layer(inp_emb)

                corruptions = self.corruption_layer(inputs, self.num_ents, 1)
                corr_emb = self.encoding_layer(corruptions)
                corr_score = self.scoring_layer(corr_emb)
            return inp_score, corr_score

        def calibrate_with_negatives(iterator):
            inputs = next(iterator)
            if self.data_shape > 3 and isinstance(inputs, tuple):
                inputs = inputs[0]
            if self.is_partitioned_training:
                inp_emb = self.process_model_inputs_for_test(inputs)
                inp_score = self.scoring_layer(inp_emb)
            else:
                inp_emb = self.encoding_layer(inputs)
                inp_score = self.scoring_layer(inp_emb)
            return inp_score

        if self.is_calibrate_with_corruption:
            calibrate_fn = calibrate_with_corruption
        else:
            calibrate_fn = calibrate_with_negatives

        if not self.run_eagerly and not self.is_partitioned_training:
            calibrate_fn = def_function.function(
                calibrate_fn, experimental_relax_shapes=True
            )

        return calibrate_fn

    def calibrate(
        self,
        X_pos,
        X_neg=None,
        positive_base_rate=None,
        batch_size=32,
        epochs=50,
        verbose=0,
    ):
        """Calibrate predictions.

        The method implements the heuristics described in :cite:`calibration`,
        using Platt scaling :cite:`platt1999probabilistic`.

        The calibrated predictions can be obtained with :meth:`predict_proba`
        after calibration is done.

        Ideally, calibration should be performed on a validation set that was not used to train the embeddings.

        There are two modes of operation, depending on the availability of negative triples:

        #. Both positive and negative triples are provided via ``X_pos`` and ``X_neg`` respectively. \
        The optimization is done using a second-order method (limited-memory BFGS), \
        therefore no hyperparameter needs to be specified.

        #. Only positive triples are provided, and the negative triples are generated by corruptions, \
        just like it is done in training or evaluation. The optimization is done using a first-order method (ADAM), \
        therefore ``batches_count`` and ``epochs`` must be specified.


        Calibration is highly dependent on the base rate of positive triples.
        Therefore, for mode (2) of operation, the user is required to provide the ``positive_base_rate`` argument.
        For mode (1), that can be inferred automatically by the relative sizes of the positive and negative sets,
        but the user can override this behaviour by providing a value to ``positive_base_rate``.

        Defining the positive base rate is the biggest challenge when calibrating without negatives. That depends on
        the user choice of triples to be evaluated during test time.
        Let's take the WN11 dataset as an example: it has around 50% positives triples on both the validation set
        and test set, so the positive base rate follows to be 50%. However, should the user resample it to have
        75% positives and 25% negatives, the previous calibration would be degraded. The user must recalibrate
        the model with a 75% positive base rate. Therefore, this parameter depends on how the user handles the
        dataset and cannot be determined automatically or a priori.

        .. Note ::
            :cite:`calibration` `calibration experiments available here
            <https://github.com/Accenture/AmpliGraph/tree/paper/ICLR-20/experiments/ICLR-20>`_.


        Parameters
        ----------
        X_pos : np.array, shape (n,3) or str or GraphDataLoader or AbstractGraphPartitioner
            Data OR Filename of the data file OR Data Handle to be used as positive triples.
        X_neg : np.array, shape (n,3) or str or GraphDataLoader or AbstractGraphPartitioner
            Data OR Filename of the data file OR Data Handle to be used as negative triples.

            If `None`, the negative triples are generated via corruptions
            and the user must provide a positive base rate instead.

        positive_base_rate: float
            Base rate of positive statements.

            For example, if we assume there is an even chance for any query to be true, the base rate would be 50%.

            If ``X_neg`` is provided and ``positive_base_rate=None``, the relative sizes of ``X_pos`` and ``X_neg``
            will be used to determine the base rate. Say we have 50 positive triples and 200 negative
            triples, the positive base rate will be assumed to be :math:`\\frac{50}{(50+200)} = \\frac{1}{5} = 0.2`.

            This value must be :math:`\\in [0,1]`.
        batches_size: int
            Batch size for positives.
        epochs: int
            Number of epochs used to train the Platt scaling model.
            Only applies when ``X_neg=None``.
        verbose: bool
            Verbosity (default: `False`).

        Example
        -------
        >>> from ampligraph.datasets import load_fb15k_237
        >>> from ampligraph.latent_features import ScoringBasedEmbeddingModel
        >>> import numpy as np
        >>> dataset = load_fb15k_237()
        >>> model = ScoringBasedEmbeddingModel(eta=5,
        >>>                                    k=300,
        >>>                                    scoring_type='ComplEx')
        >>> model.compile(optimizer='adam', loss='nll')
        >>> model.fit(dataset['train'],
        >>>           batch_size=10000,
        >>>           epochs=5)
        >>> print('Raw scores (sorted):', np.sort(model.predict(dataset['test'])))
        >>> print('Indices obtained by sorting (scores):', np.argsort(model.predict(dataset['test'])))
        Raw scores (sorted): [-1.0689778   -0.42082012  -0.39887887 ...  3.261838  3.2755773  3.2768354 ]
        Indices obtained by sorting (scores): [ 3834 18634  4066 ...  6237 13633 10961]
        >>> model.calibrate(dataset['test'],
        >>>                 batch_size=10000,
        >>>                 positive_base_rate=0.9,
        >>>                 epochs=100)
        >>> print('Calibrated scores (sorted):', np.sort(model.predict_proba(dataset['test'])))
        >>> print('Indices obtained by sorting (Calibrated):', np.argsort(model.predict_proba(dataset['test'])))
        Calibrated scores (sorted): [0.49547982 0.5396996  0.54118955 ... 0.7624245  0.7631044  0.76316655]
        Indices obtained by sorting (Calibrated): [ 3834 18634  4066 ...  6237 13633 10961]

        """
        self.is_calibrated = False
        data_handler_calibrate_pos = data_adapter.DataHandler(
            X_pos,
            batch_size=batch_size,
            dataset_type="test",
            epochs=epochs,
            use_filter=False,
            use_indexer=self.data_indexer,
        )

        pos_size = data_handler_calibrate_pos._parent_adapter.get_data_size()
        neg_size = pos_size

        if X_neg is None:
            assert (
                positive_base_rate is not None
            ), "Please provide the negatives or positive base rate!"
            self.is_calibrate_with_corruption = True
        else:
            self.is_calibrate_with_corruption = False

            pos_batch_count = int(np.ceil(pos_size / batch_size))

            data_handler_calibrate_neg = data_adapter.DataHandler(
                X_neg,
                batch_size=batch_size,
                dataset_type="test",
                epochs=epochs,
                use_filter=False,
                use_indexer=self.data_indexer,
            )

            neg_size = (
                data_handler_calibrate_neg._parent_adapter.get_data_size()
            )
            neg_batch_count = int(np.ceil(neg_size / batch_size))

            if pos_batch_count != neg_batch_count:
                batch_size_neg = int(np.ceil(neg_size / pos_batch_count))
                data_handler_calibrate_neg = data_adapter.DataHandler(
                    X_neg,
                    batch_size=batch_size_neg,
                    dataset_type="test",
                    epochs=epochs,
                    use_filter=False,
                    use_indexer=self.data_indexer,
                )

            if positive_base_rate is None:
                positive_base_rate = pos_size / (pos_size + neg_size)

        if positive_base_rate is not None and (
            positive_base_rate <= 0 or positive_base_rate >= 1
        ):
            raise ValueError(
                "positive_base_rate must be a value between 0 and 1."
            )

        self.calibration_layer = CalibrationLayer(
            pos_size, neg_size, positive_base_rate
        )
        calibrate_function = self.make_calibrate_function()

        optimizer = tf.keras.optimizers.Adam()

        if not self.is_calibrate_with_corruption:
            negative_iterator = iter(
                data_handler_calibrate_neg.enumerate_epochs()
            )

        for _, iterator in data_handler_calibrate_pos.enumerate_epochs(True):
            if not self.is_calibrate_with_corruption:
                _, neg_handle = next(negative_iterator)

            with data_handler_calibrate_pos.catch_stop_iteration():
                for step in data_handler_calibrate_pos.steps():
                    if self.is_calibrate_with_corruption:
                        scores_pos, scores_neg = calibrate_function(iterator)

                    else:
                        scores_pos = calibrate_function(iterator)
                        with data_handler_calibrate_neg.catch_stop_iteration():
                            scores_neg = calibrate_function(neg_handle)

                    with tf.GradientTape() as tape:
                        out = self.calibration_layer(scores_pos, scores_neg, 1)

                    gradients = tape.gradient(
                        out, self.calibration_layer._trainable_weights
                    )
                    # update the trainable params
                    optimizer.apply_gradients(
                        zip(
                            gradients,
                            self.calibration_layer._trainable_weights,
                        )
                    )
        self.is_calibrated = True

    def predict_proba(self, x, batch_size=32, verbose=0, callbacks=None):
        """
        Compute calibrated scores (:math:`0 ≤ score ≤ 1`) for the input triples.

        Parameters
        ----------
        x: np.array, shape (n,3) or str or GraphDataLoader or AbstractGraphPartitioner
            Data OR Filename of the data file OR Data Handle to be used for training.
        batch_size: int
            Batch size to use during training.
            May be overridden if ``x`` is `GraphDataLoader` or `AbstractGraphPartitioner` instance.
        verbose: bool
            Verbosity mode (default: `False`).
        callbacks: list of keras.callbacks.Callback instances
            List of callbacks to apply during evaluation.

        Returns
        -------
        scores: np.array, shape (n, )
            Calibrated scores for the input triples.

        Example
        -------
        >>> from ampligraph.datasets import load_fb15k_237
        >>> from ampligraph.latent_features import ScoringBasedEmbeddingModel
        >>> import numpy as np
        >>> dataset = load_fb15k_237()
        >>> model = ScoringBasedEmbeddingModel(eta=5,
        >>>                                    k=300,
        >>>                                    scoring_type='ComplEx')
        >>> model.compile(optimizer='adam', loss='nll')
        >>> model.fit(dataset['train'],
        >>>           batch_size=10000,
        >>>           epochs=5)
        >>> print('Raw scores (sorted):', np.sort(model.predict(dataset['test'])))
        >>> print('Indices obtained by sorting (scores):', np.argsort(model.predict(dataset['test'])))
        Raw scores (sorted): [-1.0384613  -0.46752608 -0.45149875 ...  3.2897844  3.3034315  3.3280635 ]
        Indices obtained by sorting (scores): [ 3834 18634  4066 ...  1355 13633 10961]
        >>> model.calibrate(dataset['test'],
        >>>                 batch_size=10000,
        >>>                 positive_base_rate=0.9,
        >>>                 epochs=100)
        >>> print('Calibrated scores (sorted):', np.sort(model.predict_proba(dataset['test'])))
        >>> print('Indices obtained by sorting (Calibrated):', np.argsort(model.predict_proba(dataset['test'])))
        Calibrated scores (sorted): [0.5553725  0.5556108  0.5568415  ... 0.6211011  0.62382233 0.6297585 ]
        Indices obtained by sorting (Calibrated): [14573 11577  4404 ... 17817 17816   733]
        """
        if not self.is_calibrated:
            msg = "Model has not been calibrated. \
            Please call `model.calibrate(...)` before predicting probabilities."

            raise RuntimeError(msg)

        self.data_handler_test = data_adapter.DataHandler(
            x,
            batch_size=batch_size,
            dataset_type="test",
            epochs=1,
            use_filter=False,
            use_indexer=self.data_indexer,
        )

        if not isinstance(callbacks, callbacks_module.CallbackList):
            callbacks = callbacks_module.CallbackList(
                callbacks,
                add_history=True,
                add_progbar=verbose != 0,
                model=self,
                verbose=verbose,
                epochs=1,
                steps=self.data_handler_test.inferred_steps,
            )

        predict_function = self.make_predict_function()
        callbacks.on_predict_begin()
        outputs = []
        for _, iterator in self.data_handler_test.enumerate_epochs():
            with self.data_handler_test.catch_stop_iteration():
                for step in self.data_handler_test.steps():
                    callbacks.on_predict_batch_begin(step)
                    batch_outputs = predict_function(iterator)
                    probas = self.calibration_layer(batch_outputs, training=0)
                    outputs.append(probas)

                    callbacks.on_predict_batch_end(
                        step, {"outputs": batch_outputs}
                    )
        callbacks.on_predict_end()
        return np.concatenate(outputs)

    def get_embeddings(self, entities, embedding_type="e"):
        """Get the embeddings of entities or relations.

        .. Note ::

            Use :meth:`ampligraph.utils.create_tensorboard_visualizations` to visualize the embeddings with TensorBoard.

        Parameters
        ----------
        entities : array-like, shape=(n)
            The entities (or relations) of interest. Element of the vector must be the original string literals, and
            not internal IDs.
        embedding_type : str
            If `'e'` is passed, ``entities`` argument will be considered as a list of knowledge graph entities
            (i.e., nodes). If set to `'r'`, ``entities`` will be treated as relations instead.
        Returns
        -------
        embeddings : ndarray, shape (n, k)
            An array of `k`-dimensional embeddings.

        Example
        -------
        >>> from ampligraph.latent_features import ScoringBasedEmbeddingModel
        >>> from ampligraph.datasets import load_fb15k_237
        >>> X = load_fb15k_237()
        >>> model = ScoringBasedEmbeddingModel(eta=5,
        >>>                                    k=300,
        >>>                                    scoring_type='ComplEx',
        >>>                                    seed=0)
        >>> model.compile(optimizer='adam', loss='nll')
        >>> model.fit(X['train'],
        >>>           batch_size=10000,
        >>>           epochs=5,
        >>>           verbose=False)
        >>> model.get_embeddings(['/m/027rn', '/m/06v8s0'], 'e')
        array([[ 0.04482496  0.11973907  0.01117733 ... -0.13391922  0.11103553  -0.08132861]
         [-0.10158381  0.08108605 -0.07608676 ...  0.0591407  0.02791426  0.07559016]], dtype=float32)
        """

        if embedding_type == "e":
            lookup_concept = self.data_indexer.get_indexes(entities, "e")
            if self.is_partitioned_training:
                emb_out = []
                with shelve.open(
                    self.partitioner_metadata["ent_map_fname"]
                ) as ent_emb:
                    for ent_id in lookup_concept:
                        emb_out.append(ent_emb[str(ent_id)])
            else:
                return tf.nn.embedding_lookup(
                    self.encoding_layer.ent_emb, lookup_concept
                ).numpy()
        elif embedding_type == "r":
            lookup_concept = self.data_indexer.get_indexes(entities, "r")
            if self.is_partitioned_training:
                emb_out = []
                with shelve.open(
                    self.partitioner_metadata["rel_map_fname"]
                ) as rel_emb:
                    for rel_id in lookup_concept:
                        emb_out.append(rel_emb[str(rel_id)])
            else:
                return tf.nn.embedding_lookup(
                    self.encoding_layer.rel_emb, lookup_concept
                ).numpy()
        else:
            msg = "Invalid entity type: {}".format(embedding_type)
            raise ValueError(msg)
