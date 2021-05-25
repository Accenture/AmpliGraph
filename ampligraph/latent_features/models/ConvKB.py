# Copyright 2019-2021 The AmpliGraph Authors. All Rights Reserved.
#
# This file is Licensed under the Apache License, Version 2.0.
# A copy of the Licence is available in LICENCE, or at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
import numpy as np
import tensorflow as tf
import logging

from .EmbeddingModel import EmbeddingModel, register_model, ENTITY_THRESHOLD
from ..initializers import DEFAULT_XAVIER_IS_UNIFORM
from ampligraph.latent_features import constants as constants
import time

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@register_model("ConvKB", {'num_filters': 32, 'filter_sizes': [1], 'dropout': 0.1})
class ConvKB(EmbeddingModel):
    r"""Convolution-based model

    The ConvKB model :cite:`Nguyen2018`:

    .. math::

        f_{ConvKB}= concat \,(g \, ([\mathbf{e}_s, \mathbf{r}_p, \mathbf{e}_o]) * \Omega)) \cdot W

    where :math:`g` is a non-linear function,  :math:`*` is the convolution operator,
    :math:`\cdot` is the dot product, :math:`concat` is the concatenation operator
    and :math:`\Omega` is a set of filters.

    .. note::
        The evaluation protocol implemented in :meth:`ampligraph.evaluation.evaluate_performance` assigns the worst rank
        to a positive test triple in case of a tie with negatives. This is the agreed upon behaviour in literature.
        The original ConvKB implementation :cite:`Nguyen2018` assigns instead the top rank, hence leading to
        `results which are not directly comparable with
        literature <https://github.com/daiquocnguyen/ConvKB/issues/5>`_ .
        We report results obtained with the agreed-upon protocol (tie=worst rank). Note that under these conditions
        the model :ref:`does not reach the state-of-the-art results claimed in the original paper<eval_experiments>`.

    Examples
    --------
    >>> from ampligraph.latent_features import ConvKB
    >>> from ampligraph.datasets import load_wn18
    >>> model = ConvKB(batches_count=2, seed=22, epochs=1, k=10, eta=1,
    >>>               embedding_model_params={'num_filters': 32, 'filter_sizes': [1],
    >>>                                       'dropout': 0.1},
    >>>               optimizer='adam', optimizer_params={'lr': 0.001},
    >>>               loss='pairwise', loss_params={}, verbose=True)
    >>>
    >>> X = load_wn18()
    >>>
    >>> model.fit(X['train'])
    >>>
    >>> print(model.predict(X['test'][:5]))
    [[0.2803744], [0.0866661], [0.012815937], [-0.004235901], [-0.010947697]]
    """

    def __init__(self,
                 k=constants.DEFAULT_EMBEDDING_SIZE,
                 eta=constants.DEFAULT_ETA,
                 epochs=constants.DEFAULT_EPOCH,
                 batches_count=constants.DEFAULT_BATCH_COUNT,
                 seed=constants.DEFAULT_SEED,
                 embedding_model_params={'num_filters': 32,
                                         'filter_sizes': [1],
                                         'dropout': 0.1},
                 optimizer=constants.DEFAULT_OPTIM,
                 optimizer_params={'lr': constants.DEFAULT_LR},
                 loss=constants.DEFAULT_LOSS,
                 loss_params={},
                 regularizer=constants.DEFAULT_REGULARIZER,
                 regularizer_params={},
                 initializer=constants.DEFAULT_INITIALIZER,
                 initializer_params={'uniform': DEFAULT_XAVIER_IS_UNIFORM},
                 large_graphs=False,
                 verbose=constants.DEFAULT_VERBOSE):
        """Initialize an EmbeddingModel

        Parameters
        ----------
        k : int
            Embedding space dimensionality.

        eta : int
            The number of negatives that must be generated at runtime during training for each positive.

        epochs : int
            The iterations of the training loop.

        batches_count : int
            The number of batches in which the training set must be split during the training loop.

        seed : int
            The seed used by the internal random numbers generator.

        embedding_model_params : dict
            ConvKB-specific hyperparams:
            - **num_filters** - Number of feature maps per convolution kernel. Default: 32
            - **filter_sizes** - List of convolution kernel sizes. Default: [1]
            - **dropout** - Dropout on the embedding layer. Default: 0.0
            - **'non_linearity'**: can be one of the following values ``linear``, ``softplus``, ``sigmoid``, ``tanh``
            - **'stop_epoch'**: specifies how long to decay (linearly) the numeric values from 1 to original value 
            until it reachs original value.
            - **'structural_wt'**: structural influence hyperparameter [0, 1] that modulates the influence of graph 
            topology. 
            - **'normalize_numeric_values'**: normalize the numeric values, such that they are scaled between [0, 1]

            The last 4 parameters are related to FocusE layers.
            
        optimizer : string
            The optimizer used to minimize the loss function. Choose between
            'sgd', 'adagrad', 'adam', 'momentum'.

        optimizer_params : dict
            Arguments specific to the optimizer, passed as a dictionary.

            Supported keys:

            - **'lr'** (float): learning rate (used by all the optimizers). Default: 0.1.
            - **'momentum'** (float): learning momentum (only used when ``optimizer=momentum``). Default: 0.9.

            Example: ``optimizer_params={'lr': 0.01}``

        loss : string
            The type of loss function to use during training.

        loss_params : dict
            Dictionary of loss-specific hyperparameters. See :ref:`loss functions <loss>`
            documentation for additional details.

            Supported keys:

            - **'lr'** (float): learning rate (used by all the optimizers). Default: 0.1.
            - **'momentum'** (float): learning momentum (only used when ``optimizer=momentum``). Default: 0.9.

            Example: ``optimizer_params={'lr': 0.01, 'label_smoothing': 0.1}``

        regularizer : string
            The regularization strategy to use with the loss function.

            - ``None``: the model will not use any regularizer (default)
            - ``LP``: the model will use L1, L2 or L3 based on the value of ``regularizer_params['p']`` (see below).

        regularizer_params : dict
            Dictionary of regularizer-specific hyperparameters. See the
            :ref:`regularizers <ref-reg>`
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

        large_graphs : bool
            Avoid loading entire dataset onto GPU when dealing with large graphs.

        verbose : bool
            Verbose mode.
        """

        num_filters = embedding_model_params['num_filters']
        filter_sizes = embedding_model_params['filter_sizes']

        if isinstance(filter_sizes, int):
            filter_sizes = [filter_sizes]

        dense_dim = (k * len(filter_sizes) - sum(filter_sizes) + len(filter_sizes)) * num_filters
        embedding_model_params['dense_dim'] = dense_dim
        embedding_model_params['filter_sizes'] = filter_sizes

        super().__init__(k=k, eta=eta, epochs=epochs,
                         batches_count=batches_count, seed=seed,
                         embedding_model_params=embedding_model_params,
                         optimizer=optimizer, optimizer_params=optimizer_params,
                         loss=loss, loss_params=loss_params,
                         regularizer=regularizer, regularizer_params=regularizer_params,
                         initializer=initializer, initializer_params=initializer_params,
                         large_graphs=large_graphs, verbose=verbose)

    def _initialize_parameters(self):
        """Initialize parameters of the model.

            This function creates and initializes entity and relation embeddings (with size k).
            If the graph is large, then it loads only the required entity embeddings (max:batch_size*2)
            and all relation embeddings.
            Overload this function if the parameters needs to be initialized differently.
        """

        with tf.variable_scope('meta'):
            self.tf_is_training = tf.Variable(False, trainable=False)
            self.set_training_true = tf.assign(self.tf_is_training, True)
            self.set_training_false = tf.assign(self.tf_is_training, False)

        timestamp = int(time.time() * 1e6)
        if not self.dealing_with_large_graphs:

            self.ent_emb = tf.get_variable('ent_emb_{}'.format(timestamp),
                                           shape=[len(self.ent_to_idx), self.k],
                                           initializer=self.initializer.get_entity_initializer(
                                           len(self.ent_to_idx), self.k), dtype=tf.float32)
            self.rel_emb = tf.get_variable('rel_emb_{}'.format(timestamp),
                                           shape=[len(self.rel_to_idx), self.k],
                                           initializer=self.initializer.get_relation_initializer(
                                           len(self.rel_to_idx), self.k), dtype=tf.float32)

        else:

            self.ent_emb = tf.get_variable('ent_emb_{}'.format(timestamp),
                                           shape=[self.batch_size * 2, self.internal_k],
                                           initializer=tf.zeros_initializer(), dtype=tf.float32)

            self.rel_emb = tf.get_variable('rel_emb_{}'.format(timestamp),
                                           shape=[len(self.rel_to_idx), self.internal_k],
                                           initializer=self.initializer.get_relation_initializer(
                                           len(self.rel_to_idx), self.internal_k), dtype=tf.float32)

        num_filters = self.embedding_model_params['num_filters']
        filter_sizes = self.embedding_model_params['filter_sizes']
        dense_dim = self.embedding_model_params['dense_dim']
        num_outputs = 1  # i.e. a single score

        self.conv_weights = {}
        for i, filter_size in enumerate(filter_sizes):
            conv_shape = [3, filter_size, 1, num_filters]
            conv_name = 'conv-maxpool-{}'.format(filter_size)
            weights_init = tf.initializers.truncated_normal(seed=self.seed)
            self.conv_weights[conv_name] = {'weights': tf.get_variable('{}_W_{}'.format(conv_name, timestamp),
                                                                       shape=conv_shape,
                                                                       trainable=True, dtype=tf.float32,
                                                                       initializer=weights_init),
                                            'biases': tf.get_variable('{}_B_{}'.format(conv_name, timestamp),
                                                                      shape=[num_filters],
                                                                      trainable=True, dtype=tf.float32,
                                                                      initializer=tf.zeros_initializer())}

        self.dense_W = tf.get_variable('dense_weights_{}'.format(timestamp),
                                       shape=[dense_dim, num_outputs], trainable=True,
                                       initializer=tf.keras.initializers.he_normal(seed=self.seed),
                                       dtype=tf.float32)
        self.dense_B = tf.get_variable('dense_bias_{}'.format(timestamp),
                                       shape=[num_outputs], trainable=False,
                                       initializer=tf.zeros_initializer(), dtype=tf.float32)

    def get_embeddings(self, entities, embedding_type='entity'):
        """Get the embeddings of entities or relations.

        .. Note ::
            Use :meth:`ampligraph.utils.create_tensorboard_visualizations` to visualize the embeddings with TensorBoard.

        Parameters
        ----------
        entities : array-like, dtype=int, shape=[n]
            The entities (or relations) of interest. Element of the vector must be the original string literals, and
            not internal IDs.
        embedding_type : string
            If 'entity', ``entities`` argument will be considered as a list of knowledge graph entities (i.e. nodes).
            If set to 'relation', they will be treated as relation types instead (i.e. predicates).

        Returns
        -------
        embeddings : ndarray, shape [n, k]
            An array of k-dimensional embeddings.

        """
        if not self.is_fitted:
            msg = 'Model has not been fitted.'
            logger.error(msg)
            raise RuntimeError(msg)

        if embedding_type == 'entity':
            emb_list = self.trained_model_params['ent_emb']
            lookup_dict = self.ent_to_idx
        elif embedding_type == 'relation':
            emb_list = self.trained_model_params['rel_emb']
            lookup_dict = self.rel_to_idx
        else:
            msg = 'Invalid entity type: {}'.format(embedding_type)
            logger.error(msg)
            raise ValueError(msg)

        idxs = np.vectorize(lookup_dict.get)(entities)
        return emb_list[idxs]

    def _save_trained_params(self):
        """After model fitting, save all the trained parameters in trained_model_params in some order.
        The order would be useful for loading the model.
        This method must be overridden if the model has any other parameters (apart from entity-relation embeddings).
        """

        params_dict = {}

        if not self.dealing_with_large_graphs:
            params_dict['ent_emb'] = self.sess_train.run(self.ent_emb)
        else:
            params_dict['ent_emb'] = self.ent_emb_cpu

        params_dict['rel_emb'] = self.sess_train.run(self.rel_emb)

        params_dict['conv_weights'] = {}
        for name in self.conv_weights.keys():
            params_dict['conv_weights'][name] = {'weights': self.sess_train.run(self.conv_weights[name]['weights']),
                                                 'biases': self.sess_train.run(self.conv_weights[name]['biases'])}

        params_dict['dense_W'] = self.sess_train.run(self.dense_W)
        params_dict['dense_B'] = self.sess_train.run(self.dense_B)
        self.trained_model_params = params_dict

    def _load_model_from_trained_params(self):
        """Load the model from trained params.
            While restoring make sure that the order of loaded parameters match the saved order.
            It's the duty of the embedding model to load the variables correctly.
            This method must be overridden if the model has any other parameters (apart from entity-relation embeddings)
            This function also set's the evaluation mode to do lazy loading of variables based on the number of
            distinct entities present in the graph.
        """

        # Generate the batch size based on entity length and batch_count
        self.batch_size = int(np.ceil(len(self.ent_to_idx) / self.batches_count))

        if len(self.ent_to_idx) > ENTITY_THRESHOLD:
            self.dealing_with_large_graphs = True

            logger.warning('Your graph has a large number of distinct entities. '
                           'Found {} distinct entities'.format(len(self.ent_to_idx)))

            logger.warning('Changing the variable loading strategy to use lazy loading of variables...')
            logger.warning('Evaluation would take longer than usual.')

        if not self.dealing_with_large_graphs:
            self.ent_emb = tf.Variable(self.trained_model_params['ent_emb'], dtype=tf.float32)
        else:
            self.ent_emb_cpu = self.trained_model_params['ent_emb']
            self.ent_emb = tf.Variable(np.zeros((self.batch_size, self.internal_k)), dtype=tf.float32)

        self.rel_emb = tf.Variable(self.trained_model_params['rel_emb'], dtype=tf.float32)

        with tf.variable_scope('meta'):
            self.tf_is_training = tf.Variable(False, trainable=False)
            self.set_training_true = tf.assign(self.tf_is_training, True)
            self.set_training_false = tf.assign(self.tf_is_training, False)

        self.conv_weights = {}
        for name in self.trained_model_params['conv_weights'].keys():
            W = self.trained_model_params['conv_weights'][name]['weights']
            B = self.trained_model_params['conv_weights'][name]['biases']
            self.conv_weights[name] = {'weights': tf.Variable(W, dtype=tf.float32),
                                       'biases': tf.Variable(B, dtype=tf.float32)}

        self.dense_W = tf.Variable(self.trained_model_params['dense_W'], dtype=tf.float32)
        self.dense_B = tf.Variable(self.trained_model_params['dense_B'], dtype=tf.float32)

    def _fn(self, e_s, e_p, e_o):
        r"""The ConvKB scoring function.

            The function implements the scoring function as defined by:
            .. math::

                \concat(g([\mathbf{e}_s, \mathbf{r}_p, \mathbf{e}_o]) * \Omega)) \cdot W

            Additional details for equivalence of the models available in :cite:`Nguyen2018`.


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
            The operation corresponding to the ConvKB scoring function.

        """

        # Inputs
        e_s = tf.expand_dims(e_s, 1)
        e_p = tf.expand_dims(e_p, 1)
        e_o = tf.expand_dims(e_o, 1)

        self.inputs = tf.expand_dims(tf.concat([e_s, e_p, e_o], axis=1), -1)

        pooled_outputs = []
        for name in self.conv_weights.keys():
            x = tf.nn.conv2d(self.inputs, self.conv_weights[name]['weights'], [1, 1, 1, 1], padding='VALID')
            x = tf.nn.bias_add(x, self.conv_weights[name]['biases'])
            x = tf.nn.relu(x)
            pooled_outputs.append(x)

        # Combine all the pooled features
        x = tf.concat(pooled_outputs, 2)
        x = tf.reshape(x, [-1, self.embedding_model_params['dense_dim']])

        dropout_rate = tf.cond(self.tf_is_training,
                               true_fn=lambda: tf.constant(self.embedding_model_params['dropout']),
                               false_fn=lambda: tf.constant(0, dtype=tf.float32))
        x = tf.nn.dropout(x, rate=dropout_rate)

        self.scores = tf.nn.xw_plus_b(x, self.dense_W, self.dense_B)

        return tf.squeeze(self.scores)

    def fit(self, X, early_stopping=False, early_stopping_params={}, focusE_numeric_edge_values=None,
            tensorboard_logs_path=None):
        """Train a ConvKB model (with optional early stopping).

        The model is trained on a training set X using the training protocol described in :cite:`trouillon2016complex`.

        Parameters
        ----------
        X : ndarray, shape [n, 3]
            The training triples
        early_stopping: bool
            Flag to enable early stopping (default:False).

            If set to ``True``, the training loop adopts the following early
            stopping heuristic:

            - The model will be trained regardless of early stopping for ``burn_in`` epochs.
            - Every ``check_interval`` epochs the method will compute the metric specified in ``criteria``.

            If such metric decreases for ``stop_interval`` checks, we stop
            training early.

            Note the metric is computed on ``x_valid``. This is usually a
            validation set that you held out.

            Also, because ``criteria`` is a ranking metric, it requires
            generating negatives.
            Entities used to generate corruptions can be specified, as long
            as the side(s) of a triple to corrupt.
            The method supports filtered metrics, by passing an array of
            positives to ``x_filter``. This will be used to
            filter the negatives generated on the fly (i.e. the corruptions).

            .. note::

                Keep in mind the early stopping criteria may introduce a
                certain overhead
                (caused by the metric computation).
                The goal is to strike a good trade-off between such overhead
                and saving training epochs.

                A common approach is to use MRR unfiltered: ::

                    early_stopping_params={x_valid=X['valid'], 'criteria':
                    'mrr'}

                Note the size of validation set also contributes to such
                overhead.
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
