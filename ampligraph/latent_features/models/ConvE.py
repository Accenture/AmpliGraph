import numpy as np
import tensorflow as tf
import logging
from sklearn.utils import check_random_state
from tqdm import tqdm
from functools import partial

from .EmbeddingModel import EmbeddingModel, register_model, ENTITY_THRESHOLD
from ..initializers import DEFAULT_XAVIER_IS_UNIFORM
from ampligraph.latent_features import constants as constants

from ...datasets import OneToNDatasetAdapter
from ..optimizers import SGDOptimizer
from ...evaluation import to_idx

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@register_model('ConvE', ['conv_filters', 'conv_kernel_size', 'dropout_embed', 'dropout_conv',
                          'dropout_dense', 'use_bias', 'use_batchnorm'], {})
class ConvE(EmbeddingModel):
    r""" Convolutional 2D KG Embeddings

    The ConvE model :cite:`DettmersMS018`.

    ConvE uses convolutional layers.
    :math:`g` is a non-linear activation function, :math:`\ast` is the linear convolution operator,
    :math:`vec` indicates 2D reshaping.

    .. math::

        f_{ConvE} =  \langle \sigma \, (vec \, ( g \, ([ \overline{\mathbf{e}_s} ; \overline{\mathbf{r}_p} ]
        \ast \Omega )) \, \mathbf{W} )) \, \mathbf{e}_o\rangle


    .. note::

        ConvE does not handle 's+o' corruptions currently, nor ``large_graph`` mode.


    Examples
    --------
    >>> import numpy as np
    >>> from ampligraph.latent_features import ConvE
    >>> model = ConvE(batches_count=1, seed=22, epochs=5, k=100)
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
    [0.42921206 0.38998795]

    """

    def __init__(self,
                 k=constants.DEFAULT_EMBEDDING_SIZE,
                 eta=constants.DEFAULT_ETA,
                 epochs=constants.DEFAULT_EPOCH,
                 batches_count=constants.DEFAULT_BATCH_COUNT,
                 seed=constants.DEFAULT_SEED,
                 embedding_model_params={'conv_filters': constants.DEFAULT_CONVE_CONV_FILTERS,
                                         'conv_kernel_size': constants.DEFAULT_CONVE_KERNEL_SIZE,
                                         'dropout_embed': constants.DEFAULT_CONVE_DROPOUT_EMBED,
                                         'dropout_conv': constants.DEFAULT_CONVE_DROPOUT_CONV,
                                         'dropout_dense': constants.DEFAULT_CONVE_DROPOUT_DENSE,
                                         'use_bias': constants.DEFAULT_CONVE_USE_BIAS,
                                         'use_batchnorm': constants.DEFAULT_CONVE_USE_BATCHNORM},
                 optimizer=constants.DEFAULT_OPTIM,
                 optimizer_params={'lr': constants.DEFAULT_LR},
                 loss='bce',
                 loss_params={'label_weighting': False,
                              'label_smoothing': 0.1},
                 regularizer=constants.DEFAULT_REGULARIZER,
                 regularizer_params={},
                 initializer=constants.DEFAULT_INITIALIZER,
                 initializer_params={'uniform': DEFAULT_XAVIER_IS_UNIFORM},
                 low_memory=False,
                 verbose=constants.DEFAULT_VERBOSE):
        """Initialize a ConvE model

        Also creates a new Tensorflow session for training.

        Parameters
        ----------
        k : int
            Embedding space dimensionality.

        eta : int
            The number of negatives that must be generated at runtime during training for each positive.
            Note: This parameter is not used in ConvE.

        epochs : int
            The iterations of the training loop.

        batches_count : int
            The number of batches in which the training set must be split during the training loop.

        seed : int
            The seed used by the internal random numbers generator.

        embedding_model_params : dict
            ConvE-specific hyperparams:

            - **conv_filters** (int): Number of convolution feature maps. Default: 32
            - **conv_kernel_size** (int): Convolution kernel size. Default: 3
            - **dropout_embed** (float|None): Dropout on the embedding layer. Default: 0.2
            - **dropout_conv** (float|None): Dropout on the convolution maps. Default: 0.3
            - **dropout_dense** (float|None): Dropout on the dense layer. Default: 0.2
            - **use_bias** (bool): Use bias layer. Default: True
            - **use_batchnorm** (bool): Use batch normalization after input, convolution, dense layers. Default: True

        optimizer : string
            The optimizer used to minimize the loss function. Choose between 'sgd', 'adagrad', 'adam', 'momentum'.

        optimizer_params : dict
            Arguments specific to the optimizer, passed as a dictionary.

            Supported keys:

            - **'lr'** (float): learning rate (used by all the optimizers). Default: 0.1.
            - **'momentum'** (float): learning momentum (only used when ``optimizer=momentum``). Default: 0.9.

            Example: ``optimizer_params={'lr': 0.01}``

        loss : string
            The type of loss function to use during training.

            - ``bce``  the model will use binary cross entropy loss function.

        loss_params : dict
            Dictionary of loss-specific hyperparameters. See :ref:`loss functions <loss>` documentation for
            additional details.

            Supported keys:

            - **'lr'** (float): learning rate (used by all the optimizers). Default: 0.1.
            - **'momentum'** (float): learning momentum (only used when ``optimizer=momentum``). Default: 0.9.
            - **'label_smoothing'** (float): applies label smoothing to one-hot outputs. Default: 0.1.
            - **'label_weighting'** (bool): applies label weighting to one-hot outputs. Default: True

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

        verbose : bool
            Verbose mode.

        low_memory : bool
            Train ConvE with a (slower) low_memory option. If MemoryError is still encountered, try raising the
            batches_count value. Default: False.

        """

        # Add default values if not provided in embedding_model_params dict
        default_embedding_model_params = {'conv_filters': constants.DEFAULT_CONVE_CONV_FILTERS,
                                          'conv_kernel_size': constants.DEFAULT_CONVE_KERNEL_SIZE,
                                          'dropout_embed': constants.DEFAULT_CONVE_DROPOUT_EMBED,
                                          'dropout_conv': constants.DEFAULT_CONVE_DROPOUT_CONV,
                                          'dropout_dense': constants.DEFAULT_CONVE_DROPOUT_DENSE,
                                          'use_batchnorm': constants.DEFAULT_CONVE_USE_BATCHNORM,
                                          'use_bias': constants.DEFAULT_CONVE_USE_BATCHNORM}

        for key, val in default_embedding_model_params.items():
            if key not in embedding_model_params.keys():
                embedding_model_params[key] = val

        # Find factor pairs (i,j) of concatenated embedding dimensions, where min(i,j) >= conv_kernel_size
        n = k * 2
        emb_img_depth = 1

        ksize = embedding_model_params['conv_kernel_size']
        nfilters = embedding_model_params['conv_filters']

        emb_img_width, emb_img_height = None, None
        for i in range(int(np.sqrt(n)) + 1, ksize, -1):
            if n % i == 0:
                emb_img_width, emb_img_height = (i, int(n / i))
                break

        if not emb_img_width and not emb_img_height:
            msg = 'Unable to determine factor pairs for embedding reshape. Choose a smaller convolution kernel size, ' \
                  'or a larger embedding dimension.'
            logger.info(msg)
            raise ValueError(msg)

        embedding_model_params['embed_image_width'] = emb_img_width
        embedding_model_params['embed_image_height'] = emb_img_height
        embedding_model_params['embed_image_depth'] = emb_img_depth

        # Calculate dense dimension
        embedding_model_params['dense_dim'] = (emb_img_width - (ksize - 1)) * (emb_img_height - (ksize - 1)) * nfilters

        self.low_memory = low_memory

        super().__init__(k=k, eta=eta, epochs=epochs,
                         batches_count=batches_count, seed=seed,
                         embedding_model_params=embedding_model_params,
                         optimizer=optimizer, optimizer_params=optimizer_params,
                         loss=loss, loss_params=loss_params,
                         regularizer=regularizer, regularizer_params=regularizer_params,
                         initializer=initializer, initializer_params=initializer_params,
                         verbose=verbose)

    def _initialize_parameters(self):
        """Initialize parameters of the model.

            This function creates and initializes entity and relation embeddings (with size k).
            If the graph is large, then it loads only the required entity embeddings (max:batch_size*2)
            and all relation embeddings.
            Overload this function if the parameters needs to be initialized differently.
        """

        if not self.dealing_with_large_graphs:

            with tf.variable_scope('meta'):
                self.tf_is_training = tf.Variable(False, trainable=False, name='is_training')
                self.set_training_true = tf.assign(self.tf_is_training, True)
                self.set_training_false = tf.assign(self.tf_is_training, False)

            nfilters = self.embedding_model_params['conv_filters']
            ninput = self.embedding_model_params['embed_image_depth']
            ksize = self.embedding_model_params['conv_kernel_size']
            dense_dim = self.embedding_model_params['dense_dim']

            self.ent_emb = tf.get_variable('ent_emb', shape=[len(self.ent_to_idx), self.k],
                                           initializer=self.initializer.get_tf_initializer(), dtype=tf.float32)
            self.rel_emb = tf.get_variable('rel_emb', shape=[len(self.rel_to_idx), self.k],
                                           initializer=self.initializer.get_tf_initializer(), dtype=tf.float32)

            self.conv2d_W = tf.get_variable('conv2d_weights', shape=[ksize, ksize, ninput, nfilters],
                                            initializer=tf.initializers.he_normal(seed=self.seed),
                                            dtype=tf.float32)
            self.conv2d_B = tf.get_variable('conv2d_bias', shape=[nfilters],
                                            initializer=tf.zeros_initializer(), dtype=tf.float32)

            self.dense_W = tf.get_variable('dense_weights', shape=[dense_dim, self.k],
                                           initializer=tf.initializers.he_normal(seed=self.seed),
                                           dtype=tf.float32)
            self.dense_B = tf.get_variable('dense_bias', shape=[self.k],
                                           initializer=tf.zeros_initializer(), dtype=tf.float32)

            if self.embedding_model_params['use_batchnorm']:

                emb_img_dim = self.embedding_model_params['embed_image_depth']

                self.bn_vars = {'batchnorm_input': {'beta': np.zeros(shape=[emb_img_dim]),
                                                    'gamma': np.ones(shape=[emb_img_dim]),
                                                    'moving_mean': np.zeros(shape=[emb_img_dim]),
                                                    'moving_variance': np.ones(shape=[emb_img_dim])},
                                'batchnorm_conv': {'beta': np.zeros(shape=[nfilters]),
                                                   'gamma': np.ones(shape=[nfilters]),
                                                   'moving_mean': np.zeros(shape=[nfilters]),
                                                   'moving_variance': np.ones(shape=[nfilters])},
                                'batchnorm_dense': {'beta': np.zeros(shape=[1]),       # shape = [1] for batch norm
                                                    'gamma': np.ones(shape=[1]),
                                                    'moving_mean': np.zeros(shape=[1]),
                                                    'moving_variance': np.ones(shape=[1])}}

            if self.embedding_model_params['use_bias']:
                self.bias = tf.get_variable('activation_bias', shape=[1, len(self.ent_to_idx)],
                                            initializer=tf.zeros_initializer(), dtype=tf.float32)

        else:
            raise NotImplementedError('ConvE not implemented when dealing with large graphs.')

    def _get_model_loss(self, dataset_iterator):
        """Get the current loss including loss due to regularization.
        This function must be overridden if the model uses combination of different losses (eg: VAE).

        Parameters
        ----------
        dataset_iterator : tf.data.Iterator
            Dataset iterator.

        Returns
        -------
        loss : tf.Tensor
            The loss value that must be minimized.
        """

        # training input placeholder
        self.x_pos_tf, self.y_true = dataset_iterator.get_next()

        # list of dependent ops that need to be evaluated before computing the loss
        dependencies = []

        # run the dependencies
        with tf.control_dependencies(dependencies):

            # look up embeddings from input training triples
            e_s_pos, e_p_pos, e_o_pos = self._lookup_embeddings(self.x_pos_tf)

            # Get positive predictions
            self.y_pred = self._fn(e_s_pos, e_p_pos, e_o_pos)

            # Label smoothing and/or weighting is applied within Loss class
            loss = self.loss.apply(self.y_true, self.y_pred)

            if self.regularizer is not None:
                loss += self.regularizer.apply([self.ent_emb, self.rel_emb])

            return loss

    def _save_trained_params(self):
        """After model fitting, save all the trained parameters in trained_model_params in some order.
        The order would be useful for loading the model.
        This method must be overridden if the model has any other parameters (apart from entity-relation embeddings).
        """

        params_dict = {}
        params_dict['ent_emb'] = self.sess_train.run(self.ent_emb)
        params_dict['rel_emb'] = self.sess_train.run(self.rel_emb)
        params_dict['conv2d_W'] = self.sess_train.run(self.conv2d_W)
        params_dict['conv2d_B'] = self.sess_train.run(self.conv2d_B)
        params_dict['dense_W'] = self.sess_train.run(self.dense_W)
        params_dict['dense_B'] = self.sess_train.run(self.dense_B)

        if self.embedding_model_params['use_batchnorm']:

            bn_dict = {}

            for scope in ['batchnorm_input', 'batchnorm_conv', 'batchnorm_dense']:

                variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
                variables = [x for x in variables if 'Adam' not in x.name]  # Filter out any Adam variables

                var_dict = {x.name.split('/')[-1].split(':')[0]: x for x in variables}
                bn_dict[scope] = {'beta': self.sess_train.run(var_dict['beta']),
                                  'gamma': self.sess_train.run(var_dict['gamma']),
                                  'moving_mean': self.sess_train.run(var_dict['moving_mean']),
                                  'moving_variance': self.sess_train.run(var_dict['moving_variance'])}

            params_dict['bn_vars'] = bn_dict

        if self.embedding_model_params['use_bias']:
            params_dict['bias'] = self.sess_train.run(self.bias)

        params_dict['output_mapping'] = self.output_mapping

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

        with tf.variable_scope('meta'):
            self.tf_is_training = tf.Variable(False, trainable=False, name='is_training')
            self.set_training_true = tf.assign(self.tf_is_training, True)
            self.set_training_false = tf.assign(self.tf_is_training, False)

        self.ent_emb = tf.Variable(self.trained_model_params['ent_emb'], dtype=tf.float32, name='ent_emb')
        self.rel_emb = tf.Variable(self.trained_model_params['rel_emb'], dtype=tf.float32, name='rel_emb')

        self.conv2d_W = tf.Variable(self.trained_model_params['conv2d_W'], dtype=tf.float32)
        self.conv2d_B = tf.Variable(self.trained_model_params['conv2d_B'], dtype=tf.float32)
        self.dense_W = tf.Variable(self.trained_model_params['dense_W'], dtype=tf.float32)
        self.dense_B = tf.Variable(self.trained_model_params['dense_B'], dtype=tf.float32)

        if self.embedding_model_params['use_batchnorm']:
            self.bn_vars = self.trained_model_params['bn_vars']

        if self.embedding_model_params['use_bias']:
            self.bias = tf.Variable(self.trained_model_params['bias'], dtype=tf.float32)

        self.output_mapping = self.trained_model_params['output_mapping']

    def _fn(self, e_s, e_p, e_o):
        r"""The ConvE scoring function.

            The function implements the scoring function as defined by
            .. math::

                f(vec(f([\overline{e_s};\overline{r_r}] * \Omega)) W ) e_o

            Additional details for equivalence of the models available in :cite:`Dettmers2016`.


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
            The operation corresponding to the ConvE scoring function.

        """

        def _dropout(X, rate):
            dropout_rate = tf.cond(self.tf_is_training, true_fn=lambda: tf.constant(rate),
                                   false_fn=lambda: tf.constant(0, dtype=tf.float32))
            out = tf.nn.dropout(X, rate=dropout_rate)
            return out

        def _batchnorm(X, key, axis):

            with tf.variable_scope(key, reuse=tf.AUTO_REUSE):
                x = tf.compat.v1.layers.batch_normalization(X, training=self.tf_is_training, axis=axis,
                                                            beta_initializer=tf.constant_initializer(
                                                                self.bn_vars[key]['beta']),
                                                            gamma_initializer=tf.constant_initializer(
                                                                self.bn_vars[key]['gamma']),
                                                            moving_mean_initializer=tf.constant_initializer(
                                                                self.bn_vars[key]['moving_mean']),
                                                            moving_variance_initializer=tf.constant_initializer(
                                                                self.bn_vars[key]['moving_variance']))
            return x

        # Inputs
        stacked_emb = tf.stack([e_s, e_p], axis=2, name='stacked_embeddings')
        self.inputs = tf.reshape(stacked_emb, name='embed_image',
                                 shape=[tf.shape(stacked_emb)[0], self.embedding_model_params['embed_image_height'],
                                        self.embedding_model_params['embed_image_width'], 1])

        x = self.inputs

        if self.embedding_model_params['use_batchnorm']:
            x = _batchnorm(x, key='batchnorm_input', axis=3)

        if not self.embedding_model_params['dropout_embed'] is None:
            x = _dropout(x, rate=self.embedding_model_params['dropout_embed'])

        # Convolution layer
        x = tf.nn.conv2d(x, self.conv2d_W, [1, 1, 1, 1], padding='VALID')

        if self.embedding_model_params['use_batchnorm']:
            x = _batchnorm(x, key='batchnorm_conv', axis=3)
        else:
            # Batch normalization will cancel out bias, so only add bias term if not using batchnorm
            x = tf.nn.bias_add(x, self.conv2d_B)

        x = tf.nn.relu(x, name='conv_relu')

        if not self.embedding_model_params['dropout_conv'] is None:
            x = _dropout(x, rate=self.embedding_model_params['dropout_conv'])

        # Dense layer
        x = tf.reshape(x, shape=[tf.shape(x)[0], self.embedding_model_params['dense_dim']])
        x = tf.matmul(x, self.dense_W)

        if self.embedding_model_params['use_batchnorm']:
            # Initializing batchnorm vars for dense layer with shape=[1] will still broadcast over the shape of
            # the specified axis, e.g. dense shape = [?, k], batchnorm on axis 1 will create k batchnorm vars.
            # This is layer normalization rather than batch normalization, so adding a dimension to keep batchnorm,
            # thus dense shape = [?, k, 1], batchnorm on axis 2.
            x = tf.expand_dims(x, -1)
            x = _batchnorm(x, key='batchnorm_dense', axis=2)
            x = tf.squeeze(x, -1)
        else:
            x = tf.nn.bias_add(x, self.dense_B)

        # Note: Reference ConvE implementation had dropout on dense layer before applying batch normalization.
        # This can cause variance shift and reduce model performance, so have moved it after as recommended in:
        # https://arxiv.org/abs/1801.05134
        if not self.embedding_model_params['dropout_dense'] is None:
            x = _dropout(x, rate=self.embedding_model_params['dropout_dense'])

        x = tf.nn.relu(x, name='dense_relu')
        x = tf.matmul(x, tf.transpose(self.ent_emb), name='matmul')

        if self.embedding_model_params['use_bias']:
            x = tf.add(x, self.bias, name='add_bias')

        self.scores = x

        return self.scores

    def get_embeddings(self, entities, embedding_type='entity'):
        """Get the embeddings of entities or relations.

        .. Note ::
            Use :meth:`ampligraph.utils.create_tensorboard_visualizations` to visualize the embeddings with TensorBoard.

        Parameters
        ----------
        entities : array-like, dtype=int, shape=[n]
            The entities (or relations) of interest. Element of the vector
            must be the original string literals, and not internal IDs.
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

    def fit(self, X, early_stopping=False, early_stopping_params={}):
        """Train a ConvE (with optional early stopping).

        The model is trained on a training set X using the training protocol
        described in :cite:`Dettmers2016`.

        Parameters
        ----------
        X : ndarray (shape [n, 3]) or object of ConvEDatasetAdapter
            Numpy array of training triples OR handle of Dataset adapter which would help retrieve data.
        early_stopping: bool
            Flag to enable early stopping (default:``False``)
        early_stopping_params: dictionary
            Dictionary of hyperparameters for the early stopping heuristics.

            The following string keys are supported:

                - **'x_valid'**: ndarray (shape [n, 3]) or object of AmpligraphDatasetAdapter :
                                 Numpy array of validation triples OR handle of Dataset adapter which
                                 would help retrieve data.
                - **'criteria'**: string : criteria for early stopping 'hits10', 'hits3', 'hits1' or 'mrr'(default).
                - **'x_filter'**: ndarray, shape [n, 3] : Positive triples to use as filter if a 'filtered' early
                                  stopping criteria is desired (i.e. filtered-MRR if 'criteria':'mrr').
                                  Note this will affect training time (no filter by default).
                                  If the filter has already been set in the adapter, pass True
                - **'burn_in'**: int : Number of epochs to pass before kicking in early stopping (default: 100).
                - **check_interval'**: int : Early stopping interval after burn-in (default:10).
                - **'stop_interval'**: int : Stop if criteria is performing worse over n consecutive checks (default: 3)
                - **'corruption_entities'**: List of entities to be used for corruptions. If 'all',
                  it uses all entities (default: 'all')
                - **'corrupt_side'**: Specifies which side to corrupt. 's', 'o', 's,o' (default). Note: ConvE does not
                    currently support 's+o' evaluation mode.

                Example: ``early_stopping_params={x_valid=X['valid'], 'criteria': 'mrr'}``

        """

        self.train_dataset_handle = None
        # try-except block is mainly to handle clean up in case of exception or manual stop in jupyter notebook
        try:
            if isinstance(X, np.ndarray):
                self.train_dataset_handle = OneToNDatasetAdapter(low_memory=self.low_memory)
                self.train_dataset_handle.set_data(X, 'train')
            elif isinstance(X, OneToNDatasetAdapter):
                self.train_dataset_handle = X
            else:
                msg = 'Invalid type for input X. Expected numpy.array or OneToNDatasetAdapter object, got {}'\
                    .format(type(X))
                logger.error(msg)
                raise ValueError(msg)

            # create internal IDs mappings
            self.rel_to_idx, self.ent_to_idx = self.train_dataset_handle.generate_mappings()

            if len(self.ent_to_idx) > ENTITY_THRESHOLD:
                self.dealing_with_large_graphs = True
                prefetch_batches = 0

                logger.warning('Your graph has a large number of distinct entities. '
                               'Found {} distinct entities'.format(len(self.ent_to_idx)))

                logger.warning('Changing the variable initialization strategy.')
                logger.warning('Changing the strategy to use lazy loading of variables...')

                if early_stopping:
                    raise Exception('Early stopping not supported for large graphs')

                if not isinstance(self.optimizer, SGDOptimizer):
                    raise Exception("This mode works well only with SGD optimizer with decay (read docs for details). "
                                    "Kindly change the optimizer and restart the experiment")

                raise NotImplementedError('ConvE not implemented when dealing with large graphs.')

                # CPU matrix of embeddings
                self.ent_emb_cpu = self.initializer.get_np_initializer(len(self.ent_to_idx), self.internal_k)

            self.train_dataset_handle.map_data()

            # This is useful when we re-fit the same model (e.g. retraining in model selection)
            if self.is_fitted:
                tf.reset_default_graph()
                self.rnd = check_random_state(self.seed)
                tf.random.set_random_seed(self.seed)

            self.sess_train = tf.Session(config=self.tf_config)

            batch_size = int(np.ceil(self.train_dataset_handle.get_size("train") / self.batches_count))
            self.batch_size = batch_size

            if len(self.ent_to_idx) > ENTITY_THRESHOLD:
                logger.warning('Only {} embeddings would be loaded in memory per batch...'.format(batch_size * 2))

            self._initialize_parameters()

            # Output mapping is dict of (s, p) to list of existing object triple indices
            self.output_mapping = self.train_dataset_handle.generate_output_mapping(dataset_type='train')
            self.train_dataset_handle.set_output_mapping(self.output_mapping)
            self.train_dataset_handle.generate_outputs(dataset_type='train', unique_pairs=True)
            train_iter = partial(self.train_dataset_handle.get_next_batch,
                                 batches_count=self.batches_count,
                                 dataset_type='train',
                                 use_filter=False,
                                 unique_pairs=True)

            dataset = tf.data.Dataset.from_generator(train_iter,
                                                     output_types=(tf.int32, tf.float32),
                                                     output_shapes=((None, 3), (None, len(self.ent_to_idx))))
            prefetch_batches = 5
            dataset = dataset.repeat().prefetch(prefetch_batches)
            dataset_iterator = dataset.make_one_shot_iterator()

            # init tf graph/dataflow for training
            # init variables (model parameters to be learned - i.e. the embeddings)
            if self.loss.get_state('require_same_size_pos_neg'):
                batch_size = batch_size * self.eta

            # Required for label smoothing
            self.loss._set_hyperparams('num_entities', len(self.ent_to_idx))

            loss = self._get_model_loss(dataset_iterator)

            # Add update_ops for batch normalization
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train = self.optimizer.minimize(loss)

            self.early_stopping_params = early_stopping_params

            # early stopping
            if early_stopping:
                self._initialize_early_stopping()

            self.sess_train.run(tf.tables_initializer())
            self.sess_train.run(tf.global_variables_initializer())
            self.sess_train.run(self.set_training_true)

            # Entity embeddings normalization
            normalize_ent_emb_op = self.ent_emb.assign(tf.clip_by_norm(self.ent_emb, clip_norm=1, axes=1))
            normalize_rel_emb_op = self.rel_emb.assign(tf.clip_by_norm(self.rel_emb, clip_norm=1, axes=1))

            if self.embedding_model_params.get('normalize_ent_emb', constants.DEFAULT_NORMALIZE_EMBEDDINGS):
                self.sess_train.run(normalize_rel_emb_op)
                self.sess_train.run(normalize_ent_emb_op)

            epoch_iterator_with_progress = tqdm(range(1, self.epochs + 1), disable=(not self.verbose), unit='epoch')

            for epoch in epoch_iterator_with_progress:
                losses = []
                for batch in range(1, self.batches_count + 1):
                    feed_dict = {}
                    self.optimizer.update_feed_dict(feed_dict, batch, epoch)

                    loss_batch, _ = self.sess_train.run([loss, train], feed_dict=feed_dict)

                    if np.isnan(loss_batch) or np.isinf(loss_batch):
                        msg = 'Loss is {}. Please change the hyperparameters.'.format(loss_batch)
                        logger.error(msg)
                        raise ValueError(msg)

                    losses.append(loss_batch)
                    if self.embedding_model_params.get('normalize_ent_emb', constants.DEFAULT_NORMALIZE_EMBEDDINGS):
                        self.sess_train.run(normalize_ent_emb_op)

                if self.verbose:
                    msg = 'Average Loss: {:10f}'.format(sum(losses) / (batch_size * self.batches_count))
                    if early_stopping and self.early_stopping_best_value is not None:
                        msg += ' — Best validation ({}): {:5f}'.format(self.early_stopping_criteria,
                                                                       self.early_stopping_best_value)

                    logger.debug(msg)
                    epoch_iterator_with_progress.set_description(msg)

                if early_stopping:

                    self.sess_train.run(self.set_training_false)
                    if self._perform_early_stopping_test(epoch):
                        self._end_training()
                        return
                    self.sess_train.run(self.set_training_true)

            self._save_trained_params()
            self._end_training()
        except BaseException as e:
            self._end_training()
            raise e

    def _initialize_eval_graph(self, mode='test'):
        """ Initialize the 1-N evaluation graph with the set protocol.

        Parameters
        ----------
        mode: string
            Indicates which data generator to use.

        Returns
        -------

        """

        logger.debug('Initializing eval graph [mode: {}]'.format(mode))

        test_generator = partial(self.eval_dataset_handle.get_next_batch,
                                 batches_count=-1,
                                 dataset_type=mode,
                                 use_filter=self.is_filtered,
                                 unique_pairs=False)

        dataset = tf.data.Dataset.from_generator(test_generator,
                                                 output_types=(tf.int32, tf.float32),
                                                 output_shapes=((None, 3), (None, len(self.ent_to_idx))))

        dataset = dataset.repeat()
        dataset = dataset.prefetch(5)
        dataset_iter = dataset.make_one_shot_iterator()

        self.X_test_tf, self.X_test_filter_tf = dataset_iter.get_next()

        if self.dealing_with_large_graphs:
            raise NotImplementedError('ConvE not implemented with large graphs (yet)')

        else:

            e_s, e_p, e_o = self._lookup_embeddings(self.X_test_tf)

            # Scores for all triples
            scores = tf.sigmoid(tf.squeeze(self._fn(e_s, e_p, e_o)), name='sigmoid_scores')

            # Score of positive triple
            self.score_positive = tf.gather(scores, indices=self.X_test_tf[:, 2], name='score_positive')

            # Scores for positive triples
            self.scores_filtered = tf.boolean_mask(scores, tf.cast(self.X_test_filter_tf, tf.bool))

            # Triple rank over all triples
            self.total_rank = tf.reduce_sum(tf.cast(scores >= self.score_positive, tf.int32))

            # Triple rank over positive triples
            self.filter_rank = tf.reduce_sum(tf.cast(self.scores_filtered >= self.score_positive, tf.int32))

            # Rank of triple, with other positives filtered out.
            self.rank = tf.subtract(self.total_rank, self.filter_rank, name='rank') + 1

            # NOTE: if having trouble with the above rank calculation, consider when test triple
            # has the highest score (total_rank=1, filter_rank=1)

    def _initialize_early_stopping(self):
        """Initializes and creates evaluation graph for early stopping.
        """

        try:
            self.x_valid = self.early_stopping_params['x_valid']
        except KeyError:
            msg = 'x_valid must be passed for early fitting.'
            logger.error(msg)
            raise KeyError(msg)

        # Set eval_dataset handler
        if isinstance(self.x_valid, np.ndarray):

            if self.x_valid.ndim <= 1 or (np.shape(self.x_valid)[1]) != 3:
                msg = 'Invalid size for input x_valid. Expected (n,3):  got {}'.format(np.shape(self.x_valid))
                logger.error(msg)
                raise ValueError(msg)

            if self.early_stopping_params['corrupt_side'] == 's+o':
                msg = "ConvE does not support `s+o` corruption strategy. Please change to: 's', 'o', or 's, o'"
                logger.error(msg)
                raise ValueError(msg)

            if self.early_stopping_params['corrupt_side'] in ['s,o', 's']:
                logger.warning('ConvE with early stopping is significantly slower with subject corruptions.'
                               'For best performance use ``corrupt_side="o"``.')

                # store the validation data in the data handler
            self.train_dataset_handle.set_data(self.x_valid, 'valid')
            self.eval_dataset_handle = self.train_dataset_handle
            logger.debug('Initialized eval_dataset from train_dataset using.')

        elif isinstance(self.x_valid, OneToNDatasetAdapter):

            if not self.eval_dataset_handle.data_exists('valid'):
                msg = 'Dataset `valid` has not been set in the DatasetAdapter.'
                logger.error(msg)
                raise ValueError(msg)

            self.eval_dataset_handle = self.x_valid
            logger.debug('Initialized eval_dataset from AmpligraphDatasetAdapter')

        else:
            msg = 'Invalid type for input X. Expected np.ndarray or OneToNDatasetAdapter object, \
                   got {}'.format(type(self.x_valid))
            logger.error(msg)
            raise ValueError(msg)

        self.early_stopping_criteria = self.early_stopping_params.get('criteria',
                                                                      constants.DEFAULT_CRITERIA_EARLY_STOPPING)

        if self.early_stopping_criteria not in ['hits10', 'hits1', 'hits3', 'mrr']:
            msg = 'Unsupported early stopping criteria.'
            logger.error(msg)
            raise ValueError(msg)

        self.early_stopping_best_value = None
        self.early_stopping_stop_counter = 0

        # Set filter
        if 'x_filter' in self.early_stopping_params.keys():

            # If the filter has already been set in the dataset adapter then just pass x_filter = True
            x_filter = self.early_stopping_params['x_filter']
            if isinstance(x_filter, np.ndarray):

                if x_filter.ndim <= 1 or (np.shape(x_filter)[1]) != 3:
                    msg = 'Invalid size for input x_valid. Expected (n,3):  got {}'.format(np.shape(x_filter))
                    logger.error(msg)
                    raise ValueError(msg)

                # set the filter triples in the data handler
                x_filter = to_idx(x_filter, ent_to_idx=self.ent_to_idx, rel_to_idx=self.rel_to_idx)
                self.eval_dataset_handle.set_filter(x_filter, mapped_status=True)

            # set the flag to perform filtering
            self.set_filter_for_eval()
        else:
            logger.debug('x_filter not found in early_stopping_params.')

        # initialize evaluation graph in validation mode i.e. to use validation set
        self._initialize_eval_graph('valid')

    def predict(self, X, from_idx=False):
        """Predict the scores of triples using a trained embedding model.
            The function returns raw scores generated by the model.

            .. note::

                To obtain probability estimates, calibrate the model with :func:`~ConvE.calibrate`,
                then call :func:`~ConvE.predict_proba`.

        Parameters
        ----------
        X : ndarray, shape [n, 3]
            The triples to score.
        from_idx : bool
            If True, will skip conversion to internal IDs. (default: False).

        Returns
        -------
        scores_predict : ndarray, shape [n]
            The predicted scores for input triples X.

        """
        if not self.is_fitted:
            msg = 'Model has not been fitted.'
            logger.error(msg)
            raise RuntimeError(msg)

        tf.reset_default_graph()
        self._load_model_from_trained_params()

        dataset_handle = OneToNDatasetAdapter(low_memory=self.low_memory)
        dataset_handle.use_mappings(self.rel_to_idx, self.ent_to_idx)
        dataset_handle.set_data(X, "test", mapped_status=from_idx)

        # Note: onehot outputs not required for prediction, but are part of the batch function
        dataset_handle.set_output_mapping(self.output_mapping)
        dataset_handle.generate_outputs(dataset_type='test', unique_pairs=False)
        self.eval_dataset_handle = dataset_handle

        self.rnd = check_random_state(self.seed)
        tf.random.set_random_seed(self.seed)
        self._initialize_eval_graph()

        with tf.Session(config=self.tf_config) as sess:

            sess.run(tf.tables_initializer())
            sess.run(tf.global_variables_initializer())
            sess.run(self.set_training_false)

            scores = []

            for i in tqdm(range(self.eval_dataset_handle.get_size('test'))):

                score = sess.run(self.score_positive)
                scores.append(score[0])

            return scores

    def get_ranks(self, dataset_handle):
        """ Used by evaluate_predictions to get the ranks for evaluation.

        Parameters
        ----------
        dataset_handle : Object of AmpligraphDatasetAdapter
                         This contains handles of the generators that would be used to get test triples and filters

        Returns
        -------
        ranks : ndarray, shape [n] or [n,2] depending on the value of use_default_protocol.
                An array of ranks of test triples.
        """

        if not self.is_fitted:
            msg = 'Model has not been fitted.'
            logger.error(msg)
            raise RuntimeError(msg)

        eval_protocol = self.eval_config.get('corrupt_side', constants.DEFAULT_PROTOCOL_EVAL)

        if 'o' in eval_protocol:
            object_ranks = self._get_object_ranks(dataset_handle)

        if 's' in eval_protocol:
            subject_ranks = self._get_subject_ranks(dataset_handle)

        if eval_protocol == 's,o':
            ranks = [[s, o] for s, o in zip(subject_ranks, object_ranks)]
        elif eval_protocol == 's':
            ranks = subject_ranks
        elif eval_protocol == 'o':
            ranks = object_ranks

        return ranks

    def _get_object_ranks(self, dataset_handle):
        """ Internal function for obtaining object ranks.

        Parameters
        ----------
        dataset_handle : Object of AmpligraphDatasetAdapter
                         This contains handles of the generators that would be used to get test triples and filters

        Returns
        -------
        ranks : ndarray, shape [n]
                An array of ranks of test triples.
        """

        self.eval_dataset_handle = dataset_handle

        # Load model parameters, build tf evaluation graph for predictions
        tf.reset_default_graph()
        self.rnd = check_random_state(self.seed)
        tf.random.set_random_seed(self.seed)
        self._load_model_from_trained_params()
        self._initialize_eval_graph()

        with tf.Session(config=self.tf_config) as sess:

            sess.run(tf.tables_initializer())
            sess.run(tf.global_variables_initializer())
            sess.run(self.set_training_false)

            ranks = []
            for _ in tqdm(range(self.eval_dataset_handle.get_size('test')), disable=(not self.verbose)):
                rank = sess.run(self.rank)
                ranks.append(rank)

            return np.array(ranks)

    def _initialize_eval_graph_subject(self, mode='test'):
        """ Initialize the graph for evaluating subject corruptions.

        Parameters
        ----------
        mode: string
            Indicates which data generator to use.

        Returns
        -------

        """

        logger.debug('Initializing eval graph for subject corruptions [mode: {}]'.format(mode))

        corruption_batch_size = constants.DEFAULT_SUBJECT_CORRUPTION_BATCH_SIZE

        test_generator = partial(self.eval_dataset_handle.get_next_batch_subject_corruptions,
                                 batch_size=corruption_batch_size,
                                 dataset_type=mode)

        dataset = tf.data.Dataset.from_generator(test_generator,
                                                 output_types=(tf.int32, tf.int32, tf.float32),
                                                 output_shapes=((None, 3), (None, 3), (None, len(self.ent_to_idx))))

        dataset = dataset.repeat()
        dataset = dataset.prefetch(5)
        dataset_iter = dataset.make_one_shot_iterator()

        self.X_test_tf, self.subject_corr, self.X_filter_tf = dataset_iter.get_next()

        e_s, e_p, e_o = self._lookup_embeddings(self.subject_corr)

        # Scores for all triples
        self.sigmoid_scores = tf.sigmoid(tf.squeeze(self._fn(e_s, e_p, e_o)), name='sigmoid_scores')

    def _get_subject_ranks(self, dataset_handle, corruption_batch_size=None):
        """ Internal function for obtaining subject ranks.

        This function performs subject corruptions. Output layer scores are accumulated in order to rank
        subject corruptions. This can cause high memory consumption, so a default subject corruption batch size
        is set in constants.py.

        Parameters
        ----------
        dataset_handle : Object of AmpligraphDatasetAdapter
                         This contains handles of the generators that would be used to get test triples and filters
        corruption_batch_size : int / None
                         Batch size for accumulating output layer scores for each input. The accumulated batch size
                         will be np.array shape=(corruption_batch_size, num_entities), and dtype=np.float32).
                         Default: 10000 has been set in constants.DEFAULT_SUBJECT_CORRUPTION_BATCH_SIZE.

        Returns
        -------
        ranks : ndarray, shape [n]
                An array of ranks of test triples.
        """

        self.eval_dataset_handle = dataset_handle

        # Load model parameters, build tf evaluation graph for predictions
        tf.reset_default_graph()
        self.rnd = check_random_state(self.seed)
        tf.random.set_random_seed(self.seed)
        self._load_model_from_trained_params()
        self._initialize_eval_graph_subject()

        if not corruption_batch_size:
            corruption_batch_size = constants.DEFAULT_SUBJECT_CORRUPTION_BATCH_SIZE

        num_entities = len(self.ent_to_idx)
        num_batch_per_relation = np.ceil(len(self.eval_dataset_handle.ent_to_idx) / corruption_batch_size)
        num_batches = int(num_batch_per_relation * len(self.eval_dataset_handle.rel_to_idx))

        with tf.Session(config=self.tf_config) as sess:

            sess.run(tf.tables_initializer())
            sess.run(tf.global_variables_initializer())
            sess.run(self.set_training_false)

            ranks = []
            # Accumulate scores from each index of the object in the output scores while corrupting subject
            scores_matrix_accum = []
            # Accumulate true/false statements from one-hot outputs while corrupting subject
            scores_filter_accum = []

            for _ in tqdm(range(num_batches), disable=(not self.verbose), unit='batch'):

                try:

                    X_test, scores_matrix, scores_filter = sess.run([self.X_test_tf, self.sigmoid_scores,
                                                                     self.X_filter_tf])

                    # Accumulate scores from X_test columns
                    scores_matrix_accum.append(scores_matrix[:, X_test[:, 2]])
                    scores_filter_accum.append(scores_filter[:, X_test[:, 2]])

                    num_rows_accum = np.sum([x.shape[0] for x in scores_matrix_accum])

                    if num_rows_accum == num_entities:
                        # When num rows accumulated equals num_entities, batch has finished a single subject corruption
                        # loop on a single relation

                        if len(X_test) == 0:
                            # If X_test is empty, reset accumulated scores and continue
                            scores_matrix_accum, scores_filter_accum = [], []
                            continue

                        scores_matrix = np.concatenate(scores_matrix_accum)
                        scores_filter = np.concatenate(scores_filter_accum)

                        for i, x in enumerate(X_test):
                            score_positive = scores_matrix[x[0], i]
                            idx_negatives = np.where(scores_filter[:, i] != 1)
                            score_negatives = scores_matrix[idx_negatives[0], i]
                            rank = np.sum(score_negatives >= score_positive) + 1
                            ranks.append(rank)

                        # Reset accumulators
                        scores_matrix_accum, scores_filter_accum = [], []

                except StopIteration:
                    break

            return np.array(ranks)
