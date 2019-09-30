from .models import *

class ConvE(EmbeddingModel):
    """ Convolutional 2D Knowledge Graph Embeddings

    The ConvE model :cite:`Dettmers2016`:

    .. math::

        f(vec(f([\overline{e_s};\overline{r_r}] * \omega)) W ) e_o

    Examples
    --------
    >>> import numpy as np
    >>> from ampligraph.latent_features import conve
    >>> model = conve(batches_count=1, seed=555, epochs=20, k=10,       # TODO
    >>>              loss='bce', loss_params={},
    >>>              regularizer='LP', regularizer_params={'lambda':0.1})
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
    >>> model.predict(np.array([['f', 'y', 'e'], ['b', 'y', 'd']]), get_ranks=True)
    ([-0.06213863, 0.01563319], [13, 3])
    >>> model.get_embeddings(['f','e'], embedding_type='entity')
        array([[ 0.17335348,  0.15826802,  0.24862595,  0.21404941, -0.00968813,
         0.06185953, -0.24956754,  0.01114257, -0.1038138 ,  0.40461722,
        -0.12298391, -0.10997348,  0.28220937,  0.34238952,  0.58363295,
         0.03315138, -0.37830347,  0.13480346,  0.49922466, -0.26328272],
        [-0.19098252,  0.20133668,  0.04635337,  0.4364128 ,  0.07014864,
         0.5713923 ,  0.28131518,  0.31721675, -0.06636801,  0.2848032 ,
        -0.2121708 ,  0.56917167, -0.05311433,  0.03093261,  0.01571475,
        -0.11373658,  0.29417998,  0.34896123,  0.22993243, -0.5499186 ]],
        dtype=float32)

    """

    def __init__(self,
                 k=DEFAULT_EMBEDDING_SIZE,
                 eta=DEFAULT_ETA,
                 epochs=DEFAULT_EPOCH,
                 batches_count=DEFAULT_BATCH_COUNT,
                 seed=DEFAULT_SEED,
                 embedding_model_params={'conv_filters': 32,
                                         'conv_kernel_size': 3,
                                         'dropout_embed': 0.2,
                                         'dropout_conv': 0.3,
                                         'dropout_dense': 0.2,
                                         'use_bias': True,
                                         'use_batchnorm': True,
                                         'corrupt_sides': 'o',
                                         'is_trainable': True,
                                         'checkerboard': False},
                 optimizer=DEFAULT_OPTIM,
                 optimizer_params={'lr': DEFAULT_LR},
                 loss='bce',
                 loss_params={'scoring_strategy': '1-N',
                              'label_weighting': False,
                              'label_smoothing': 0.1},
                 regularizer=DEFAULT_REGULARIZER,
                 regularizer_params={},
                 initializer=DEFAULT_INITIALIZER,
                 initializer_params={'uniform': DEFAULT_XAVIER_IS_UNIFORM},
                 low_memory=False,
                 verbose=DEFAULT_VERBOSE):
        """Initialize an EmbeddingModel

        Also creates a new Tensorflow session for training.

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
            ConvE-specific hyperparams:
            - **conv_filters** - Number of convolution feature maps.
            - **conv_kernel_size** - Convolution kernel size.
            - **dropout_embed** - Dropout on the embedding layer.
            - **dropout_conv** -  Dropout on the convolution maps.
            - **dropout_dense** - Dropout on the dense layer.
            - **use_bias** - Use bias layer.
            - **use_batchnorm** - Use batch normalization after input, convolution, and dense layers.

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

            - ``bce``  the model will use binary cross entropy loss function.

        loss_params : dict
            Dictionary of loss-specific hyperparameters. See :ref:`loss functions <loss>`
            documentation for additional details.

            Supported keys:

            - **'lr'** (float): learning rate (used by all the optimizers). Default: 0.1.
            - **'momentum'** (float): learning momentum (only used when ``optimizer=momentum``). Default: 0.9.
            - **'label_smoothing'** (float): applies label smoothing to onehot outputs. Default: None.
            - **'label_weighting'** (bool): applies label weighting to onehot outputs. Default: False

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
        """

        # Find factor pairs (i,j) of concatenated embedding dimensions, where min(i,j) >= conv_kernel_size
        if embedding_model_params['checkerboard']:
            n = k*2
            emb_img_depth = 1
        else:
            n = k
            emb_img_depth = 2

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

        # self.saver = tf.train.Saver()

        if not self.dealing_with_large_graphs:

            with tf.variable_scope('meta'):
                self.tf_is_training = tf.Variable(False, trainable=False, name='is_training')
                self.set_training_true = tf.assign(self.tf_is_training, True)
                self.set_training_false = tf.assign(self.tf_is_training, False)

            is_trainable = True # not self.embedding_model_params['is_trainable']
            nfilters = self.embedding_model_params['conv_filters']
            ninput = self.embedding_model_params['embed_image_depth']
            ksize = self.embedding_model_params['conv_kernel_size']
            dense_dim = self.embedding_model_params['dense_dim']

            self.ent_emb = tf.get_variable('ent_emb', shape=[len(self.ent_to_idx), self.k],
                                           initializer=self.initializer.get_tf_initializer(), dtype=tf.float32)
            self.rel_emb = tf.get_variable('rel_emb', shape=[len(self.rel_to_idx), self.k],
                                           initializer=self.initializer.get_tf_initializer(), dtype=tf.float32)

            self.conv2d_W = tf.get_variable('conv2d_weights', shape=[ksize, ksize, ninput, nfilters], trainable=is_trainable,
                                            initializer=tf.initializers.he_normal(seed=self.seed),
                                            dtype=tf.float32)
            self.conv2d_B = tf.get_variable('conv2d_bias', shape=[nfilters], trainable=is_trainable,
                                            initializer=tf.zeros_initializer(), dtype=tf.float32)

            self.dense_W = tf.get_variable('dense_weights', shape=[dense_dim, self.k],
                                           trainable=is_trainable,
                                           initializer=tf.initializers.he_normal(seed=self.seed),
                                           dtype=tf.float32)
            self.dense_B = tf.get_variable('dense_bias', shape=[self.k], trainable=is_trainable,
                                           initializer=tf.zeros_initializer(), dtype=tf.float32)

            if self.embedding_model_params['use_batchnorm']:

                if self.embedding_model_params['checkerboard']:
                    bn_input_shape = [1]
                else:
                    bn_input_shape = [2]

                self.bn_input_beta = tf.get_variable('bn_input_beta', shape=bn_input_shape, dtype=tf.float32,
                                                     trainable=is_trainable, initializer=tf.zeros_initializer())
                self.bn_input_gamma = tf.get_variable('bn_input_gamma', shape=bn_input_shape, dtype=tf.float32,
                                                      trainable=is_trainable, initializer=tf.ones_initializer())
                self.bn_conv_beta = tf.get_variable('bn_conv_beta', shape=[nfilters], dtype=tf.float32,
                                                    trainable=is_trainable, initializer=tf.zeros_initializer())
                self.bn_conv_gamma = tf.get_variable('bn_conv_gamma', shape=[nfilters], dtype=tf.float32,
                                                     trainable=is_trainable, initializer=tf.ones_initializer())
                self.bn_dense_beta = tf.get_variable('bn_dense_beta', shape=[1], dtype=tf.float32,
                                                     trainable=is_trainable, initializer=tf.zeros_initializer())
                self.bn_dense_gamma = tf.get_variable('bn_dense_gamma', shape=[1], dtype=tf.float32,
                                                      trainable=is_trainable, initializer=tf.ones_initializer())

            if self.embedding_model_params['use_bias']:
                self.bias = tf.get_variable('activation_bias', shape=[1, len(self.ent_to_idx)],
                                            initializer=tf.zeros_initializer(), trainable=is_trainable,
                                            dtype=tf.float32)

        else:
            raise NotImplementedError('ConvE not implemented when dealing with large graphs (yet)')

    def _dense(self, X):
        """
        Internal function to create dense layer. Note this does not create new variables.
        Parameters
        ----------
        X: tf.Variable

        Returns
        -------

        """
        out = tf.matmul(X, self.dense_W)
        out = tf.nn.bias_add(out, self.dense_B)
        return out

    def _conv2d(self, X):
        """
        Internal function to create conv2d layer. Note this does not create new variables.
        Parameters
        ----------
        X

        Returns
        -------

        """
        out = tf.nn.conv2d(X, self.conv2d_W, [1, 1, 1, 1], padding='VALID')
        out = tf.nn.bias_add(out, self.conv2d_B)
        return out

    def _batch_norm(self, X, beta, gamma, axes, name=None):
        """ Internal function to create batch normalization layer. Note this does not create new variables.

        Parameters
        ----------
        X : tf.Variable
        beta : tf.Variable
        gamma : tf.Variable
        axes : axes to perform normalization over
        name : string

        Returns
        -------

        """

        batch_mean, batch_var = tf.nn.moments(X, axes, name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(self.tf_is_training,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))

        normed = tf.nn.batch_normalization(X, mean, var, beta, gamma, 1e-3, name=name)

        return normed

    def _dropout(self, X, rate):
        """
        Internal function to create dropout layer.
        Parameters
        ----------
        X
        rate

        Returns
        -------

        """

        dropout_rate = tf.cond(self.tf_is_training,
                               true_fn=lambda: tf.constant(rate),
                               false_fn=lambda: tf.constant(0, dtype=tf.float32))
        out = tf.nn.dropout(X, rate=dropout_rate, name='dropout_dense')
        return out

    def _get_model_loss(self, dataset_iterator):
        """Get the current loss including loss due to regularization.
        This function must be overridden if the model uses combination of different losses(eg: VAE).

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
        x_pos_tf, self.y_true = dataset_iterator.get_next()

        # list of dependent ops that need to be evaluated before computing the loss
        dependencies = []

        # if the graph is large
        if self.dealing_with_large_graphs:
            raise NotImplementedError('ConvE not implemented when dealing with large graphs (yet)')

        # run the dependencies
        with tf.control_dependencies(dependencies):

            # look up embeddings from input training triples
            e_s_pos, e_p_pos, e_o_pos = self._lookup_embeddings(x_pos_tf)

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
        if not self.dealing_with_large_graphs:

            params_dict = {}
            params_dict['ent_emb'] = self.sess_train.run(self.ent_emb)
            params_dict['rel_emb'] = self.sess_train.run(self.rel_emb)
            params_dict['conv2d_W'] = self.sess_train.run(self.conv2d_W)
            params_dict['conv2d_B'] = self.sess_train.run(self.conv2d_B)
            params_dict['dense_W'] = self.sess_train.run(self.dense_W)
            params_dict['dense_B'] = self.sess_train.run(self.dense_B)

            if self.embedding_model_params['use_batchnorm']:
                params_dict['bn_input_beta'] = self.sess_train.run(self.bn_input_beta)
                params_dict['bn_input_gamma'] = self.sess_train.run(self.bn_input_gamma)
                params_dict['bn_conv_beta'] = self.sess_train.run(self.bn_conv_beta)
                params_dict['bn_conv_gamma'] = self.sess_train.run(self.bn_conv_gamma)
                params_dict['bn_dense_beta'] = self.sess_train.run(self.bn_dense_beta)
                params_dict['bn_dense_gamma'] = self.sess_train.run(self.bn_dense_gamma)

            if self.embedding_model_params['use_bias']:
                params_dict['bias'] = self.sess_train.run(self.bias)

            params_dict['output_mapping'] = self.output_mapping

            self.trained_model_params = params_dict

        else:
            raise NotImplementedError('ConvE not implemented when dealing with large graphs (yet)')

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
                self.bn_input_beta = tf.Variable(self.trained_model_params['bn_input_beta'], dtype=tf.float32)
                self.bn_input_gamma = tf.Variable(self.trained_model_params['bn_input_gamma'], dtype=tf.float32)
                self.bn_conv_beta = tf.Variable(self.trained_model_params['bn_conv_beta'], dtype=tf.float32)
                self.bn_conv_gamma = tf.Variable(self.trained_model_params['bn_conv_gamma'], dtype=tf.float32)
                self.bn_dense_beta = tf.Variable(self.trained_model_params['bn_dense_beta'], dtype=tf.float32)
                self.bn_dense_gamma = tf.Variable(self.trained_model_params['bn_dense_gamma'], dtype=tf.float32)

            if self.embedding_model_params['use_bias']:
                self.bias = tf.Variable(self.trained_model_params['bias'], dtype=tf.float32)

            self.output_mapping = self.trained_model_params['output_mapping']

        else:
            raise NotImplementedError('ConvE not implemented when dealing with large graphs (yet)')


    def _fn(self, e_s, e_p, e_o):
        """The ConvE scoring function.

            The function implements the scoring function as defined by
            .. math::

                f(vec(f([\overline{e_s};\overline{r_r}] * \omega)) W ) e_o

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

        # Inputs
        if self.embedding_model_params['checkerboard']:
            stacked_emb = tf.stack([e_s, e_p], axis=2, name='stacked_embeddings')
            self.inputs = tf.reshape(stacked_emb, shape=[tf.shape(stacked_emb)[0], self.embedding_model_params['embed_image_height'],
                                     self.embedding_model_params['embed_image_width'], 1], name='embed_image')
        else:
            e_s_img = tf.reshape(e_s, shape=[tf.shape(e_s)[0], self.embedding_model_params['embed_image_height'],
                                     self.embedding_model_params['embed_image_width']])
            e_p_img = tf.reshape(e_s, shape=[tf.shape(e_p)[0], self.embedding_model_params['embed_image_height'],
                                     self.embedding_model_params['embed_image_width']])
            self.inputs = tf.stack([e_s_img, e_p_img], axis=3, name='embed_image')

        x = self.inputs

        if self.embedding_model_params['use_batchnorm']:
            x = self._batch_norm(x, self.bn_input_beta, self.bn_input_gamma, axes=[0, 1, 2], name='input')

        if not self.embedding_model_params['dropout_embed'] is None:
            x = self._dropout(x, rate=self.embedding_model_params['dropout_embed'])

        x = self._conv2d(x)

        if self.embedding_model_params['use_batchnorm']:
            x = self._batch_norm(x, self.bn_conv_beta, self.bn_conv_gamma, axes=[0, 1, 2], name='conv')

        x = tf.nn.relu(x, name='conv_relu')

        if not self.embedding_model_params['dropout_conv'] is None:
            x = self._dropout(x, rate=self.embedding_model_params['dropout_conv'])

        x = tf.reshape(x, shape=[tf.shape(x)[0], self.embedding_model_params['dense_dim']])
        x = self._dense(x)

        if not self.embedding_model_params['dropout_dense'] is None:
            x = self._dropout(x, rate=self.embedding_model_params['dropout_dense'])

        if self.embedding_model_params['use_batchnorm']:
            x = self._batch_norm(x, self.bn_dense_beta, self.bn_dense_gamma, axes=[0], name='dense')

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

    def _training_data_generator(self):
        """Generates the training data for ConvE model.
        """

        logger.info('Initializing training data generator.')
        logger.info('Size of training data: {}'.format(self.train_dataset_handle.get_size('train')))

        # create iterator to iterate over the train batches
        batch_iterator = iter(self.train_dataset_handle.get_next_train_batch(self.batch_size, 'train'))

        for i in range(self.batches_count):

            try:
                out, out_onehot = next(batch_iterator)
                # logger.info('gen - batch {} out {} onehot {}'.format(i, out.shape, out_onehot.shape))

                # If large graph, load batch_size*2 entities on GPU memory
                if self.dealing_with_large_graphs:
                    raise NotImplementedError('ConvE not implemented when dealing with large graphs (yet)')

                yield out, out_onehot
            except StopIteration:
                logger.info('Hit stop iteration at batch {}'.format(i))
                break

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
                - **'corrupt_side'**: Specifies which side to corrupt. 's', 'o', 's+o' (default)

                Example: ``early_stopping_params={x_valid=X['valid'], 'criteria': 'mrr'}``

        """

        self.train_dataset_handle = None
        # try-except block is mainly to handle clean up in case of exception or manual stop in jupyter notebook
        try:
            if isinstance(X, np.ndarray):
                # Adapt the numpy data in the internal format - to generalize
                self.train_dataset_handle = ConvEDatasetAdapter(low_memory=self.low_memory)
                self.train_dataset_handle.set_data(X, "train")
            elif isinstance(X, AmpligraphDatasetAdapter):
                self.train_dataset_handle = X
            else:
                msg = 'Invalid type for input X. Expected ndarray/AmpligraphDataset object, got {}'.format(type(X))
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
            self.train_dataset_handle.generate_onehot_outputs(dataset_type='train')

            dataset = tf.data.Dataset.from_generator(self._training_data_generator,
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

            if self.embedding_model_params.get('normalize_ent_emb', DEFAULT_NORMALIZE_EMBEDDINGS):
                self.sess_train.run(normalize_rel_emb_op)
                self.sess_train.run(normalize_ent_emb_op)

            epoch_iterator_with_progress = tqdm(range(1, self.epochs + 1), disable=(not self.verbose), unit='epoch')

            for epoch in epoch_iterator_with_progress:
                losses = []
                for batch in range(1, self.batches_count + 1):
                    feed_dict = {}
                    self.optimizer.update_feed_dict(feed_dict, batch, epoch)

                    if self.dealing_with_large_graphs:
                        raise NotImplementedError('ConvE not implemented when dealing with large graphs (yet)')
                    else:
                        loss_batch, _ = self.sess_train.run([loss, train], feed_dict=feed_dict)

                    if np.isnan(loss_batch) or np.isinf(loss_batch):
                        msg = 'Loss is {}. Please change the hyperparameters.'.format(loss_batch)
                        logger.error(msg)
                        raise ValueError(msg)

                    losses.append(loss_batch)
                    if self.embedding_model_params.get('normalize_ent_emb', DEFAULT_NORMALIZE_EMBEDDINGS):
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

    def _test_generator(self, mode):
        """Generates the test/validation data. If filter_triples are passed, then it returns the False Negatives
           that could be present in the generated corruptions.

           If we are dealing with large graphs, then along with the above, this method returns the idx of the
           entities present in the batch and their embeddings.
        """

        logger.debug('Initializing test generator [Mode: {}, Filtered: {}]'.format(mode, self.is_filtered))

        if self.is_filtered:
            test_generator = partial(self.eval_dataset_handle.get_next_batch_with_filter, batch_size=1,
                                     dataset_type=mode)
        else:
            test_generator = partial(self.eval_dataset_handle.get_next_eval_batch, batch_size=1, dataset_type=mode)

        batch_iterator = iter(test_generator())

        while True:
            try:
                out, out_onehot_filter = next(batch_iterator)
            except StopIteration:
                break
            else:
                yield out, out_onehot_filter

    def _initialize_eval_graph(self, mode="test", scoring_strategy='1-1'):
        """ Initialize the evaluation graph with the provided scoring strategy.

        Parameters
        ----------
        mode: string
            Indicates which data generator to use.
        scoring_strategy: string
            Valid options are '1-1' (Default) or '1-N'.

        Returns
        -------

        """

        logger.info('Initializing evaluation graph [{}], scoring strategy {}'.format(mode, scoring_strategy))

        if scoring_strategy == '1-1':
            self._initialize_eval_graph_1_1()
        elif scoring_strategy == '1-N':
            self._initialize_eval_graph_1_N()
        else:
            raise ValueError('Invalid scoring strategy {}, please use either `1-1` or `1-N`.'.format(scoring_strategy))



    def _initialize_eval_graph_1_1(self, mode="test"):
        """Initialize the evaluation graph for 1-1 scoring (i.e. the standard evaluation protocol).

        Parameters
        ----------
        mode: string
            Indicates which data generator to use.
        """

        # Use a data generator which returns a test triple along with the subjects and objects indices for filtering
        # The last two data are used if the graph is large. They are the embeddings of the entities that must be
        # loaded on the GPU before scoring and the indices of those embeddings.
        dataset = tf.data.Dataset.from_generator(partial(self._test_generator, mode=mode),
                                                 output_types=(tf.int32, tf.int32, tf.int32, tf.float32, tf.int32),
                                                 output_shapes=((1, 3), (None, 1), (None, 1),
                                                                (None, self.internal_k), (None, 1)))
        dataset = dataset.repeat()
        dataset = dataset.prefetch(1)
        dataset_iter = dataset.make_one_shot_iterator()
        self.X_test_tf, indices_obj, indices_sub, entity_embeddings, unique_ent = dataset_iter.get_next()

        use_default_protocol = self.eval_config.get('default_protocol', DEFAULT_PROTOCOL_EVAL)
        corrupt_side = self.eval_config.get('corrupt_side', DEFAULT_CORRUPT_SIDE_EVAL)
        # Dependencies that need to be run before scoring
        test_dependency = []
        # For large graphs
        if self.dealing_with_large_graphs:
            # Add a dependency to load the embeddings on the GPU
            init_ent_emb_batch = self.ent_emb.assign(entity_embeddings, use_locking=True)
            test_dependency.append(init_ent_emb_batch)

            # Add a dependency to create lookup tables(for remapping the entity indices to the order of variables on GPU
            self.sparse_mappings = tf.contrib.lookup.MutableDenseHashTable(key_dtype=tf.int32,
                                                                           value_dtype=tf.int32,
                                                                           default_value=-1,
                                                                           empty_key=-2,
                                                                           deleted_key=-1)

            insert_lookup_op = self.sparse_mappings.insert(unique_ent,
                                                           tf.reshape(tf.range(tf.shape(unique_ent)[0],
                                                                      dtype=tf.int32), (-1, 1)))
            test_dependency.append(insert_lookup_op)

            # Execute the dependency
            with tf.control_dependencies(test_dependency):
                # Compute scores for positive - single triple
                e_s, e_p, e_o = self._lookup_embeddings(self.X_test_tf)
                self.score_positive = tf.squeeze(self._fn(e_s, e_p, e_o))

                # Generate corruptions in batches
                self.corr_batches_count = int(np.ceil(len(self.ent_to_idx) / self.batch_size))

                # Corruption generator -
                # returns corruptions and their corresponding embeddings that need to be loaded on the GPU
                corruption_generator = tf.data.Dataset.from_generator(self._generate_corruptions_for_large_graphs,
                                                                      output_types=(tf.int32, tf.float32),
                                                                      output_shapes=((None, 1),
                                                                                     (None, self.internal_k)))

                corruption_generator = corruption_generator.repeat()
                corruption_generator = corruption_generator.prefetch(0)

                corruption_iter = corruption_generator.make_one_shot_iterator()

                # Create tensor arrays for storing the scores of subject and object evals
                scores_predict_s_corruptions = tf.TensorArray(dtype=tf.float32, size=(len(self.ent_to_idx)))
                scores_predict_o_corruptions = tf.TensorArray(dtype=tf.float32, size=(len(self.ent_to_idx)))

                def loop_cond(i,
                              scores_predict_s_corruptions_in,
                              scores_predict_o_corruptions_in):
                    return i < self.corr_batches_count

                def compute_score_corruptions(i,
                                              scores_predict_s_corruptions_in,
                                              scores_predict_o_corruptions_in):
                    corr_dependency = []
                    corr_batch, entity_embeddings_corrpt = corruption_iter.get_next()
                    # if self.dealing_with_large_graphs: #for debugging
                    # Add dependency to load the embeddings
                    init_ent_emb_corrpt = self.ent_emb.assign(entity_embeddings_corrpt, use_locking=True)
                    corr_dependency.append(init_ent_emb_corrpt)

                    # Add dependency to remap the indices to the corresponding indices on the GPU
                    insert_lookup_op2 = self.sparse_mappings.insert(corr_batch,
                                                                    tf.reshape(tf.range(tf.shape(corr_batch)[0],
                                                                                        dtype=tf.int32),
                                                                               (-1, 1)))
                    corr_dependency.append(insert_lookup_op2)
                    # end if

                    # Execute the dependency
                    with tf.control_dependencies(corr_dependency):
                        emb_corr = tf.squeeze(self._entity_lookup(corr_batch))
                        if corrupt_side == 's+o' or corrupt_side == 's':
                            # compute and store the scores batch wise
                            scores_predict_s_c = self._fn(emb_corr, e_p, e_o)
                            scores_predict_s_corruptions_in = \
                                scores_predict_s_corruptions_in.scatter(tf.squeeze(corr_batch),
                                                                        tf.squeeze(scores_predict_s_c))

                        if corrupt_side == 's+o' or corrupt_side == 'o':
                            scores_predict_o_c = self._fn(e_s, e_p, emb_corr)
                            scores_predict_o_corruptions_in = \
                                scores_predict_o_corruptions_in.scatter(tf.squeeze(corr_batch),
                                                                        tf.squeeze(scores_predict_o_c))

                    return i + 1, scores_predict_s_corruptions_in, scores_predict_o_corruptions_in

                # compute the scores for all the corruptions
                counter, scores_predict_s_corr_out, scores_predict_o_corr_out = \
                    tf.while_loop(loop_cond,
                                  compute_score_corruptions,
                                  (0,
                                   scores_predict_s_corruptions,
                                   scores_predict_o_corruptions),
                                  back_prop=False,
                                  parallel_iterations=1)

                if corrupt_side == 's+o' or corrupt_side == 's':
                    subj_corruption_scores = scores_predict_s_corr_out.stack()

                if corrupt_side == 's+o' or corrupt_side == 'o':
                    obj_corruption_scores = scores_predict_o_corr_out.stack()

                if corrupt_side == 's+o':
                    self.scores_predict = tf.concat([obj_corruption_scores, subj_corruption_scores], axis=0)
                elif corrupt_side == 'o':
                    self.scores_predict = obj_corruption_scores
                else:
                    self.scores_predict = subj_corruption_scores

        else:
            # Rather than generating corruptions in batches do it at once on the GPU for small or medium sized graphs
            all_entities_np = np.arange(len(self.ent_to_idx))

            corruption_entities = self.eval_config.get('corruption_entities', DEFAULT_CORRUPTION_ENTITIES)

            if corruption_entities == 'all':
                corruption_entities = all_entities_np
            elif isinstance(corruption_entities, np.ndarray):
                corruption_entities = corruption_entities
            else:
                msg = 'Invalid type for corruption entities.'
                logger.error(msg)
                raise ValueError(msg)

            # Entities that must be used while generating corruptions
            self.corruption_entities_tf = tf.constant(corruption_entities, dtype=tf.int32)

            corrupt_side = self.eval_config.get('corrupt_side', DEFAULT_CORRUPT_SIDE_EVAL)
            # Generate corruptions
            self.out_corr = generate_corruptions_for_eval(self.X_test_tf,
                                                          self.corruption_entities_tf,
                                                          corrupt_side)

            # Compute scores for negatives
            e_s, e_p, e_o = self._lookup_embeddings(self.out_corr)

            # NB: This is the only difference with the overriden inherited function
            scores = tf.sigmoid(tf.squeeze(self._fn(e_s, e_p, e_o)))
            # Score of positive triple
            self.score_positive = tf.gather(scores, indices=self.X_test_tf[:, 2], name='score_positive')

            # Compute scores for positive
            e_s, e_p, e_o = self._lookup_embeddings(self.X_test_tf)
            self.score_positive = tf.squeeze(self._fn(e_s, e_p, e_o))

            use_default_protocol = self.eval_config.get('default_protocol', DEFAULT_PROTOCOL_EVAL)

            if use_default_protocol:
                obj_corruption_scores = tf.slice(self.scores_predict,
                                                 [0],
                                                 [tf.shape(self.scores_predict)[0] // 2])

                subj_corruption_scores = tf.slice(self.scores_predict,
                                                  [tf.shape(self.scores_predict)[0] // 2],
                                                  [tf.shape(self.scores_predict)[0] // 2])

        # this is to remove the positives from corruptions - while ranking with filter
        positives_among_obj_corruptions_ranked_higher = tf.constant(0, dtype=tf.int32)
        positives_among_sub_corruptions_ranked_higher = tf.constant(0, dtype=tf.int32)

        if self.is_filtered:
            # If a list of specified entities were used for corruption generation
            if isinstance(self.eval_config.get('corruption_entities',
                                               DEFAULT_CORRUPTION_ENTITIES), np.ndarray):
                corruption_entities = self.eval_config.get('corruption_entities',
                                                           DEFAULT_CORRUPTION_ENTITIES).astype(np.int32)
                if corruption_entities.ndim == 1:
                    corruption_entities = np.expand_dims(corruption_entities, 1)
                # If the specified key is not present then it would return the length of corruption_entities
                corruption_mapping = tf.contrib.lookup.MutableDenseHashTable(key_dtype=tf.int32,
                                                                             value_dtype=tf.int32,
                                                                             default_value=len(corruption_entities),
                                                                             empty_key=-2,
                                                                             deleted_key=-1)

                insert_lookup_op = corruption_mapping.insert(corruption_entities,
                                                             tf.reshape(tf.range(tf.shape(corruption_entities)[0],
                                                                                 dtype=tf.int32), (-1, 1)))

                with tf.control_dependencies([insert_lookup_op]):
                    # remap the indices of objects to the smaller set of corruptions
                    indices_obj = corruption_mapping.lookup(indices_obj)
                    # mask out the invalid indices (i.e. the entities that were not in corruption list
                    indices_obj = tf.boolean_mask(indices_obj, indices_obj < len(corruption_entities))
                    # remap the indices of subject to the smaller set of corruptions
                    indices_sub = corruption_mapping.lookup(indices_sub)
                    # mask out the invalid indices (i.e. the entities that were not in corruption list
                    indices_sub = tf.boolean_mask(indices_sub, indices_sub < len(corruption_entities))

            # get the scores of positives present in corruptions
            if use_default_protocol:
                scores_pos_obj = tf.gather(obj_corruption_scores, indices_obj)
                scores_pos_sub = tf.gather(subj_corruption_scores, indices_sub)
            else:
                scores_pos_obj = tf.gather(self.scores_predict, indices_obj)
                if corrupt_side == 's+o':
                    scores_pos_sub = tf.gather(self.scores_predict, indices_sub + len(corruption_entities))
                else:
                    scores_pos_sub = tf.gather(self.scores_predict, indices_sub)
            # compute the ranks of the positives present in the corruptions and
            # see how many are ranked higher than the test triple
            if corrupt_side == 's+o' or corrupt_side == 'o':
                positives_among_obj_corruptions_ranked_higher = tf.reduce_sum(
                    tf.cast(scores_pos_obj >= self.score_positive, tf.int32))
            if corrupt_side == 's+o' or corrupt_side == 's':
                positives_among_sub_corruptions_ranked_higher = tf.reduce_sum(
                    tf.cast(scores_pos_sub >= self.score_positive, tf.int32))

        # compute the rank of the test triple and subtract the positives(from corruptions) that are ranked higher
        if use_default_protocol:
            self.rank = tf.stack([tf.reduce_sum(tf.cast(
                subj_corruption_scores >= self.score_positive,
                tf.int32)) + 1 - positives_among_sub_corruptions_ranked_higher,
                tf.reduce_sum(tf.cast(obj_corruption_scores >= self.score_positive,
                                      tf.int32)) + 1 - positives_among_obj_corruptions_ranked_higher], 0)
        else:
            self.rank = tf.reduce_sum(tf.cast(
                self.scores_predict >= self.score_positive,
                tf.int32)) + 1 - positives_among_sub_corruptions_ranked_higher - \
                positives_among_obj_corruptions_ranked_higher


    def _initialize_eval_graph_1_N(self, mode='test'):
        """Initialize the evaluation graph for 1-N scoring.

        Parameters
        ----------
        mode: string
            Indicates which data generator to use.
        """

        logger.debug('Initializing eval graph [mode: {}]'.format(mode))

        dataset = tf.data.Dataset.from_generator(partial(self._test_generator, mode=mode),
                                                 output_types=(tf.int32, tf.float32),
                                                 output_shapes=((None, 3), (None, len(self.ent_to_idx))))

        dataset = dataset.repeat()
        dataset = dataset.prefetch(5)
        dataset_iter = dataset.make_one_shot_iterator()

        self.X_test_tf, self.X_test_filter_tf = dataset_iter.get_next()

        # Dependencies that need to be run before scoring
        test_dependency = []

        # For large graphs
        if self.dealing_with_large_graphs:
            raise NotImplementedError('ConvE not implemented with large graphs (yet)')

        else:

            # Compute scores for positive
            e_s, e_p, e_o = self._lookup_embeddings(self.X_test_tf)
            scores = tf.sigmoid(tf.squeeze(self._fn(e_s, e_p, e_o)), name='sigmoid_scores')

            # Score of positive triple
            self.score_positive = tf.gather(scores, indices=self.X_test_tf[:, 2], name='score_positive')

            # Score of every other positive sample for <s, p>, excluding positive sample
            # this is to remove the positives from output
            self.scores_filtered = tf.boolean_mask(scores, tf.cast(self.X_test_filter_tf, tf.bool))

            # Total rank over all possible entities
            self.total_rank = tf.reduce_sum(tf.cast(scores >= self.score_positive, tf.int32)) + 1

            # Rank over positive triples
            self.filter_rank = tf.reduce_sum(tf.cast(self.scores_filtered >= self.score_positive, tf.int32))

            # Rank of positive sample, with other positives filtered out
            self.rank = tf.subtract(self.total_rank, self.filter_rank, name='rank')

    def _initialize_early_stopping(self):
        """Initializes and creates evaluation graph for early stopping.
        """

        try:
            self.x_valid = self.early_stopping_params['x_valid']

            if isinstance(self.x_valid, np.ndarray):
                if self.x_valid.ndim <= 1 or (np.shape(self.x_valid)[1]) != 3:
                    msg = 'Invalid size for input x_valid. Expected (n,3):  got {}'.format(np.shape(self.x_valid))
                    logger.error(msg)
                    raise ValueError(msg)

                # store the validation data in the data handler
                self.x_valid = to_idx(self.x_valid, ent_to_idx=self.ent_to_idx, rel_to_idx=self.rel_to_idx)
                self.train_dataset_handle.set_data(self.x_valid, "valid", mapped_status=True)
                self.eval_dataset_handle = self.train_dataset_handle
                logger.debug('Initialized eval_dataset from train_dataset')

            elif isinstance(self.x_valid, AmpligraphDatasetAdapter):

                if not self.eval_dataset_handle.data_exists('valid'):
                    msg = 'Dataset `valid` has not been set in the DatasetAdapter.'
                    logger.error(msg)
                    raise ValueError(msg)

                self.eval_dataset_handle = self.x_valid
                print('Initialized eval_dataset from AmpligraphDatasetAdapter')

            else:
                msg = 'Invalid type for input X. Expected np.ndarray or AmpligraphDatasetAdapter object, \
                       got {}'.format(type(self.x_valid))
                logger.error(msg)
                raise ValueError(msg)
        except KeyError:
            msg = 'x_valid must be passed for early fitting.'
            logger.error(msg)
            raise KeyError(msg)

        self.early_stopping_criteria = self.early_stopping_params.get('criteria', DEFAULT_CRITERIA_EARLY_STOPPING)

        if self.early_stopping_criteria not in ['hits10', 'hits1', 'hits3', 'mrr']:
            msg = 'Unsupported early stopping criteria.'
            logger.error(msg)
            raise ValueError(msg)

        self.early_stopping_best_value = None
        self.early_stopping_stop_counter = 0
        try:
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
        except KeyError:
            logger.debug('x_filter not found in early_stopping_params.')
            pass

        # initialize evaluation graph in validation mode i.e. to use validation set
        self._initialize_eval_graph("valid", scoring_strategy=self.loss_params['scoring_strategy'])

    def predict(self, X, from_idx=False):
        """Predict the scores of triples using a trained embedding model.
            The function returns raw scores generated by the model.

            .. note::

                To obtain probability estimates, use a logistic sigmoid: ::

                    >>> model.fit(X)
                    >>> y_pred = model.predict(np.array([['f', 'y', 'e'], ['b', 'y', 'd']]))
                    >>> print(y_pred)
                    [-0.13863425, -0.09917116]
                    >>> from scipy.special import expit
                    >>> expit(y_pred)
                    array([0.4653968 , 0.47522753], dtype=float32)

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

        # adapt the data with conve adapter for internal use
        dataset_handle = ConvEDatasetAdapter(low_memory=self.low_memory)
        dataset_handle.use_mappings(self.rel_to_idx, self.ent_to_idx)
        dataset_handle.set_data(X, 'test', mapped_status=from_idx)
        self.eval_dataset_handle = dataset_handle

        # build tf graph for predictions
        if self.sess_predict is None:
            tf.reset_default_graph()
            self.rnd = check_random_state(self.seed)
            tf.random.set_random_seed(self.seed)
            self._load_model_from_trained_params()
            self._initialize_eval_graph(scoring_strategy=self.loss_params['scoring_strategy'])
            sess = tf.Session()
            sess.run(tf.tables_initializer())
            sess.run(tf.global_variables_initializer())
            self.sess_predict = sess

        self.sess_predict.run(self.set_training_false)
        self.eval_dataset_handle.set_output_mapping(self.output_mapping)
        self.eval_dataset_handle.generate_onehot_outputs(dataset_type='test')

        scores = []

        for i in tqdm(range(self.eval_dataset_handle.get_size('test'))):

            score = self.sess_predict.run([self.score_positive])

            if self.eval_config.get('default_protocol', DEFAULT_PROTOCOL_EVAL):
                scores.extend(list(score))
            else:
                scores.append(score)

        return scores

    def get_ranks(self, dataset_handle):
        """ Used by evaluate_predictions to get the ranks for evaluation.

        Parameters
        ----------
        dataset_handle : Object of AmpligraphDatasetAdapter
                         This contains handles of the generators used to get test triples and filters.

        Returns
        -------
        ranks : ndarray, shape [n] or [n,2] depending on the value of use_default_protocol.
                An array of ranks of test triples.
        """

        if not self.is_fitted:
            msg = 'Model has not been fitted.'
            logger.error(msg)
            raise RuntimeError(msg)

        self.eval_dataset_handle = dataset_handle

        # build tf graph for predictions
        if self.sess_predict is None:
            tf.reset_default_graph()
            self.rnd = check_random_state(self.seed)
            tf.random.set_random_seed(self.seed)
            self._load_model_from_trained_params()
            self._initialize_eval_graph(mode='test', scoring_strategy=self.loss_params['scoring_strategy'])
            sess = tf.Session()
            sess.run(tf.tables_initializer())
            sess.run(tf.global_variables_initializer())
            self.sess_predict = sess

        self.sess_predict.run(self.set_training_false)
        ranks = []

        for i in tqdm(range(self.eval_dataset_handle.get_size('test'))):

            rank = self.sess_predict.run(self.rank)

            if self.eval_config.get('default_protocol', DEFAULT_PROTOCOL_EVAL):
                ranks.append(list(rank))
            else:
                ranks.append(rank)

        return ranks
