from .models import *
from skimage.filters import gabor_kernel

class GaborE(EmbeddingModel):
    """ Convolutional 2D Knowledge Graph Embeddings with Gabor Filters

    .. math::

            concat(f([e_s;r_r;e_o] * \G)) W )

            where G is a Gabor filter bank. Additional details available at:                                                     #TODO.

    Examples
    --------
    >>> import numpy as np
    >>> from ampligraph.latent_features import GaborE
    >>> model = GaborE(batches_count=1, seed=555, epochs=20, k=10,
    >>>                embedding_model_params={'dropout': 0.1,
    >>>                                        'use_bias': True},
    >>>              loss='pairwise', loss_params={},
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
    >>> model.predict(np.array([['f', 'y', 'e'], ['b', 'y', 'd']]))                                                              # TODO:
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
                 embedding_model_params={'dropout': 0.0,
                                         'use_bias': True},
                 optimizer=DEFAULT_OPTIM,
                 optimizer_params={'lr': DEFAULT_LR},
                 loss='pairwise',
                 loss_params={},
                 regularizer=DEFAULT_REGULARIZER,
                 regularizer_params={},
                 initializer=DEFAULT_INITIALIZER,
                 initializer_params={'uniform': DEFAULT_XAVIER_IS_UNIFORM},
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
            GaborE-specific hyperparams:
            - **dropout** -  Dropout on the dense layer. Default: 0.0.
            - **use_bias** - Use bias layer. Default: True.

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

        verbose : bool
            Verbose mode.
        """

        # prepare filter bank kernels
        kernels = []
        kernel_params = []
        for theta in range(8):
            theta = theta / 8. * np.pi
            for sigma in [1]:
                for frequency in (0.05, 0.10, 0.15, 0.2, 0.25):
                    kernel = np.real(gabor_kernel(frequency, theta=theta, sigma_x=sigma, sigma_y=sigma))
                    kernels.append(kernel)
                    kernel_params.append({'frequency': frequency, 'theta': theta, 'sigma_x': sigma, 'sigma_y': sigma})

        num_kernels = len(kernels)
        kernel_size = kernels[0].shape[0]
        kernel_shapes = [x.shape for x in kernels]
        if len(np.unique(kernel_shapes)) != 1:
            raise ValueError('Gabor kernels are not size consistent. Change parameters!')

        # Find factor pairs (i,j) of concatenated embedding dimensions, where min(i,j) >= conv_kernel_size
        n = k*3
        emb_img_depth = 1

        emb_img_width, emb_img_height = None, None
        for i in range(int(np.sqrt(n)) + 1, kernel_size, -1):
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
        embedding_model_params['num_kernels'] = len(kernels)
        embedding_model_params['kernel_size'] = kernel_size
        embedding_model_params['gabor_kernel_params'] = kernel_params

        self.gabor_kernels = np.expand_dims(np.rollaxis(np.array(kernels), 0, 3), 2)

        # Calculate dense dimension
        embedding_model_params['dense_dim'] = (emb_img_width - (kernel_size - 1)) * (emb_img_height - (kernel_size - 1)) * num_kernels

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

            ninput = self.embedding_model_params['embed_image_depth']
            num_kernels = self.embedding_model_params['num_kernels']
            ksize = self.embedding_model_params['kernel_size']
            dense_dim = self.embedding_model_params['dense_dim']

            self.ent_emb = tf.get_variable('ent_emb', shape=[len(self.ent_to_idx), self.k],
                                           initializer=self.initializer.get_tf_initializer(), dtype=tf.float32)
            self.rel_emb = tf.get_variable('rel_emb', shape=[len(self.rel_to_idx), self.k],
                                           initializer=self.initializer.get_tf_initializer(), dtype=tf.float32)

            # kernel_weights =
            weights_init = tf.initializers.constant(self.gabor_kernels)

            self.conv2d_W = tf.get_variable('conv2d_gabor', shape=self.gabor_kernels.shape, trainable=False,
                                            initializer=weights_init, dtype=tf.float32)

            self.conv2d_B = tf.get_variable('conv2d_bias', shape=[num_kernels], trainable=True,
                                            initializer=tf.zeros_initializer(), dtype=tf.float32)

            self.dense_W = tf.get_variable('dense_weights', shape=[dense_dim, 1],
                                           trainable=True,
                                           initializer=tf.initializers.he_normal(seed=self.seed),
                                           dtype=tf.float32)
            self.dense_B = tf.get_variable('dense_bias', shape=[1], trainable=True,
                                           initializer=tf.zeros_initializer(), dtype=tf.float32)


        else:
            raise NotImplementedError('ConvE not implemented when dealing with large graphs (yet)')

    def _save_trained_params(self):
        """After model fitting, save all the trained parameters in trained_model_params in some order.
        The order would be useful for loading the model.
        This method must be overridden if the model has any other parameters (apart from entity-relation embeddings).
        """
        if not self.dealing_with_large_graphs:
            params_dict = {}
            params_dict['ent_emb'] = self.sess_train.run(self.ent_emb)
            params_dict['rel_emb'] = self.sess_train.run(self.rel_emb)
            params_dict['conv2d_B'] = self.sess_train.run(self.conv2d_B)
            params_dict['dense_W'] = self.sess_train.run(self.dense_W)
            params_dict['dense_B'] = self.sess_train.run(self.dense_B)
            self.trained_model_params = params_dict
        else:
            raise NotImplementedError('GaborE not implemented when dealing with large graphs (yet)')

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

            with tf.variable_scope('embeddings'):
                self.ent_emb = tf.Variable(self.trained_model_params['ent_emb'], dtype=tf.float32, name='ent_emb')
                self.rel_emb = tf.Variable(self.trained_model_params['rel_emb'], dtype=tf.float32, name='rel_emb')

            with tf.variable_scope('gabore'):
                self.conv2d_W = tf.Variable(self.gabor_kernels, dtype=tf.float32)
                self.conv2d_B = tf.Variable(self.trained_model_params['conv2d_B'], dtype=tf.float32)
                self.dense_W = tf.Variable(self.trained_model_params['dense_W'], dtype=tf.float32)
                self.dense_B = tf.Variable(self.trained_model_params['dense_B'], dtype=tf.float32)

        else:
            raise NotImplementedError('ConvE not implemented when dealing with large graphs (yet)')


    def _fn(self, e_s, e_p, e_o):
        """The GaborE scoring function.

            The function implements the scoring function as defined by
            .. math::

                concat(f([e_s;r_r;e_o] * \G)) W )

            where G is a Gabor filter bank. Additional details available at:                                                    #TODO


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
        stacked_emb = tf.stack([e_s, e_p, e_o], axis=2, name='stacked_embeddings')
        self.inputs = tf.reshape(stacked_emb, shape=[tf.shape(stacked_emb)[0], self.embedding_model_params['embed_image_height'],
                                 self.embedding_model_params['embed_image_width'], 1], name='embed_image')

        x = self.inputs

        x = tf.nn.conv2d(x, self.conv2d_W, [1, 1, 1, 1], padding='VALID')
        x = tf.nn.bias_add(x, self.conv2d_B)
        x = tf.nn.relu(x, name='conv_relu')

        x = tf.reshape(x, shape=[tf.shape(x)[0], self.embedding_model_params['dense_dim']])

        dropout_rate = tf.cond(self.tf_is_training,
                               true_fn=lambda: tf.constant(self.embedding_model_params['dropout']),
                               false_fn=lambda: tf.constant(0, dtype=tf.float32))
        x = tf.nn.dropout(x, rate=dropout_rate, name='dropout_dense')

        self.scores = tf.nn.xw_plus_b(x, self.dense_W, self.dense_B, name="scores")

        return tf.squeeze(self.scores)

