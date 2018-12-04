import numpy as np
import tensorflow as tf
from sklearn.utils import check_random_state
from functools import partial
import abc
from tqdm import tqdm
from ..evaluation import generate_corruptions_for_fit, to_idx, create_mappings
from .loss_functions import negative_log_likelihood_loss, pairwise_loss, absolute_margin_loss


DEFAULT_PAIRWISE_MARGIN = 1


class EmbeddingModel(abc.ABC):
    """Abstract class for a neural knowledge graph embedding model.
    """

    def __init__(self, k=100, lr=.1, eta=2, epochs=100, batches_count=100, seed=0,
                 loss='pairwise', optimizer="adagrad", regularizer=None, lambda_reg=0.1,
                 model_checkpoint_path='saved_model/', verbose=False, **kwargs):
        """Initialize an EmbeddingModel

            Also creates a news tf Session for training.

        Parameters
        ----------
        k : int
            Embedding space dimensionality
        lr : float
            Initial learning rate for the optimizer
        eta : int
            The number of negatives that must be generated at runtime during training for each positive.
        epochs : int
            The iterations of the training loop.
        batches_count : int
            The number of batches in which the training set must be split during the training loop.
        seed : int
            The seed used by the internal random numbers generator.
        loss : string
            The type of loss function to use during training. If ``pairwise``, the pairwise margin-based loss function
            is chosen. If ``nll``, the model will use negative loss likelihood.
        optimizer : string
            The optimizer used to minimize the loss function. Choose between ``sgd``,
            ``adagrad``, ``adam``, ``momentum``.
        regularizer : string
            The regularization strategy to use with the loss function. ``L2`` or None.
        lambda_reg : float
            The L2 regularization weight (ignored if `regularizer` is None.)
        verbose : bool
            Verbose mode
        kwargs : dict
            Additional, model-specific hyperparameters (e.g.: pairwise_margin (float):
            The margin used by the pairwise margin-based loss function to discriminate positives form negatives).
        """
        tf.reset_default_graph()

        self.k = k
        self.lr = lr
        self.seed = seed
        self.epochs = epochs
        self.eta = eta
        self.lambda_reg = lambda_reg
        self.batches_count = batches_count
        if batches_count == 1:
            print('WARN: when batches_count=1 all triples will be processed in the same batch. '
                  'This may introduce memory issues.')

        self.loss_params = {}
        if loss == 'pairwise':
            self.loss = pairwise_loss
            self.loss_params['margin'] = kwargs.get('pairwise_margin', DEFAULT_PAIRWISE_MARGIN)
        elif loss == 'nll':
            self.loss = negative_log_likelihood_loss
        elif loss == 'absolute_margin':
            self.loss = absolute_margin_loss
        else:
            raise ValueError('Unsupported loss function: %s' % loss)

        if optimizer == "adagrad":
            self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.lr)
        elif optimizer == "adam":
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        elif optimizer == "momentum":
            self.optimizer = tf.train.MomentumOptimizer(self.lr, kwargs.get('momentum', 0.1))
        elif optimizer == "sgd":
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
        else:
            raise ValueError('Unsupported optimizer: %s' % optimizer)

        if regularizer == 'L2':
            self.regularizer = tf.contrib.layers.l2_regularizer(scale=lambda_reg)
        elif regularizer is None:
            self.regularizer = None
        else:
            raise ValueError('Unsupported regularizer: %s' % regularizer)

        self.verbose = verbose

        self.rnd = check_random_state(self.seed)

        self.initializer = tf.contrib.layers.xavier_initializer(uniform=False, seed=self.seed)
        self.tf_config = tf.ConfigProto(allow_soft_placement=True)
        self.tf_config.gpu_options.allow_growth = True
        self.sess_train = None
        self.sess_predict = None

        self.is_fitted = False
        self.model_checkpoint_path = model_checkpoint_path

    @abc.abstractmethod
    def _fn(self, e_s, e_p, e_o):
        """The scoring function of the model.

            Assigns a score to a list of triples, with a model-specific strategy.
            Triples are passed as lists of subject, predicate, object embeddings.

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
            The operation corresponding to the TransE scoring function.

        """
        NotImplementedError("This function is a placeholder in an abstract class")

    def get_embeddings(self, entities, type='entity'):
        """Get the embeddings of entities or relations

        Parameters
        ----------
        entities : array-like, dtype=int, shape=[n]
            The entities (or relations) of interest. Element of the vector must be the original string literals, and
            not internal IDs.
        type : string
            If 'entity', will consider input as KG entities. If `relation`, they will be treated as KG predicates.

        Returns
        -------
        embeddings : ndarray, shape [n, k]
            An array of k-dimensional embeddings.

        """

        if not self.is_fitted:
            raise RuntimeError('Model has not been fitted.')

        if type is 'entity':
            emb_list = self.ent_emb_const
            lookup_dict = self.ent_to_idx
        elif type is 'relation':
            emb_list = self.rel_emb_const
            lookup_dict = self.rel_to_idx
        else:
            raise ValueError('Invalid entity type: %s' % type)

        idxs = np.vectorize(lookup_dict.get)(entities)
        return emb_list[idxs]

    def _lookup_embeddings(self, x):
        """Get the embeddings for subjects, predicates, and objects of a a list of statements used to train the model.

        Parameters
        ----------
        x : ndarray, shape [n, k]
            A list of k-dimensional embeddings

        Returns
        -------
        e_s : Tensor
            A Tensor that includes the embeddings of the subjects.
        e_p : Tensor
            A Tensor that includes the embeddings of the predicates.
        e_o : Tensor
            A Tensor that includes the embeddings of the objects.
        """
        e_s = tf.nn.embedding_lookup(self.ent_emb, x[:, 0], name='embedding_lookup_subject')
        e_p = tf.nn.embedding_lookup(self.rel_emb, x[:, 1], name='embedding_lookup_predicate')
        e_o = tf.nn.embedding_lookup(self.ent_emb, x[:, 2], name='embedding_lookup_object')
        return e_s, e_p, e_o

    def fit(self, X):
        """Train an EmbeddingModel.

            The model is trained on a training set X using the training protocol
            described in :cite:`trouillon2016complex`.

        Parameters
        ----------
        x : ndarray, shape [n, 3]
            The training triples

        """
        if type(X) != np.ndarray:
            raise ValueError('Invalid type for input X. Expected ndarray, got %s' % (type(X)))

        if (np.shape(X)[1]) != 3:
            raise ValueError('Invalid size for input X. Expected number of column 3, got %s' % (np.shape(X)[1]))

        # create internal IDs mappings
        self.rel_to_idx, self.ent_to_idx = create_mappings(X)
        #  convert training set into internal IDs
        X = to_idx(X, ent_to_idx=self.ent_to_idx, rel_to_idx=self.rel_to_idx)

        # This is useful when we re-fit the same model (e.g. retraining in model selection)
        if self.is_fitted:
            tf.reset_default_graph()

        self.sess_train = tf.Session(config=self.tf_config)

        # init tf graph/dataflow for training
        # init variables (model parameters to be learned - i.e. the embeddings)
        self.ent_emb = tf.get_variable('ent_emb', shape=[len(self.ent_to_idx), self.k],
                                       initializer=self.initializer,
                                       regularizer=self.regularizer)
        self.rel_emb = tf.get_variable('rel_emb', shape=[len(self.rel_to_idx), self.k],
                                       initializer=self.initializer,
                                       regularizer=self.regularizer)

        # training input placeholder
        x_pos_tf = tf.placeholder(tf.int32, shape=[None, 3])
        


        all_ent_tf = tf.squeeze(tf.constant(list(self.ent_to_idx.values()), dtype=tf.int32))
        #generate negatives
        x_neg_tf = generate_corruptions_for_fit(x_pos_tf, all_ent_tf, self.eta, rnd=self.seed)

        x_pos = tf.cast(tf.keras.backend.repeat(x_pos_tf, self.eta), tf.int32)
        x_pos = tf.reshape(x_pos, [-1, 3])
        # look up embeddings from input training triples
        e_s_pos, e_p_pos, e_o_pos = self._lookup_embeddings(x_pos)
        e_s_neg, e_p_neg, e_o_neg = self._lookup_embeddings(x_neg_tf)

        scores_neg = self._fn(e_s_neg, e_p_neg, e_o_neg)
        scores_pos = self._fn(e_s_pos, e_p_pos, e_o_pos)

        loss = self.loss(scores_pos, scores_neg, **self.loss_params)

        train = self.optimizer.minimize(loss, var_list=[self.ent_emb, self.rel_emb])

        # Entity embeddings normalization
        normalize_ent_emb_op = self.ent_emb.assign(tf.clip_by_norm(self.ent_emb, clip_norm=1, axes=1))

        init = tf.global_variables_initializer()

        self.sess_train.run(init)

        X_batches = np.array_split(X, self.batches_count)

        # X_negs = Parallel(n_jobs=os.cpu_count(),
        #                   verbose=5 if self.verbose else 0)(delayed(generate_corruptions_for_fit)(X_batch,
        #                                                                                           eta=self.eta,
        #                                                                                           rnd=self.rnd,
        #                                                                                           ent_to_idx=self.ent_to_idx)
        #                                                     for X_batch in X_batches)

        losses = []
        for epoch in tqdm(range(self.epochs), disable=(not self.verbose), unit='epoch'):
            for j in range(self.batches_count):
                X_pos_b = X_batches[j]
                _, loss_batch = self.sess_train.run([train, loss], {x_pos_tf: X_pos_b})
                self.sess_train.run(normalize_ent_emb_op)
                if self.verbose:
                    mean_loss = loss_batch / (len(X_pos_b)*self.eta)
                    losses.append(mean_loss)
                    tqdm.write('epoch: %d, batch %d: mean loss: %.10f' % (epoch, j, mean_loss))
            # TODO TEC-1529: add early stopping criteria

        # Move embeddings to constants for predict()
        self.ent_emb_const = self.sess_train.run(self.ent_emb)
        self.rel_emb_const = self.sess_train.run(self.rel_emb)
        self.sess_train.close()

        self.is_fitted = True

    def predict(self, X, from_idx=False):
        """Predict the score of triples using a trained embedding model.

            The function returns raw scores generated by the model.
            To obtain probability estimates, use a logistic sigmoid.

        Parameters
        ----------
        X : ndarray, shape [n, 3]
            The triples to score.
        from_idx : bool
            If True, will skip conversion to internal IDs. (default: False).

        Returns
        -------
        y_pred : ndarray, shape [n]
            The predicted scores for input triples X.

        """

        if type(X) != np.ndarray:
            raise ValueError('Invalid type for input X. Expected ndarray, got %s' % (type(X)))

        if (np.shape(X)[1]) != 3:
            raise ValueError('Invalid size for input X. Expected number of column 3, got %s' % (np.shape(X)[1]))

        if not self.is_fitted:
            raise RuntimeError('Model has not been fitted.')

        if not from_idx:
            X = to_idx(X, ent_to_idx=self.ent_to_idx, rel_to_idx=self.rel_to_idx)

        # build tf graph for predictions
        if self.sess_predict is None:
            self.ent_emb = tf.constant(self.ent_emb_const)
            self.rel_emb = tf.constant(self.rel_emb_const)
            self.X_test_tf = tf.placeholder(tf.int32, shape=[None, 3])
            e_s, e_p, e_o = self._lookup_embeddings(self.X_test_tf)
            self.scores_predict = self._fn(e_s, e_p, e_o)
            sess = tf.Session()
            sess.run(tf.global_variables_initializer())
            self.sess_predict = sess

        return self.sess_predict.run(self.scores_predict, feed_dict={self.X_test_tf: X}) #TODO TEC-1567 performance: check feed_dict

    def generate_approximate_embeddings(self, e, neighbouring_triples, pool='avg', schema_aware=False):
        """Generate approximate embeddings for entity, given neighbouring triples
            from auxiliary graph and a defined pooling function.
            
                    
        Parameters
        ----------
        entity : str
            An entity label.
        neighbouring_triples : ndarray, shape [n, 3]
            The neighbouring triples. 
        pool : str {'avg', 'max', 'sum'} : 
            The pooling function to approximate the entity embedding. (Default: 'avg')
        schema_aware: bool 
            Flag if approximate embeddings are aware of schema. 
        """

        # Get entity neighbours
        neighbour_entities = neighbouring_triples[:,[0,2]]
        N_ent = np.unique(neighbour_entities[np.where(neighbour_entities != e)])

        # Raise ValueError if a neighbour entity is not found in entity dict
        if not np.all([x in self.ent_to_idx.keys() for x in N_ent]):
            invalid_triples = neighbouring_triples[np.where([x not in self.ent_to_idx for x in N_ent])]
            raise ValueError('Neighbouring triples contain two OOKG entities: ', invalid_triples)

        # Get embeddings of each set and concatenate
        neighbour_vectors = self.get_embeddings(N_ent, type='entity')

        if pool == 'avg':
            pool_fn = partial(np.mean, axis=0)
        elif pool == 'max':
            pool_fn = partial(np.max, axis=0)
        elif pool == 'sum':
            pool_fn = partial(np.sum, axis=0)
        else:
            raise ValueError('Unsupported pooling function: %s' % pool)

        # Apply pooling function
        approximate_embedding = pool_fn(neighbour_vectors)

        new_emb = tf.constant(approximate_embedding, shape=[1, self.k], dtype=self.ent_emb.dtype)
        new_emb_idx = int(self.ent_emb.shape[0])

        # Add e and approximate embedding to self.ent_emb
        self.ent_emb_const = np.concatenate([self.ent_emb_const, np.expand_dims(approximate_embedding, 0)], axis=0)
        self.ent_emb = tf.concat([self.ent_emb, new_emb], axis=0)
        self.ent_to_idx[e] = new_emb_idx

        return approximate_embedding


# TODO: missing docstring
class RandomBaseline():

    def __init__(self, seed=0):
        self.seed = seed
        self.rnd = check_random_state(self.seed)

    def fit(self, X):
        self.rel_to_idx, self.ent_to_idx = create_mappings(X)

    def predict(self, X, from_idx=False):
        return self.rnd.uniform(low=0, high=1, size=len(X))


class TransE(EmbeddingModel):
    """The Translating Embedding model (TransE)

        The model as described in :cite:`bordes2013translating`.

        .. math::

            f_{TransE}=-||(\mathbf{e}_s + \mathbf{r}_p) - \mathbf{e}_o||_n

        Examples
        --------
        >>> import numpy as np
        >>> from ampligraph.latent_features import TransE
        >>> model = TransE(batches_count=1, seed=555, epochs=20, k=10)
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
        array([-2.2089951 , -4.65821838], dtype=float32)

    """

    def __init__(self, k=100, lr=.1, norm=1, eta=2, epochs=100, batches_count=100, seed=0,
                 loss='pairwise', pairwise_margin=5, optimizer="adagrad", verbose=False):
        self.hyperparams = {
            'k': k,
            'lr': lr,
            'norm': norm,
            'eta': eta,
            'epochs': epochs,
            'batches_count': batches_count,
            'seed': seed,
            'loss': loss,
            'pairwise_margin': pairwise_margin,            
            'optimizer': optimizer,
            'verbose': verbose
        }
        self.norm = norm
        super().__init__(k=k, lr=lr, eta=eta, epochs=epochs, batches_count=batches_count, seed=seed,
                         loss=loss, optimizer=optimizer, pairwise_margin=pairwise_margin, verbose=verbose)

    def _fn(self, e_s, e_p, e_o):
        """The TransE scoring function.

            .. math::

                f_{TransE}=-||(\mathbf{e}_s + \mathbf{r}_p) - \mathbf{e}_o||_n

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
            The operation corresponding to the TransE scoring function.

        """

        return tf.negative(tf.norm(e_s + e_p - e_o, ord=self.norm, axis=1))

    def fit(self, X):
        """Train an Translating Embeddings model.

            The model is trained on a training set X using the training protocol
            described in :cite:`trouillon2016complex`.

        Parameters
        ----------
        x : ndarray, shape [n, 3]
            The training triples

        """
        super().fit(X)

    def predict(self, X, from_idx=False):
        """Predict the score of triples using a trained embedding model.

            The function returns raw scores generated by the model.
            To obtain probability estimates, use a logistic sigmoid.

        Parameters
        ----------
        X : ndarray, shape [n, 3]
            The triples to score.
        from_idx : bool
            If True, will skip conversion to internal IDs. (default: False).

        Returns
        -------
        y_pred : ndarray, shape [n]
            The probability estimates predicted for input triples X.

        """
        return super().predict(X, from_idx=from_idx)

class DistMult(EmbeddingModel):
    """The DistMult model.

        The model as described in :cite:`yang2014embedding`.

        .. math::

            f_{DistMult}=\langle \mathbf{r}_p, \mathbf{e}_s, \mathbf{e}_o \\rangle

        Examples
        --------
        >>> import numpy as np
        >>> from ampligraph.latent_features import DistMult
        >>> model = DistMult(batches_count=1, seed=555, epochs=20, k=10)
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
        array([ 0.86894906, -0.57398415], dtype=float32)

    """

    def __init__(self, k=100, lr=.1, pairwise_margin=5, eta=2, epochs=100, batches_count=100, seed=0,
                 loss='pairwise', optimizer="adagrad", verbose=False):
        self.hyperparams = {
            'k': k,
            'lr': lr,
            'pairwise_margin': pairwise_margin,
            'eta': eta,
            'epochs': epochs,
            'batches_count': batches_count,
            'seed': seed,
            'loss': loss,
            'optimizer': optimizer,
            'verbose': verbose
        }
        super().__init__(k=k, lr=lr, eta=eta, epochs=epochs, batches_count=batches_count, seed=seed,
                         loss=loss, pairwise_margin=pairwise_margin, optimizer=optimizer, verbose=verbose)

    def _fn(self, e_s, e_p, e_o):
        """The DistMult scoring function.

        .. math::

            f_{DistMult}=\langle \mathbf{r}_p, \mathbf{e}_s, \mathbf{e}_o \\rangle


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
            The operation corresponding to the TransE scoring function.

        """

        return tf.reduce_sum(e_s * e_p * e_o, axis=1)

    def fit(self, X):
        """Train an DistMult.

            The model is trained on a training set X using the training protocol
            described in :cite:`trouillon2016complex`.

        Parameters
        ----------
        x : ndarray, shape [n, 3]
            The training triples

        """
        super().fit(X)

    def predict(self, X, from_idx=False):
        """Predict the score of triples using a trained embedding model.

            The function returns raw scores generated by the model.
            To obtain probability estimates, use a logistic sigmoid.

        Parameters
        ----------
        X : ndarray, shape [n, 3]
            The triples to score.
        from_idx : bool
            If True, will skip conversion to internal IDs. (default: False).

        Returns
        -------
        y_pred : ndarray, shape [n]
            The probability estimates predicted for input triples X.

        """
        return super().predict(X, from_idx=from_idx)
    
class ComplEx(EmbeddingModel):
    """ Complex Embeddings model.

        The model as described in :cite:`trouillon2016complex`.

        .. math::

            f_{ComplEx}=Re(\langle \mathbf{r}_p, \mathbf{e}_s, \overline{\mathbf{e}_o}  \\rangle)

        Examples
        --------
        >>> import numpy as np
        >>> from ampligraph.latent_features import ComplEx
        >>> model = ComplEx(batches_count=1, seed=555, epochs=20, k=10)
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
        array([ 1.23718071, -0.67018294], dtype=float32)
        >>> model.get_embeddings(['f','e'], type='entity')
        array([[-0.30896732, -0.31423435,  0.56920165, -0.05230011,  0.06782568,
                -0.59959912,  0.03503535, -0.02522233,  0.22599372,  0.24902618],
               [-0.65675682,  0.00590804, -0.26364127,  0.08479995, -0.0316195 ,
                -0.37855583, -0.23295116,  0.03035271,  0.38755456, -0.37715697]], dtype=float32)

    """

    def __init__(self, k=100, lr=.1, pairwise_margin=1, eta=2, epochs=100, batches_count=100, seed=0,
                 lambda_reg=0.1, loss='pairwise', optimizer="adagrad", regularizer='L2', verbose=False):
        self.hyperparams = {
            'k': k,
            'lr': lr,
            'pairwise_margin': pairwise_margin,
            'eta': eta,
            'epochs': epochs,
            'batches_count': batches_count,
            'seed': seed,
            'lambda_reg': lambda_reg,
            'loss': loss,
            'optimizer': optimizer,
            'regularizer': regularizer,
            'verbose': verbose
        }

        super().__init__(k=k, lr=lr, pairwise_margin=pairwise_margin, eta=eta, epochs=epochs,
                         batches_count=batches_count, seed=seed, loss=loss, optimizer=optimizer,
                         lambda_reg=lambda_reg, regularizer=regularizer, verbose=verbose)

    def _fn(self, e_s, e_p, e_o):
        """The ComplEx scoring function.

            .. math::

                f_{ComplEx}=Re(\langle \mathbf{r}_p, \mathbf{e}_s, \overline{\mathbf{e}_o}  \\rangle)

            Additional details available in :cite:`trouillon2016complex` (Equation 9).

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
            The operation corresponding to the TransE scoring function.

        """

        # Assume each embedding is made of an img and real component.
        # (These components are actually real numbers, see [Trouillon17].
        e_s_real, e_s_img = tf.split(e_s, 2, axis=1)
        e_p_real, e_p_img = tf.split(e_p, 2, axis=1)
        e_o_real, e_o_img = tf.split(e_o, 2, axis=1)

        # See Eq. 9 [Trouillon17):
        return tf.reduce_sum(e_p_real * e_s_real * e_o_real, axis=1) + \
               tf.reduce_sum(e_p_real * e_s_img * e_o_img, axis=1) + \
               tf.reduce_sum(e_p_img * e_s_real * e_o_img, axis=1) - \
               tf.reduce_sum(e_p_img * e_s_img * e_o_real, axis=1)

    def fit(self, X):
        """Train a ComplEx model.

            The model is trained on a training set X using the training protocol
            described in :cite:`trouillon2016complex`.

        Parameters
        ----------
        x : ndarray, shape [n, 3]
            The training triples

        """
        super().fit(X)

    def predict(self, X, from_idx=False):
        """Predict the score of triples using a trained embedding model.

            The function returns raw scores generated by the model.
            To obtain probability estimates, use a logistic sigmoid.

        Parameters
        ----------
        X : ndarray, shape [n, 3]
            The triples to score.
        from_idx : bool
            If True, will skip conversion to internal IDs. (default: False).

        Returns
        -------
        y_pred : ndarray, shape [n]
            The probability estimates predicted for input triples X.

        """
        return super().predict(X, from_idx=from_idx)