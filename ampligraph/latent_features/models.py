import numpy as np
import tensorflow as tf
from sklearn.utils import check_random_state
from functools import partial
import abc
from tqdm import tqdm
from ..evaluation import generate_corruptions_for_fit, to_idx, create_mappings, generate_corruptions_for_eval
from .loss_functions import AbsoluteMarginLoss, NLLLoss, PairwiseLoss, SelfAdverserialLoss
from .regularizers import l1_regularizer, l2_regularizer
import os

DEFAULT_PAIRWISE_MARGIN = 1


class EmbeddingModel(abc.ABC):
    """Abstract class for a neural knowledge graph embedding model.
    """
    def __init__(self, k=100, eta=2, epochs=100, batches_count=100, seed=0,
                 embedding_model_params = {},
                 optimizer="adagrad", optimizer_params={}, 
                 loss='nll', loss_params = {}, 
                 regularizer=None, regularizer_params = {},
                 model_checkpoint_path='saved_model/', verbose=False, **kwargs):
        """Initialize an EmbeddingModel

            Also creates a news tf Session for training.

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
            Parameter values of embedding model specific hyperparams
        optimizer : string
            The optimizer used to minimize the loss function. Choose between ``sgd``,
            ``adagrad``, ``adam``, ``momentum``.
        optimizer_params : dict    
            Parameters values specific to the optimizer. (lr, momentum, etc)
        loss : string
            The type of loss function to use during training. If ``pairwise``, the pairwise margin-based loss function
            is chosen. If ``nll``, the model will use negative loss likelihood.
        loss_params : dict
            Parameters values specific to the loss.
        regularizer : string
            The regularization strategy to use with the loss function. ``L2`` or ``L1`` or None.
        regularizer_params : dict
            Parameters values specific to the regularizer.
        model_checkpoint_path: string
            Path to save the model.
        verbose : bool
            Verbose mode
        kwargs : dict
            Additional inputs, if any
        """
        #Store for restoring later.
        self.all_params = \
        {
            'k':k,
            'eta': eta,
            'epochs':epochs,
            'batches_count':batches_count,
            'seed': seed,
            'embedding_model_params':embedding_model_params,
            'optimizer':optimizer,
            'optimizer_params':optimizer_params,
            'loss':loss,
            'loss_params':loss_params,
            'regularizer':regularizer,
            'regularizer_params':regularizer_params,
            'model_checkpoint_path':model_checkpoint_path,
            'verbose':verbose
            
            
            
        }
        tf.reset_default_graph()
        
        self.is_filtered = False
        self.loss_params = loss_params
        self.loss_params['eta'] = eta
        
        self.embedding_model_params = embedding_model_params
        
        self.k = k
        self.seed = seed
        self.epochs = epochs
        self.eta = eta
        self.regularizer_params = regularizer_params
        self.batches_count = batches_count
        if batches_count == 1:
            print('WARN: when batches_count=1 all triples will be processed in the same batch. '
                  'This may introduce memory issues.')
        if loss == 'pairwise':
            self.loss = PairwiseLoss(self.loss_params)
        elif loss == 'nll':
            self.loss = NLLLoss(self.loss_params)
        elif loss == 'absolute_margin':
            self.loss = AbsoluteMarginLoss(self.loss_params)
        elif loss == 'self_adverserial':
            self.loss = SelfAdverserialLoss(self.loss_params)
        else:
            raise ValueError('Unsupported loss function: %s' % loss)
        
        self.optimizer_params = optimizer_params
        if optimizer == "adagrad":
            self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.optimizer_params.get('lr', 0.1))
        elif optimizer == "adam":
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.optimizer_params.get('lr', 0.1))
        elif optimizer == "momentum":
            self.optimizer = tf.train.MomentumOptimizer(self.optimizer_params.get('lr', 0.1), 
                                                        self.optimizer_params.get('momentum',0.1))
        elif optimizer == "sgd":
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.optimizer_params.get('lr', 0.1))
        else:
            raise ValueError('Unsupported optimizer: %s' % optimizer)

        if regularizer == 'L2':
            self.regularizer = l2_regularizer
        elif regularizer == 'L1':
            self.regularizer = l1_regularizer
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
    
    def _initialize_embeddings(self):
        """ Initialize the embeddings
        """
        self.ent_emb = tf.get_variable('ent_emb', shape=[len(self.ent_to_idx), self.k],
                                       initializer=self.initializer)
        self.rel_emb = tf.get_variable('rel_emb', shape=[len(self.rel_to_idx), self.k],
                                       initializer=self.initializer)

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

        self.all_entities_np = np.int64(np.array(list(self.ent_to_idx.values())))
        self.entity_size = self.all_entities_np.shape[0]
        
        self.all_reln_np = np.int64(np.array(list(self.rel_to_idx.values())))
        self.reln_size = self.all_reln_np.shape[0]
        
        
        # This is useful when we re-fit the same model (e.g. retraining in model selection)
        if self.is_fitted:
            tf.reset_default_graph()

        self.sess_train = tf.Session(config=self.tf_config)

        # init tf graph/dataflow for training
        # init variables (model parameters to be learned - i.e. the embeddings)
        self._initialize_embeddings()

        # training input placeholder
        x_pos_tf = tf.placeholder(tf.int32, shape=[None, 3])
        


        all_ent_tf = tf.squeeze(tf.constant(list(self.ent_to_idx.values()), dtype=tf.int32))
        #generate negatives
        x_neg_tf = generate_corruptions_for_fit(x_pos_tf, all_ent_tf, self.eta, rnd=self.seed)
        if self.loss.get_state('require_same_size_pos_neg'):
            x_pos = tf.cast(tf.keras.backend.repeat(x_pos_tf, self.eta), tf.int32)
            x_pos = tf.reshape(x_pos, [-1, 3])
        else:
            x_pos = x_pos_tf
        # look up embeddings from input training triples
        e_s_pos, e_p_pos, e_o_pos = self._lookup_embeddings(x_pos)
        e_s_neg, e_p_neg, e_o_neg = self._lookup_embeddings(x_neg_tf)

        scores_neg = self._fn(e_s_neg, e_p_neg, e_o_neg)
        scores_pos = self._fn(e_s_pos, e_p_pos, e_o_pos)

        loss = self.loss.apply(scores_pos, scores_neg)
        
        if self.regularizer is not None:
            loss += self.regularizer([self.ent_emb, self.rel_emb], self.regularizer_params)

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
        
    def set_filter_for_eval(self, x_filter):
        """Set the filter to be used during evaluation (filtered_corruption = corruptions - filter)
        
        We would be using a prime number based assignment and product for do the filtering.
        We associate a unique prime number for subject entities, object entities and to relations.
        Product of three prime numbers is divisible only by those three prime numbers. 
        So we generate this profuct for the filter triples and store it in a hash map.
        When corruptions are generated for a triple during evaluation, we follow a similar approach 
        and look up the product of corruption in the above hash table. If the corrupted triple is 
        present in the hashmap, it means that it was present in the filter list.

        Parameters
        ----------
        x_filter : ndarray, shape [n, 3]
            The filter triples

        """
        
        self.x_filter = x_filter
        
        first_million_primes_list = []
        curr_dir, _ = os.path.split(__file__)
        with open(os.path.join(curr_dir, "prime_number_list.txt"), "r") as f:
            line = f.readline()
            i=0
            for line in f:
                p_nums_line = line.split(' ')
                first_million_primes_list.extend([np.int64(x) for x in p_nums_line if x !='' and x!='\n'])
                if len(first_million_primes_list)>(2*self.entity_size+self.reln_size):
                    break
        
        
        #subject
        self.entity_primes_left = first_million_primes_list[:self.entity_size]
        #obj
        self.entity_primes_right = first_million_primes_list[self.entity_size:2*self.entity_size]
        #reln
        self.relation_primes = first_million_primes_list[2*self.entity_size:(2*self.entity_size+self.reln_size)]

        self.filter_keys = []
        #subject
        self.filter_keys = [ self.entity_primes_left[x_filter[i,0]] for i in range(x_filter.shape[0]) ]
        #obj
        self.filter_keys = [self.filter_keys[i] * self.entity_primes_right[x_filter[i,2]]
                       for i in range(x_filter.shape[0]) ] 
        #reln
        self.filter_keys = [self.filter_keys[i] * self.relation_primes[x_filter[i,1]] 
                       for i in range(x_filter.shape[0]) ] 
        
        self.is_filtered = True
        
        
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
        scores_predict : ndarray, shape [n]
            The predicted scores for input triples X.
            
        rank : ndarray, shape [n]
            Rank of the triple

        """

        if type(X) != np.ndarray:
            raise ValueError('Invalid type for input X. Expected ndarray, got %s' % (type(X)))

        if not self.is_fitted:
            raise RuntimeError('Model has not been fitted.')

        if not from_idx:
            X = to_idx(X, ent_to_idx=self.ent_to_idx, rel_to_idx=self.rel_to_idx)

        # build tf graph for predictions
        if self.sess_predict is None:
            self.ent_emb = tf.constant(self.ent_emb_const)
            self.rel_emb = tf.constant(self.rel_emb_const)
            self.X_test_tf = tf.placeholder(tf.int64, shape=[1, 3])
            
            self.table_entity_lookup_left = None
            self.table_entity_lookup_right = None
            self.table_reln_lookup = None
            
            if self.is_filtered:
                self.table_entity_lookup_left = tf.contrib.lookup.HashTable(
                                tf.contrib.lookup.KeyValueTensorInitializer(self.all_entities_np, 
                                                                            np.array(self.entity_primes_left, dtype=np.int64))
                                , 0) 
                self.table_entity_lookup_right = tf.contrib.lookup.HashTable(
                                tf.contrib.lookup.KeyValueTensorInitializer(self.all_entities_np, 
                                                                            np.array(self.entity_primes_right, dtype=np.int64))
                                , 0)                                   
                self.table_reln_lookup = tf.contrib.lookup.HashTable(
                                tf.contrib.lookup.KeyValueTensorInitializer(self.all_reln_np, 
                                                                            np.array(self.relation_primes, dtype=np.int64))
                                , 0)       

                #Create table to store train+test+valid triplet prime values(product)
                self.table_filter_lookup = tf.contrib.lookup.HashTable(
                                tf.contrib.lookup.KeyValueTensorInitializer(np.array(self.filter_keys, dtype=np.int64), np.zeros(len(self.filter_keys), dtype=np.int64))
                                , 1)

            self.all_entities=tf.constant(self.all_entities_np)
             
            self.out_corr, self.out_corr_prime = generate_corruptions_for_eval(self.X_test_tf, self.all_entities,
                                                             self.table_entity_lookup_left, 
                                                             self.table_entity_lookup_right,  
                                                             self.table_reln_lookup)
            
            
            if self.is_filtered:
                #check if corruption prime product is present in dataset prime product            
                self.presense_mask = self.table_filter_lookup.lookup(self.out_corr_prime)
                self.filtered_corruptions = tf.boolean_mask(self.out_corr, self.presense_mask)
            else:
                self.filtered_corruptions = self.out_corr

            self.concatinated_set = tf.concat([self.X_test_tf, self.filtered_corruptions], 0)
    
            e_s, e_p, e_o = self._lookup_embeddings(self.concatinated_set)
            self.scores_predict = self._fn(e_s, e_p, e_o)
            self.score_positive = tf.gather(self.scores_predict, 0)
            self.rank = tf.reduce_sum(tf.cast(self.scores_predict > self.score_positive, tf.int32)) + 1
            sess = tf.Session()
            sess.run(tf.tables_initializer())
            sess.run(tf.global_variables_initializer())
            self.sess_predict = sess

        return self.sess_predict.run([self.scores_predict, self.rank], feed_dict={self.X_test_tf: [X]}) #TODO TEC-1567 performance: check feed_dict

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
        >>> TransE(batches_count=1, seed=555, epochs=20, k=10, loss='pairwise', loss_params={'margin':5})
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
        array([-1.0393388, -2.1970382], dtype=float32)

    """

    def __init__(self, k=100, eta=2, epochs=100, batches_count=100, seed=0, 
                 embedding_model_params = {}, 
                 optimizer="adagrad", optimizer_params={}, 
                 loss='nll', loss_params = {}, 
                 regularizer=None, regularizer_params = {},
                 model_checkpoint_path='saved_model/', verbose=False, **kwargs):
        
        super().__init__(k=k, eta=eta, epochs=epochs, batches_count=batches_count, seed=seed,
                         embedding_model_params = embedding_model_params,
                         optimizer=optimizer, optimizer_params=optimizer_params,
                         loss=loss, loss_params=loss_params,
                         regularizer=regularizer, regularizer_params =regularizer_params,
                         model_checkpoint_path=model_checkpoint_path, verbose=verbose, **kwargs)

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

        return tf.negative(tf.norm(e_s + e_p - e_o, ord=self.embedding_model_params.get('norm', 1), axis=1))

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
        >>> DistMult(batches_count=1, seed=555, epochs=20, k=10, loss='pairwise', loss_params={'margin':5})
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
        array([ 0.9718404, -0.3637435], dtype=float32)

    """

    def __init__(self, k=100, eta=2, epochs=100, batches_count=100, seed=0, 
                 embedding_model_params = {}, 
                 optimizer="adagrad", optimizer_params={}, 
                 loss='nll', loss_params = {}, 
                 regularizer=None, regularizer_params = {},
                 model_checkpoint_path='saved_model/', verbose=False, **kwargs):
        
        super().__init__(k=k, eta=eta, epochs=epochs, batches_count=batches_count, seed=seed,
                         embedding_model_params = embedding_model_params,
                         optimizer=optimizer, optimizer_params=optimizer_params,
                         loss=loss, loss_params=loss_params,
                         regularizer=regularizer, regularizer_params =regularizer_params,
                         model_checkpoint_path=model_checkpoint_path, verbose=verbose, **kwargs)

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
        >>> model = ComplEx(batches_count=1, seed=555, epochs=20, k=10, 
        >>>             loss='pairwise', loss_params={'margin':1}, 
        >>>             regularizer='L2', regularizer_params={'lambda':0.1})

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
        array([ 1.0089628 , -0.51218843], dtype=float32)
        >>> model.get_embeddings(['f','e'], type='entity')
        array([[-0.43793657, -0.4387298 ,  0.38817367, -0.0461139 , -0.11901201,
            -0.65326005,  0.12317057,  0.05477504,  0.06185133,  0.00359724],
           [-0.6584873 , -0.24720612, -0.04071414,  0.03415959, -0.08960017,
            -0.20853978, -0.27599704, -0.5155798 ,  0.12172926,  0.27685398]],
          dtype=float32)

    """

    def __init__(self, k=100, eta=2, epochs=100, batches_count=100, seed=0, 
                 embedding_model_params = {}, 
                 optimizer="adagrad", optimizer_params={}, 
                 loss='nll', loss_params = {}, 
                 regularizer=None, regularizer_params = {},
                 model_checkpoint_path='saved_model/', verbose=False, **kwargs):
        
        super().__init__(k=k, eta=eta, epochs=epochs, batches_count=batches_count, seed=seed,
                         embedding_model_params = embedding_model_params,
                         optimizer=optimizer, optimizer_params=optimizer_params,
                         loss=loss, loss_params=loss_params,
                         regularizer=regularizer, regularizer_params =regularizer_params,
                         model_checkpoint_path=model_checkpoint_path, verbose=verbose, **kwargs)

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
    