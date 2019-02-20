import numpy as np
import tensorflow as tf
from sklearn.utils import check_random_state
from functools import partial
import abc
from tqdm import tqdm



MODEL_REGISTRY = {}


from .loss_functions import LOSS_REGISTRY
from .regularizers import REGULARIZER_REGISTRY
from ..evaluation import generate_corruptions_for_fit, to_idx, create_mappings, generate_corruptions_for_eval, hits_at_n_score, mrr_score
import os

DEFAULT_PAIRWISE_MARGIN = 1



def register_model(name, external_params=[], class_params= {}):
    def insert_in_registry(class_handle):
        MODEL_REGISTRY[name] = class_handle
        class_handle.name = name
        MODEL_REGISTRY[name].external_params = external_params
        MODEL_REGISTRY[name].class_params = class_params
        return class_handle
    return insert_in_registry

class EmbeddingModel(abc.ABC):
    """Abstract class for a neural knowledge graph embedding model.
    """
    def __init__(self, k=100, eta=2, epochs=100, batches_count=100, seed=0,
                 embedding_model_params = {},
                 optimizer="adagrad", optimizer_params={}, 
                 loss='nll', loss_params = {}, 
                 regularizer="None", regularizer_params = {},
                 model_checkpoint_path='saved_model/', verbose=False, **kwargs):
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
            Parameter values of embedding model specific hyperparams
        optimizer : string
            The optimizer used to minimize the loss function. Choose between ``sgd``,
            ``adagrad``, ``adam``.
        optimizer_params : dict    
            Parameters values specific to the optimizer. (lr, etc)
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
            
        try:
            self.loss = LOSS_REGISTRY[loss](self.eta, self.loss_params, verbose=verbose)
        except KeyError:
            raise ValueError('Unsupported loss function: %s' % loss)
            
        try:
            self.regularizer = REGULARIZER_REGISTRY[regularizer](self.regularizer_params, verbose=verbose)
        except KeyError:
            raise ValueError('Unsupported regularizer: %s' % regularizer)
            
        
        self.optimizer_params = optimizer_params
        if optimizer == "adagrad":
            self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.optimizer_params.get('lr', 0.1))
        elif optimizer == "adam":
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.optimizer_params.get('lr', 0.1))
        elif optimizer == "sgd":
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.optimizer_params.get('lr', 0.1))
        else:
            raise ValueError('Unsupported optimizer: %s' % optimizer)

        self.verbose = verbose

        self.rnd = check_random_state(self.seed)

        self.initializer = tf.contrib.layers.xavier_initializer(uniform=False, seed=self.seed)
        self.tf_config = tf.ConfigProto(allow_soft_placement=True)
        self.tf_config.gpu_options.allow_growth = True
        self.sess_train = None
        self.sess_predict = None
        self.trained_model_params = []
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

    def get_embedding_model_params(self, output_dict):
        """save the model parameters in the dictionary.
        Parameters
        ----------
        output_dict : dictionary
            Dictionary of saved params. 
            It's the duty of the model to save all the variables correctly, so that it can be used for loading later.
        """
        output_dict['model_params'] = self.trained_model_params
        
    def restore_model_params(self, in_dict):
        """Load the model parameters from the input dictionary.
        
        Parameters
        ----------
        in_dict : dictionary
            Dictionary of saved params. It's the duty of the model to load the variables correctly
        """
        
        self.trained_model_params = in_dict['model_params']
        
    def _save_trained_params(self):
        """Save all the trained parameters that would be required for restoring. 
        """
        self.trained_model_params = self.sess_train.run([self.ent_emb, self.rel_emb])
        
    def _load_model_from_trained_params(self):
        """Load the model from trained params. 
            While restoring make sure that the order of loaded parameters match the saved order.
            It's the duty of the model to load the variables correctly
        """
        self.ent_emb = tf.constant(self.trained_model_params[0])
        self.rel_emb = tf.constant(self.trained_model_params[1])
        
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
            emb_list = self.trained_model_params[0]
            lookup_dict = self.ent_to_idx
        elif type is 'relation':
            emb_list = self.trained_model_params[1]
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

    def fit(self, X, early_stopping=False, early_stopping_params={}):
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

        batch_size = X.shape[0]//self.batches_count
        dataset = tf.data.Dataset.from_tensor_slices(X).repeat().batch(batch_size).prefetch(2)
        dataset_iter = dataset.make_one_shot_iterator()
        # init tf graph/dataflow for training
        # init variables (model parameters to be learned - i.e. the embeddings)
        self._initialize_embeddings()

        # training input placeholder
        x_pos_tf = tf.cast(dataset_iter.get_next(), tf.int32)
        all_ent_tf = tf.squeeze(tf.constant(list(self.ent_to_idx.values()), dtype=tf.int32))
        #generate negatives
        x_neg_tf = generate_corruptions_for_fit(x_pos_tf, all_ent_tf, self.eta, rnd=self.seed)
        if self.loss.get_state('require_same_size_pos_neg'):
            x_pos =  tf.reshape(tf.tile(tf.reshape(x_pos_tf,[-1]),[self.eta]),[tf.shape(x_pos_tf)[0]*self.eta,3])
            batch_size = batch_size * self.eta
        else:
            x_pos = x_pos_tf
        # look up embeddings from input training triples
        e_s_pos, e_p_pos, e_o_pos = self._lookup_embeddings(x_pos)
        e_s_neg, e_p_neg, e_o_neg = self._lookup_embeddings(x_neg_tf)

        scores_neg = self._fn(e_s_neg, e_p_neg, e_o_neg)
        scores_pos = self._fn(e_s_pos, e_p_pos, e_o_pos)

        loss = self.loss.apply(scores_pos, scores_neg)
        
        loss += self.regularizer.apply([self.ent_emb, self.rel_emb])

        train = self.optimizer.minimize(loss, var_list=[self.ent_emb, self.rel_emb])

        # Entity embeddings normalization
        normalize_ent_emb_op = self.ent_emb.assign(tf.clip_by_norm(self.ent_emb, clip_norm=1, axes=1))
        
        #early stopping 
        if early_stopping:
            try:
                x_valid = early_stopping_params['x_valid']
                if type(x_valid) != np.ndarray:
                    raise ValueError('Invalid type for input x_valid. Expected ndarray, got %s' % (type(x_valid)))

                if (np.shape(x_valid)[1]) != 3:
                    raise ValueError('Invalid size for input x_valid. Expected number of column 3, got %s' % (np.shape(x_valid)[1]))
                x_valid = to_idx(x_valid, ent_to_idx=self.ent_to_idx, rel_to_idx=self.rel_to_idx)
            
            except KeyError:
                raise KeyError('x_valid must be passed for early fitting.')
                                 
                
            early_stopping_criteria = early_stopping_params.get('criteria', 'hits10')
            if early_stopping_criteria not in ['hits10','hits1', 'hits3', 'mrr']:
                raise ValueError('Unsupported early stopping criteria')
            
                             
            early_stopping_best_value = 0
            early_stopping_stop_counter = 0
            try:
                x_filter = early_stopping_params['x_filter']
                x_filter = to_idx(x_filter, ent_to_idx=self.ent_to_idx, rel_to_idx=self.rel_to_idx)
                self.set_filter_for_eval(x_filter)
            except KeyError:
                pass
            
            self._initialize_eval_graph()    
            
        
        self.sess_train.run(tf.tables_initializer())
        self.sess_train.run(tf.global_variables_initializer())
        
        #X_batches = np.array_split(X, self.batches_count)
        
            
        
        for epoch in tqdm(range(self.epochs), disable=(not self.verbose), unit='epoch'):
            losses = []
            for batch in range(self.batches_count):
                loss_batch, _ = self.sess_train.run([ loss, train])#, {x_pos_tf: X_pos_b})
                if self.embedding_model_params.get('normalize_entity_embd', False):
                    self.sess_train.run(normalize_ent_emb_op)

                losses.append(loss_batch)
            if self.verbose:  
                tqdm.write('epoch: %d: mean loss: %.10f' % (epoch, sum(losses)/ (batch_size*self.batches_count)))
            
            # TODO TEC-1529: add early stopping criteria
            if early_stopping and epoch >= early_stopping_params.get('burn_in', 100)  \
                                and epoch%early_stopping_params.get('check_interval',10)==0:
                #compute and store test_loss
                ranks = []
                if x_valid.ndim>1:
                    for x_test_triple in x_valid:
                        rank_triple = self.sess_train.run([self.rank], feed_dict={self.X_test_tf: [x_test_triple]})
                        ranks.append(rank_triple)
                else:
                    ranks = self.sess_train.run([self.rank], feed_dict={self.X_test_tf: [x_test_triple]})
                
                if early_stopping_criteria == 'hits10':
                    current_test_value = hits_at_n_score(ranks,10)
                elif early_stopping_criteria == 'hits3':
                    current_test_value = hits_at_n_score(ranks,3)
                elif early_stopping_criteria == 'hits1':
                    current_test_value = hits_at_n_score(ranks,1)
                elif early_stopping_criteria == 'mrr':
                    current_test_value = mrr_score(ranks)
                
                
                if early_stopping_best_value >= current_test_value:
                    early_stopping_stop_counter += 1
                    if early_stopping_stop_counter == early_stopping_params.get('stop_interval', 3):
                        self.is_filtered = False
                        self.sess_train.close()
                        self.is_fitted = True
                        return
                else:
                    early_stopping_best_value = current_test_value
                    early_stopping_stop_counter = 0
                    self._save_trained_params()
        
        
        self._save_trained_params()
        self.sess_train.close()
        self.is_fitted = True
                    
    
    def _get_loss(self, scores_pos, scores_neg):
        return self.loss.apply(scores_pos, scores_neg)
    
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
        
        entity_size = len(self.ent_to_idx)
        reln_size = len(self.rel_to_idx)
        
        first_million_primes_list = []
        curr_dir, _ = os.path.split(__file__)
        with open(os.path.join(curr_dir, "prime_number_list.txt"), "r") as f:
            line = f.readline()
            i=0
            for line in f:
                p_nums_line = line.split(' ')
                first_million_primes_list.extend([np.int64(x) for x in p_nums_line if x !='' and x!='\n'])
                if len(first_million_primes_list)>(2*entity_size+reln_size):
                    break
        
        
        #subject
        self.entity_primes_left = first_million_primes_list[:entity_size]
        #obj
        self.entity_primes_right = first_million_primes_list[entity_size:2*entity_size]
        #reln
        self.relation_primes = first_million_primes_list[2*entity_size:(2*entity_size+reln_size)]

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
    
    def _initialize_eval_graph(self):
        self.X_test_tf = tf.placeholder(tf.int64, shape=[1, 3])
            
        self.table_entity_lookup_left = None
        self.table_entity_lookup_right = None
        self.table_reln_lookup = None

        all_entities_np = np.int64(np.array(list(self.ent_to_idx.values())))
        self.all_entities=tf.constant(all_entities_np)

        if self.is_filtered:
            all_reln_np = np.int64(np.array(list(self.rel_to_idx.values())))
            self.table_entity_lookup_left = tf.contrib.lookup.HashTable(
                            tf.contrib.lookup.KeyValueTensorInitializer(all_entities_np, 
                                                                        np.array(self.entity_primes_left, dtype=np.int64))
                            , 0) 
            self.table_entity_lookup_right = tf.contrib.lookup.HashTable(
                            tf.contrib.lookup.KeyValueTensorInitializer(all_entities_np, 
                                                                        np.array(self.entity_primes_right, dtype=np.int64))
                            , 0)                                   
            self.table_reln_lookup = tf.contrib.lookup.HashTable(
                            tf.contrib.lookup.KeyValueTensorInitializer(all_reln_np, 
                                                                        np.array(self.relation_primes, dtype=np.int64))
                            , 0)       

            #Create table to store train+test+valid triplet prime values(product)
            self.table_filter_lookup = tf.contrib.lookup.HashTable(
                            tf.contrib.lookup.KeyValueTensorInitializer(np.array(self.filter_keys, dtype=np.int64), np.zeros(len(self.filter_keys), dtype=np.int64))
                            , 1)



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
        self.rank = tf.reduce_sum(tf.cast(self.scores_predict >= self.score_positive, tf.int32))
        
    def end_evaluation(self):
        self.sess_predict.close()
        self.sess_predict=None
        self.is_filtered = False
        
        
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
            self._load_model_from_trained_params()
            
            self._initialize_eval_graph()
            
            sess = tf.Session()
            sess.run(tf.tables_initializer())
            sess.run(tf.global_variables_initializer())
            self.sess_predict = sess

        scores = []
        ranks = []
        if X.ndim>1:
            for x in X:
                all_scores, rank = self.sess_predict.run([self.scores_predict, self.rank], feed_dict={self.X_test_tf: [x]})
                scores.append(all_scores[0])
                ranks.append(rank)
        else:
            all_scores, ranks = self.sess_predict.run([self.scores_predict, self.rank], feed_dict={self.X_test_tf: [X]})
            scores = all_scores[0]
        
        return [scores, ranks]

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
        self.trained_model_params[0] = np.concatenate([self.trained_model_params[0], np.expand_dims(approximate_embedding, 0)], axis=0)
        self.ent_emb = tf.concat([self.ent_emb, new_emb], axis=0)
        self.ent_to_idx[e] = new_emb_idx

        return approximate_embedding


# TODO: missing docstring
class RandomBaseline():

    def __init__(self, seed=0):
        self.seed = seed
        self.rnd = check_random_state(self.seed)

    def fit(self, X, early_stopping=False, early_stopping_params={}):
        self.rel_to_idx, self.ent_to_idx = create_mappings(X)

    def predict(self, X, from_idx=False):
        return self.rnd.uniform(low=0, high=1, size=len(X))

@register_model("TransE", ["norm", "normalize_entity_embd"])
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
                 regularizer="None", regularizer_params = {},
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

    def fit(self, X, early_stopping=False, early_stopping_params={}):
        """Train an Translating Embeddings model.

            The model is trained on a training set X using the training protocol
            described in :cite:`trouillon2016complex`.

        Parameters
        ----------
        x : ndarray, shape [n, 3]
            The training triples

        """
        super().fit(X, early_stopping, early_stopping_params)

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
        return super().predict(X, from_idx=from_idx)

@register_model("DistMult", ["normalize_entity_embd"])
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
                 regularizer="None", regularizer_params = {},
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

    def fit(self, X, early_stopping=False, early_stopping_params={}):
        """Train an DistMult.

            The model is trained on a training set X using the training protocol
            described in :cite:`trouillon2016complex`.

        Parameters
        ----------
        x : ndarray, shape [n, 3]
            The training triples

        """
        super().fit(X, early_stopping, early_stopping_params)

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
        return super().predict(X, from_idx=from_idx)

@register_model("ComplEx")
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
                 regularizer="None", regularizer_params = {},
                 model_checkpoint_path='saved_model/', verbose=False, **kwargs):
        
        super().__init__(k=k, eta=eta, epochs=epochs, batches_count=batches_count, seed=seed,
                         embedding_model_params = embedding_model_params,
                         optimizer=optimizer, optimizer_params=optimizer_params,
                         loss=loss, loss_params=loss_params,
                         regularizer=regularizer, regularizer_params =regularizer_params,
                         model_checkpoint_path=model_checkpoint_path, verbose=verbose, **kwargs)

    def _initialize_embeddings(self):
        """ Initialize the embeddings
        """
        self.ent_emb = tf.get_variable('ent_emb', shape=[len(self.ent_to_idx), self.k*2],
                                        initializer=self.initializer)
        self.rel_emb = tf.get_variable('rel_emb', shape=[len(self.rel_to_idx), self.k*2],
                                        initializer=self.initializer)
        
        
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

    def fit(self, X, early_stopping=False, early_stopping_params={}):
        """Train a ComplEx model.

            The model is trained on a training set X using the training protocol
            described in :cite:`trouillon2016complex`.

        Parameters
        ----------
        x : ndarray, shape [n, 3]
            The training triples

        """
        super().fit(X, early_stopping, early_stopping_params)

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
        return super().predict(X, from_idx=from_idx)
    


@register_model("HolE")
class HolE(ComplEx):
    """ Holographic Embeddings model.

        The model as described in :cite:`NickelRP15` and cite:`HayashiS17`.

        .. math::

            f_{HolE}= 2 / n * f_{ComplEx}

        Examples
        --------
        >>> import numpy as np
        >>> from ampligraph.latent_features import HolE
        >>> model = HolE(batches_count=1, seed=555, epochs=20, k=10, 
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
        [[0.3046168, -0.0379385], [3, 9]]
        >>> model.get_embeddings(['f','e'], type='entity')
        array([[-0.2704807 , -0.05434025,  0.13363852,  0.04879733,  0.00184516,
        -0.1149573 , -0.1177371 , -0.20798951,  0.01935115,  0.13033926,
        -0.81528974,  0.22864424,  0.2045117 ,  0.1145515 ,  0.248952  ,
         0.03513691, -0.08550065, -0.06037813,  0.23231442, -0.39326245],
       [ 0.204738  ,  0.10758886, -0.11931524,  0.14881928,  0.0929039 ,
         0.25577265,  0.05722341,  0.2549932 , -0.16462566,  0.43789816,
        -0.91011846,  0.3533137 ,  0.1144442 ,  0.00359709, -0.09599967,
        -0.03151475,  0.14198618,  0.16138661,  0.07511608, -0.2465882 ]],
      dtype=float32)

    """

    def __init__(self, k=100, eta=2, epochs=100, batches_count=100, seed=0, 
                 embedding_model_params = {}, 
                 optimizer="adagrad", optimizer_params={}, 
                 loss='nll', loss_params = {}, 
                 regularizer="None", regularizer_params = {},
                 model_checkpoint_path='saved_model/', verbose=False, **kwargs):
        
        super().__init__(k=k, eta=eta, epochs=epochs, batches_count=batches_count, seed=seed,
                         embedding_model_params = embedding_model_params,
                         optimizer=optimizer, optimizer_params=optimizer_params,
                         loss=loss, loss_params=loss_params,
                         regularizer=regularizer, regularizer_params =regularizer_params,
                         model_checkpoint_path=model_checkpoint_path, verbose=verbose, **kwargs)

        
        
    def _fn(self, e_s, e_p, e_o):
        """The Hole scoring function.

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
        return (2/self.k) * (super()._fn(e_s, e_p, e_o))

    def fit(self, X, early_stopping=False, early_stopping_params={}):
        """Train a HolE model.

            The model is trained on a training set X using the training protocol
            described in :cite:`NickelRP15`.

        Parameters
        ----------
        x : ndarray, shape [n, 3]
            The training triples

        """
        super().fit(X, early_stopping, early_stopping_params)

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
        return super().predict(X, from_idx=from_idx) 
