import numpy as np
import tensorflow as tf
from sklearn.utils import check_random_state
import abc
from tqdm import tqdm
import logging

MODEL_REGISTRY = {}

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

from .loss_functions import LOSS_REGISTRY
from .regularizers import REGULARIZER_REGISTRY
from ..evaluation import generate_corruptions_for_fit, to_idx, create_mappings, generate_corruptions_for_eval, \
    hits_at_n_score, mrr_score
import os

#######################################################################################################
# If not specified, following defaults will be used at respective locations

# Default learning rate for the optimizers
DEFAULT_LR = 0.0005

# Default momentum for the optimizers
DEFAULT_MOMENTUM = 0.9

# Default burn in for early stopping
DEFAULT_BURN_IN_EARLY_STOPPING = 100

# Default check interval for early stopping
DEFAULT_CHECK_INTERVAL_EARLY_STOPPING = 10

# Default stop interval for early stopping
DEFAULT_STOP_INTERVAL_EARLY_STOPPING = 3

# default evaluation criteria for early stopping
DEFAULT_CRITERIA_EARLY_STOPPING = 'mrr'

# default value which indicates whether to normalize the embeddings after each batch update
DEFAULT_NORMALIZE_EMBEDDINGS = False


# Default side to corrupt for evaluation
DEFAULT_CORRUPT_SIDE = 's+o'

# default hyperparameter for transE
DEFAULT_NORM_TRANSE = 1

# initial value for criteria in early stopping
INITIAL_EARLY_STOPPING_CRITERIA_VALUE = 0

# default value for the way in which the corruptions are to be generated while training/testing.
# Uses all entities
DEFAULT_CORRUPTION_ENTITIES = 'all'

# Threshold (on number of unique entities) to categorize the data as Huge Dataset (to warn user)
ENTITY_WARN_THRESHOLD = 5e5

# Default value for k (embedding size)
DEFAULT_EMBEDDING_SIZE = 100

# Default value for eta (number of corrputions to be generated for training)
DEFAULT_ETA = 2

# Default value for number of epochs
DEFAULT_EPOCH = 100

# Default value for batch count
DEFAULT_BATCH_COUNT = 100

# Default value for seed
DEFAULT_SEED = 0

# Default value for optimizer
DEFAULT_OPTIM = "adam"

# Default value for loss type
DEFAULT_LOSS = "nll"

# Default value for regularizer type 
DEFAULT_REGULARIZER = None

# Default value for verbose
DEFAULT_VERBOSE = False

#######################################################################################################

def register_model(name, external_params=[], class_params={}):
    def insert_in_registry(class_handle):
        MODEL_REGISTRY[name] = class_handle
        class_handle.name = name
        MODEL_REGISTRY[name].external_params = external_params
        MODEL_REGISTRY[name].class_params = class_params
        return class_handle

    return insert_in_registry


class EmbeddingModel(abc.ABC):
    """Abstract class for embedding models

    AmpliGraph neural knowledge graph embeddings models extend this class and its functionalities.

    """

    def __init__(self, 
                 k=DEFAULT_EMBEDDING_SIZE, 
                 eta=DEFAULT_ETA, 
                 epochs=DEFAULT_EPOCH, 
                 batches_count=DEFAULT_BATCH_COUNT, 
                 seed=DEFAULT_SEED,
                 embedding_model_params={},
                 optimizer=DEFAULT_OPTIM, 
                 optimizer_params={'lr':DEFAULT_LR},
                 loss=DEFAULT_LOSS, 
                 loss_params={},
                 regularizer=DEFAULT_REGULARIZER, 
                 regularizer_params={},
                 verbose=DEFAULT_VERBOSE):
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
            Model-specific hyperparams, passed to the model as a dictionary.
            Refer to model-specific documentation for details.

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

            - 'pairwise'  the model will use pairwise margin-based loss function.
            - 'nll' the model will use negative loss likelihood.
            - 'absolute_margin' the model will use absolute margin likelihood.
            - 'self_adversarial' the model will use adversarial sampling loss function.

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

        verbose : bool
            Verbose mode
        """
        # Store for restoring later.
        self.all_params = \
            {
                'k': k,
                'eta': eta,
                'epochs': epochs,
                'batches_count': batches_count,
                'seed': seed,
                'embedding_model_params': embedding_model_params,
                'optimizer': optimizer,
                'optimizer_params': optimizer_params,
                'loss': loss,
                'loss_params': loss_params,
                'regularizer': regularizer,
                'regularizer_params': regularizer_params,
                'verbose': verbose

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
            logger.warn(
                'batches_count=1. All triples will be processed in the same batch. This may introduce memory issues.')
            print('WARN: when batches_count=1 all triples will be processed in the same batch. '
                  'This may introduce memory issues.')

        try:
            self.loss = LOSS_REGISTRY[loss](self.eta, self.loss_params, verbose=verbose)
        except KeyError:
            msg = 'Unsupported loss function: {}'.format(loss)
            logger.error(msg)
            raise ValueError(msg)

        try:
            if regularizer is not None:
                self.regularizer = REGULARIZER_REGISTRY[regularizer](self.regularizer_params, verbose=verbose)
            else:
                self.regularizer = regularizer
        except KeyError:
            msg = 'Unsupported regularizer: {}'.format(regularizer)
            logger.error(msg)
            raise ValueError(msg)

        self.optimizer_params = optimizer_params
        if optimizer == "adagrad":
            self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.optimizer_params.get('lr', DEFAULT_LR))
        elif optimizer == "adam":
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.optimizer_params.get('lr', DEFAULT_LR))
        elif optimizer == "sgd":
            self.optimizer = tf.train.GradientDescentOptimizer(
                learning_rate=self.optimizer_params.get('lr', DEFAULT_LR))
        elif optimizer == "momentum":
            self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.optimizer_params.get('lr', DEFAULT_LR),
                                                        momentum=self.optimizer_params.get('momentum',
                                                                                           DEFAULT_MOMENTUM))
        else:
            msg = 'Unsupported optimizer: {}'.format(optimizer)
            logger.error(msg)
            raise ValueError(msg)

        self.verbose = verbose

        self.rnd = check_random_state(self.seed)

        self.initializer = tf.contrib.layers.xavier_initializer(uniform=False, seed=self.seed)
        self.tf_config = tf.ConfigProto(allow_soft_placement=True)
        self.tf_config.gpu_options.allow_growth = True
        self.sess_train = None
        self.sess_predict = None
        self.trained_model_params = []
        self.is_fitted = False
        self.eval_config = {}

    @abc.abstractmethod
    def _fn(self, e_s, e_p, e_o):
        """The scoring function of the model.

            Assigns a score to a list of triples, with a model-specific strategy.
            Triples are passed as lists of subject, predicate, object embeddings.
            This function must be overridden by every model to return corresponding score.

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
            The operation corresponding to the scoring function.

        """
        logger.error('_fn is a placeholder function in an abstract class')
        NotImplementedError("This function is a placeholder in an abstract class")

    def get_embedding_model_params(self, output_dict):
        """save the model parameters in the dictionary.

        Parameters
        ----------
        output_dict : dictionary
            Dictionary of saved params. 
            It's the duty of the model to save all the variables correctly, so that it can be used for restoring later.
        
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
        """After model fitting, save all the trained parameters in trained_model_params in some order. 
        The order would be useful for loading the model. 
        This method must be overridden if the model has any other parameters (apart from entity-relation embeddings)
        """
        self.trained_model_params = self.sess_train.run([self.ent_emb, self.rel_emb])

    def _load_model_from_trained_params(self):
        """Load the model from trained params. 
            While restoring make sure that the order of loaded parameters match the saved order.
            It's the duty of the embedding model to load the variables correctly.
            This method must be overridden if the model has any other parameters (apart from entity-relation embeddings)
        """
        self.ent_emb = tf.constant(self.trained_model_params[0])
        self.rel_emb = tf.constant(self.trained_model_params[1])

    def get_embeddings(self, entities, type='entity'):
        """Get the embeddings of entities or relations.

        Parameters
        ----------
        entities : array-like, dtype=int, shape=[n]
            The entities (or relations) of interest. Element of the vector must be the original string literals, and
            not internal IDs.
        type : string
            If 'entity', the ``entities`` argument will be considered as a list of knowledge graph entities (i.e. nodes).
            If set to 'relation', they will be treated as relation types instead (i.e. predicates).

        Returns
        -------
        embeddings : ndarray, shape [n, k]
            An array of k-dimensional embeddings.

        """
        # TODO - Rename type with something else. This is masking the built-in function "type" #44
        if not self.is_fitted:
            msg = 'Model has not been fitted.'
            logger.error(msg)
            raise RuntimeError(msg)

        if type is 'entity':
            emb_list = self.trained_model_params[0]
            lookup_dict = self.ent_to_idx
        elif type is 'relation':
            emb_list = self.trained_model_params[1]
            lookup_dict = self.rel_to_idx
        else:
            msg = 'Invalid entity type: {}'.format(type)
            logger.error(msg)
            raise ValueError(msg)

        idxs = np.vectorize(lookup_dict.get)(entities)
        return emb_list[idxs]

    def _lookup_embeddings(self, x):
        """Get the embeddings for subjects, predicates, and objects of a list of statements used to train the model.

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

    def _initialize_parameters(self):
        """ Initialize parameters of the model. 
            
            This function creates and initializes entity and relation embeddings (with size k). 
            Overload this function if the parameters needs to be initialized differently.
        """
        self.ent_emb = tf.get_variable('ent_emb', shape=[len(self.ent_to_idx), self.k],
                                       initializer=self.initializer)
        self.rel_emb = tf.get_variable('rel_emb', shape=[len(self.rel_to_idx), self.k],
                                       initializer=self.initializer)

    def _get_model_loss(self, scores_pos, scores_neg):
        """ Get the current batch loss including loss due to regularization.
            This function must be overridden if the model uses combination of different losses(eg: VAE) 
            
        Parameters
        ----------
        scores_pos : tf.Tensor
            A tensor of scores assigned to positive statements.
        scores_neg : tf.Tensor
            A tensor of scores assigned to negative statements.
            
        Returns
        -------
        loss : tf.Tensor
            The loss value that must be minimized.    
        """

        if self.regularizer is not None:
            return self.loss.apply(scores_pos, scores_neg) + \
                   self.regularizer.apply([self.ent_emb, self.rel_emb])
        else:
            return self.loss.apply(scores_pos, scores_neg)

    def _initialize_early_stopping(self):
        """ Initializes and creates evaluation graph for early stopping
        """
        try:
            self.x_valid = self.early_stopping_params['x_valid']
            if type(self.x_valid) != np.ndarray:
                msg = 'Invalid type for input x_valid. Expected ndarray, got {}'.format(type(self.x_valid))
                logger.error(msg)
                raise ValueError(msg)

            if self.x_valid.ndim <= 1 or (np.shape(self.x_valid)[1]) != 3:
                msg = 'Invalid size for input x_valid. Expected (n,3):  got {}'.format(np.shape(self.x_valid))
                logger.error(msg)
                raise ValueError(msg)
            self.x_valid = to_idx(self.x_valid, ent_to_idx=self.ent_to_idx, rel_to_idx=self.rel_to_idx)

        except KeyError:
            msg = 'x_valid must be passed for early fitting.'
            logger.error(msg)
            raise KeyError(msg)

        self.early_stopping_criteria = self.early_stopping_params.get('criteria', DEFAULT_CRITERIA_EARLY_STOPPING)
        if self.early_stopping_criteria not in ['hits10', 'hits1', 'hits3', 'mrr']:
            msg = 'Unsupported early stopping criteria.'
            logger.error(msg)
            raise ValueError(msg)
            
        self.eval_config['corruption_entities'] = self.early_stopping_params.get('corruption_entities', 
                                                                                 DEFAULT_CORRUPTION_ENTITIES)
        self.eval_config['corrupt_side'] = self.early_stopping_params.get('corrupt_side', DEFAULT_CORRUPT_SIDE)

        self.early_stopping_best_value = INITIAL_EARLY_STOPPING_CRITERIA_VALUE
        self.early_stopping_stop_counter = 0
        try:
            x_filter = self.early_stopping_params['x_filter']
            x_filter = to_idx(x_filter, ent_to_idx=self.ent_to_idx, rel_to_idx=self.rel_to_idx)
            self.set_filter_for_eval(x_filter)
        except KeyError:
            logger.debug('x_filter not found in early_stopping_params.')
            pass

        self._initialize_eval_graph()

    def _perform_early_stopping_test(self, epoch):
        """perform regular validation checks and stop early if the criteria is acheived
        Parameters
        ----------
        epoch : int 
            current training epoch
        Returns
        -------
        stopped: bool
            Flag to indicate if the early stopping criteria is acheived
        """

        if epoch >= self.early_stopping_params.get('burn_in', DEFAULT_BURN_IN_EARLY_STOPPING) \
                and epoch % self.early_stopping_params.get('check_interval',
                                                           DEFAULT_CHECK_INTERVAL_EARLY_STOPPING) == 0:
            # compute and store test_loss
            ranks = []

            for x_test_triple in self.x_valid:
                rank_triple = self.sess_train.run([self.rank], feed_dict={self.X_test_tf: [x_test_triple]})
                ranks.append(rank_triple)

            if self.early_stopping_criteria == 'hits10':
                current_test_value = hits_at_n_score(ranks, 10)
            elif self.early_stopping_criteria == 'hits3':
                current_test_value = hits_at_n_score(ranks, 3)
            elif self.early_stopping_criteria == 'hits1':
                current_test_value = hits_at_n_score(ranks, 1)
            elif self.early_stopping_criteria == 'mrr':
                current_test_value = mrr_score(ranks)

            if self.early_stopping_best_value >= current_test_value:
                self.early_stopping_stop_counter += 1
                if self.early_stopping_stop_counter == self.early_stopping_params.get('stop_interval',
                                                                                      DEFAULT_STOP_INTERVAL_EARLY_STOPPING):

                    # If the best value for the criteria has not changed from initial value then
                    # save the model before early stopping
                    if self.early_stopping_best_value == INITIAL_EARLY_STOPPING_CRITERIA_VALUE:
                        self._save_trained_params()

                    if self.verbose:
                        msg = 'Early stopping at epoch:{}'.format(epoch)
                        logger.info(msg)
                        print(msg)
                        msg = 'Best {}: {:10f}'.format(self.early_stopping_criteria, self.early_stopping_best_value)
                        logger.info(msg)
                        print(msg)
                    return True
            else:
                self.early_stopping_best_value = current_test_value
                self.early_stopping_stop_counter = 0
                self._save_trained_params()

            if self.verbose:
                msg = 'Current best:{}'.format(self.early_stopping_best_value)
                logger.info(msg)
                print(msg)
                msg = 'Current:{}'.format(current_test_value)
                logger.info(msg)
                print(msg)

        return False
    
    def _end_training(self):
        """Perform clean up tasks after training.
        """
        # Reset this variable as it is reused during evaluation phase
        self.is_filtered = False
        self.eval_config = {}
        
        #close the tf session
        self.sess_train.close()
        
        #set is_fitted to true to indicate that the model fitting is completed
        self.is_fitted = True
        

    def fit(self, X, early_stopping=False, early_stopping_params={}):
        """Train an EmbeddingModel (with optional early stopping).

            The model is trained on a training set X using the training protocol
            described in :cite:`trouillon2016complex`.
   
        Parameters
        ----------
        X : ndarray, shape [n, 3]
            The training triples
        early_stopping: bool
            Flag to enable early stopping (default:``False``)
        early_stopping_params: dictionary
            Dictionary of hyperparameters for the early stopping heuristics.

            The following string keys are supported:

                - **'x_valid'**: ndarray, shape [n, 3] : Validation set to be used for early stopping.
                - **'criteria'**: string : criteria for early stopping 'hits10', 'hits3', 'hits1' or 'mrr'(default).
                - **'x_filter'**: ndarray, shape [n, 3] : Positive triples to use as filter if a 'filtered' early stopping criteria is desired (i.e. filtered-MRR if 'criteria':'mrr'). Note this will affect training time (no filter by default).
                - **'burn_in'**: int : Number of epochs to pass before kicking in early stopping (default: 100).
                - **check_interval'**: int : Early stopping interval after burn-in (default:10).
                - **'stop_interval'**: int : Stop if criteria is performing worse over n consecutive checks (default: 3)
                - **'corruption_entities'**: List of entities to be used for corruptions. If 'all', it uses all entities (default: 'all')
                - **'corrupt_side'**: Specifies which side to corrupt. 's', 'o', 's+o' (default)

                Example: ``early_stopping_params={x_valid=X['valid'], 'criteria': 'mrr'}``

        """
        if type(X) != np.ndarray:
            msg = 'Invalid type for input X. Expected ndarray, got {}'.format(type(X))
            logger.error(msg)
            raise ValueError(msg)

        if (np.shape(X)[1]) != 3:
            msg = 'Invalid size for input X. Expected number of column 3, got {}'.format(np.shape(X)[1])
            logger.error(msg)
            raise ValueError(msg)

        # create internal IDs mappings
        self.rel_to_idx, self.ent_to_idx = create_mappings(X)
        #  convert training set into internal IDs
        X = to_idx(X, ent_to_idx=self.ent_to_idx, rel_to_idx=self.rel_to_idx)
        
        if len(self.ent_to_idx) > ENTITY_WARN_THRESHOLD:
            logger.warn('Number of unique entities are large: {} entities'.format(len(self.ent_to_idx)))
            logger.warn("If you get memory issues don't using early stopping or use reduced set for 'corruption_entities' for early stopping corruption generation!!!")
            

        # This is useful when we re-fit the same model (e.g. retraining in model selection)
        if self.is_fitted:
            tf.reset_default_graph()

        self.sess_train = tf.Session(config=self.tf_config)

        batch_size = X.shape[0] // self.batches_count
        dataset = tf.data.Dataset.from_tensor_slices(X).repeat().batch(batch_size).prefetch(2)
        dataset_iterator = dataset.make_one_shot_iterator()
        # init tf graph/dataflow for training
        # init variables (model parameters to be learned - i.e. the embeddings)
        self._initialize_parameters()

        # training input placeholder
        x_pos_tf = tf.cast(dataset_iterator.get_next(), tf.int32)
        
        #all_ent_tf = tf.squeeze(tf.constant(list(self.ent_to_idx.values()), dtype=tf.int32))
        
        entities_size = 0
        entities_list = None
        
        negative_corruption_entities = self.embedding_model_params.get('negative_corruption_entities',
                                                                       DEFAULT_CORRUPTION_ENTITIES)
        
        if negative_corruption_entities=='all':
            logger.info('Using all entities for generation of corruptions')
            entities_size = len(self.ent_to_idx)
        elif negative_corruption_entities=='batch':
            #default is batch (entities_size=0 and entities_list=None)
            logger.info('Using batch entities for generation of corruptions')
        elif isinstance(negative_corruption_entities, list):
            logger.info('Using the supplied entities for generation of corruptions')
            entities_list=tf.squeeze(tf.constant(negative_corruption_entities, dtype=tf.int32))
        elif isinstance(negative_corruption_entities, int):
            logger.info('Using first {} entities for generation of corruptions'.format(negative_corruption_entities))
            entities_size = negative_corruption_entities
        
        # generate negatives
        x_neg_tf = generate_corruptions_for_fit(x_pos_tf, 
                                                entities_list=entities_list, 
                                                eta=self.eta, 
                                                entities_size=entities_size, 
                                                rnd=self.seed)
        
        if self.loss.get_state('require_same_size_pos_neg'):
            logger.debug('Requires the same size of postive and negative')
            x_pos = tf.reshape(tf.tile(tf.reshape(x_pos_tf, [-1]), [self.eta]), [tf.shape(x_pos_tf)[0] * self.eta, 3])
            batch_size = batch_size * self.eta
        else:
            x_pos = x_pos_tf
        # look up embeddings from input training triples
        e_s_pos, e_p_pos, e_o_pos = self._lookup_embeddings(x_pos)
        e_s_neg, e_p_neg, e_o_neg = self._lookup_embeddings(x_neg_tf)

        scores_neg = self._fn(e_s_neg, e_p_neg, e_o_neg)
        scores_pos = self._fn(e_s_pos, e_p_pos, e_o_pos)

        loss = self._get_model_loss(scores_pos, scores_neg)

        train = self.optimizer.minimize(loss)

        # Entity embeddings normalization
        normalize_ent_emb_op = self.ent_emb.assign(tf.clip_by_norm(self.ent_emb, clip_norm=1, axes=1))

        self.early_stopping_params = early_stopping_params

        # early stopping
        if early_stopping:
            self._initialize_early_stopping()

        self.sess_train.run(tf.tables_initializer())
        self.sess_train.run(tf.global_variables_initializer())

        # X_batches = np.array_split(X, self.batches_count)

        normalize_rel_emb_op = self.rel_emb.assign(tf.clip_by_norm(self.rel_emb, clip_norm=1, axes=1))

        if self.embedding_model_params.get('normalize_ent_emb', DEFAULT_NORMALIZE_EMBEDDINGS):
            self.sess_train.run(normalize_rel_emb_op)
            self.sess_train.run(normalize_ent_emb_op)

        for epoch in tqdm(range(1, self.epochs + 1), disable=(not self.verbose), unit='epoch'):
            losses = []
            for batch in range(1, self.batches_count + 1):
                loss_batch, _ = self.sess_train.run([loss, train])

                if np.isnan(loss_batch) or np.isinf(loss_batch):
                    msg = 'Loss is {}. Please change the hyperparameters!!!'.format(loss_batch)
                    logger.error(msg)
                    raise ValueError(msg)

                losses.append(loss_batch)
                if self.embedding_model_params.get('normalize_ent_emb', DEFAULT_NORMALIZE_EMBEDDINGS):
                    self.sess_train.run(normalize_ent_emb_op)
            if self.verbose:
                msg = 'epoch: {}: mean loss: {:10f}'.format(epoch, sum(losses) / (batch_size * self.batches_count))
                logger.info(msg)
                tqdm.write(msg)

            if early_stopping:
                if self._perform_early_stopping_test(epoch):
                    self._end_training()
                    return

        self._save_trained_params()
        self._end_training()
        

    def set_filter_for_eval(self, x_filter):
        """Set the filter to be used during evaluation (filtered_corruption = corruptions - filter).
       
        We would be using a prime number based assignment and product for do the filtering.
        We associate a unique prime number for subject entities, object entities and to relations.
        Product of three prime numbers is divisible only by those three prime numbers.
        So we generate this product for the filter triples and store it in a hash map.
        When corruptions are generated for a triple during evaluation, we follow a similar approach 
        and look up the product of corruption in the above hash table. If the corrupted triple is 
        present in the hashmap, it means that it was present in the filter list.
        
        Parameters
        ----------
        x_filter : ndarray, shape [n, 3]
            Filter triples. If the generated corruptions are present in this, they will be removed.

        """
        self.x_filter = x_filter

        entity_size = len(self.ent_to_idx)
        reln_size = len(self.rel_to_idx)

        first_million_primes_list = []
        curr_dir, _ = os.path.split(__file__)
        with open(os.path.join(curr_dir, "prime_number_list.txt"), "r") as f:
            logger.debug('Reading from prime_number_list.txt.')
            line = f.readline()
            i = 0
            for line in f:
                p_nums_line = line.split(' ')
                first_million_primes_list.extend([np.int64(x) for x in p_nums_line if x != '' and x != '\n'])
                if len(first_million_primes_list) > (2 * entity_size + reln_size):
                    break

        # subject
        self.entity_primes_left = first_million_primes_list[:entity_size]
        # obj
        self.entity_primes_right = first_million_primes_list[entity_size:2 * entity_size]
        # reln
        self.relation_primes = first_million_primes_list[2 * entity_size:(2 * entity_size + reln_size)]

        self.filter_keys = []
        # subject
        self.filter_keys = [self.entity_primes_left[self.x_filter[i, 0]] for i in range(self.x_filter.shape[0])]
        # obj
        self.filter_keys = [self.filter_keys[i] * self.entity_primes_right[self.x_filter[i, 2]]
                            for i in range(self.x_filter.shape[0])]
        # reln
        self.filter_keys = [self.filter_keys[i] * self.relation_primes[self.x_filter[i, 1]]
                            for i in range(self.x_filter.shape[0])]

        self.is_filtered = True

    def configure_evaluation_protocol(self, config={'corruption_entities': DEFAULT_CORRUPTION_ENTITIES, \
                                                    'corrupt_side':DEFAULT_CORRUPT_SIDE}):
        """ Set the configuration for evaluation
        
        Parameters
        ----------
        config : dictionary
            Dictionary of parameters for evaluation configuration. Can contain following keys:
            
            - **corruption_entities**: List of entities to be used for corruptions. If ``all``, it uses all entities (default: ``all``)
            - **corrupt_side**: Specifies which side to corrupt. ``s``, ``o``, ``s+o`` (default)
            
        """
        self.eval_config = config

    def _initialize_eval_graph(self):
        """Initialize the evaluation graph. 
        
        Use prime number based filtering strategy (refer set_filter_for_eval()), if the filter is set
        """
        self.X_test_tf = tf.placeholder(tf.int64, shape=[1, 3])

        self.table_entity_lookup_left = None
        self.table_entity_lookup_right = None
        self.table_reln_lookup = None

        all_entities_np = np.int64(np.arange(len(self.ent_to_idx)))

        if self.is_filtered:
            all_reln_np = np.int64(np.arange(len(self.rel_to_idx)))
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

            # Create table to store train+test+valid triplet prime values(product)
            self.table_filter_lookup = tf.contrib.lookup.HashTable(
                tf.contrib.lookup.KeyValueTensorInitializer(np.array(self.filter_keys, dtype=np.int64),
                                                            np.zeros(len(self.filter_keys), dtype=np.int64))
                , 1)

        corruption_entities = self.eval_config.get('corruption_entities', DEFAULT_CORRUPTION_ENTITIES)

        if corruption_entities == 'all':
            corruption_entities = all_entities_np
        elif isinstance(corruption_entities, list):
            corruption_entities = corruption_entities
        else:
            msg = 'Invalid type for corruption entities!!!'
            logger.error(msg)
            raise ValueError(msg)

        self.corruption_entities_tf = tf.constant(corruption_entities, dtype=tf.int64)

        self.out_corr, self.out_corr_prime = generate_corruptions_for_eval(self.X_test_tf,
                                                                           self.corruption_entities_tf,
                                                                           self.eval_config.get('corrupt_side',
                                                                                                DEFAULT_CORRUPT_SIDE),
                                                                           self.table_entity_lookup_left,
                                                                           self.table_entity_lookup_right,
                                                                           self.table_reln_lookup)

        if self.is_filtered:
            # check if corruption prime product is present in dataset prime product
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
        """End the evaluation and close the Tensorflow session.
        """
        if self.sess_predict is not None:
            self.sess_predict.close()
        self.sess_predict = None
        self.is_filtered = False
        self.eval_config = {}

    def predict(self, X, from_idx=False, get_ranks=False):
        """Predict the scores of triples using a trained embedding model.

             The function returns raw scores generated by the model.

             .. note::

                 To obtain probability estimates, use a logistic sigmoid: ::

                     >>> model.fit(X)
                     >>> y_pred = model.predict(np.array([['f', 'y', 'e'], ['b', 'y', 'd']]))
                     >>> print(y_pred)
                     array([1.2052395, 1.5818497], dtype=float32)
                     >>> from scipy.special import expit
                     >>> expit(y_pred)
                     array([0.7694556 , 0.82946634], dtype=float32)


         Parameters
         ----------
         X : ndarray, shape [n, 3]
             The triples to score.
         from_idx : bool
             If True, will skip conversion to internal IDs. (default: False).
         get_ranks : bool
             Flag to compute ranks by scoring against corruptions (default: False).

         Returns
         -------
         scores_predict : ndarray, shape [n]
             The predicted scores for input triples X.

         rank : ndarray, shape [n]
             Ranks of the triples (only returned if ``get_ranks=True``.

        """

        if type(X) != np.ndarray:
            msg = 'Invalid type for input X. Expected ndarray, got {}'.format(type(X))
            logger.error(msg)
            raise ValueError(msg)

        if not self.is_fitted:
            msg = 'Model has not been fitted.'
            logger.error(msg)
            raise RuntimeError(msg)

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
        if X.ndim > 1:
            for x in X:
                all_scores = self.sess_predict.run(self.scores_predict, feed_dict={self.X_test_tf: [x]})
                scores.append(all_scores[0])

                if get_ranks:
                    rank = self.sess_predict.run(self.rank, feed_dict={self.X_test_tf: [x]})
                    ranks.append(rank)
        else:
            all_scores = self.sess_predict.run(self.scores_predict, feed_dict={self.X_test_tf: [X]})
            scores = all_scores[0]
            if get_ranks:
                ranks = self.sess_predict.run(self.rank, feed_dict={self.X_test_tf: [X]})

        if get_ranks:
            return scores, ranks

        return scores


@register_model("RandomBaseline")
class RandomBaseline(EmbeddingModel):
    """Random baseline

        A dummy model that assigns a pseudo-random score included between 0 and 1,
        drawn from a uniform distribution.

        The model is useful whenever you need to compare the performance of
        another model on a custom knowledge graph, and no other baseline is available. 
        
        .. note:: Although the model still requires invoking the ``fit()`` method,
            no actual training will be carried out.

        Examples
        --------
        >>> import numpy as np
        >>> from ampligraph.latent_features import RandomBaseline
        >>> model = RandomBaseline()
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
        [0.5488135039273248, 0.7151893663724195]
    """

    def __init__(self, seed=DEFAULT_SEED):
        """ Initialize the model
        
        Parameters
        ----------
        seed : int
            The seed used by the internal random numbers generator.
            
        """
        self.seed = seed
        self.is_fitted = False
        self.rnd = check_random_state(self.seed)
        self.eval_config = {}

    def _fn(e_s, e_p, e_o):
        pass
    
    def get_embeddings(self, entities, type='entity'):
        """Get the embeddings of entities or relations.

        Parameters
        ----------
        entities : array-like, dtype=int, shape=[n]
            The entities (or relations) of interest. Element of the vector must be the original string literals, and
            not internal IDs.
        type : string
            If 'entity', will consider input as KG entities. If `relation`, they will be treated as KG predicates.

        Returns
        -------
        embeddings : None
            Returns None as this model does not have any embeddings. While scoring, it creates a random score for a triplet.

        """
        return None

    def fit(self, X):
        """Train the random model

        Parameters
        ----------
        X : ndarray, shape [n, 3]
            The training triples
        """
        self.rel_to_idx, self.ent_to_idx = create_mappings(X)
        self.is_fitted = True

    def end_evaluation(self):
        """End the evaluation
        """
        self.is_filtered = False
        self.eval_config = {}

    def predict(self, X, from_idx=False, get_ranks=False):
        """Assign random scores to candidate triples and then ranks them

        Parameters
        ----------
        X : ndarray, shape [n, 3]
            The triples to score.
        from_idx : bool
            If True, will skip conversion to internal IDs. (default: False).
        get_ranks : bool
            Flag to compute ranks by scoring against corruptions (default: False).
            
        Returns
        -------
        scores : ndarray, shape [n]
            The predicted scores for input triples X.
            
        ranks : ndarray, shape [n]
            Rank of the triple

        """
        if X.ndim == 1:
            X = np.expand_dims(X, 0)

        positive_scores = self.rnd.uniform(low=0, high=1, size=len(X)).tolist()
        if get_ranks:
            corruption_entities = self.eval_config.get('corruption_entities', DEFAULT_CORRUPTION_ENTITIES)
            if corruption_entities is None:
                corruption_length = len(self.ent_to_idx)
            else:
                corruption_length = len(corruption_entities)

            corrupt_side = self.eval_config.get('corrupt_side', DEFAULT_CORRUPT_SIDE)
            if corrupt_side == 's+o':
                # since we are corrupting both subject and object
                corruption_length *= 2
                # to account for the positive that we are testing
                corruption_length -= 2
            else:
                # to account for the positive that we are testing
                corruption_length -= 1
            ranks = []
            for i in range(len(X)):
                rank = np.sum(self.rnd.uniform(low=0, high=1, size=corruption_length) >= positive_scores[i]) + 1
                ranks.append(rank)

            return positive_scores, ranks

        return positive_scores


@register_model("TransE", ["norm", "normalize_ent_emb", "negative_corruption_entities"])
class TransE(EmbeddingModel):
    """Translating Embeddings (TransE)

        The model as described in :cite:`bordes2013translating`.


        The scoring function of TransE computes a similarity between the embedding of the subject
        :math:`\mathbf{e}_{sub}` translated by the embedding of the predicate :math:`\mathbf{e}_{pred}`
        and the embedding of the object :math:`\mathbf{e}_{obj}`,
        using the :math:`L_1` or :math:`L_2` norm :math:`||\cdot||`:

        .. math::

            f_{TransE}=-||\mathbf{e}_{sub} + \mathbf{e}_{pred} - \mathbf{e}_{obj}||_n


        Such scoring function is then used on positive and negative triples :math:`t^+, t^-` in the loss function.


        Examples
        --------
        >>> import numpy as np
        >>> from ampligraph.latent_features import TransE
        >>> model = TransE(batches_count=1, seed=555, epochs=20, k=10, loss='pairwise',
        >>>                loss_params={'margin':5})
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
        [-2.219729, -3.9848995]
        >>> model.get_embeddings(['f','e'], type='entity')
        array([[-0.65229136, -0.50060457,  1.2316223 ,  0.23738968,  0.29145557,
        -0.20187911, -0.3053819 , -0.6947149 ,  0.9377473 ,  0.12985024],
        [-1.1272118 ,  0.10723944,  0.79431695,  0.6795645 , -0.14428931,
        -0.34959725, -0.60184777, -1.1885864 ,  1.0374763 , -0.36612505]],
        dtype=float32)

    """

    def __init__(self, 
                 k=DEFAULT_EMBEDDING_SIZE, 
                 eta=DEFAULT_ETA, 
                 epochs=DEFAULT_EPOCH, 
                 batches_count=DEFAULT_BATCH_COUNT, 
                 seed=DEFAULT_SEED,
                 embedding_model_params={'norm':DEFAULT_NORM_TRANSE, 
                                         'normalize_ent_emb':DEFAULT_NORMALIZE_EMBEDDINGS,
                                         'negative_corruption_entities':DEFAULT_CORRUPTION_ENTITIES},
                 optimizer=DEFAULT_OPTIM, 
                 optimizer_params={'lr':DEFAULT_LR},
                 loss=DEFAULT_LOSS, 
                 loss_params={},
                 regularizer=DEFAULT_REGULARIZER, 
                 regularizer_params={},
                 verbose=DEFAULT_VERBOSE):
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
            TransE-specific hyperparams, passed to the model as a dictionary.

            Supported keys:

            - **'norm'** (int): the norm to be used in the scoring function (1 or 2-norm - default: 1).
            - **'normalize_ent_emb'** (bool): flag to indicate whether to normalize entity embeddings after each batch update (default: False).
            - **negative_corruption_entities** : entities to be used for generation of corruptions while training. It can take the following values : ``all`` (default: all entities), ``batch`` (entities present in each batch), list of entities or an int (which indicates how many entities that should be used for corruption generation).

            Example: ``embedding_model_params={'norm': 1, 'normalize_ent_emb': False}``


            
            
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

            - 'pairwise'  the model will use pairwise margin-based loss function.
            - 'nll' the model will use negative loss likelihood.
            - 'absolute_margin' the model will use absolute margin likelihood.
            - 'self_adversarial' the model will use adversarial sampling loss function.
            
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

        verbose : bool
            Verbose mode
        """
        super().__init__(k=k, eta=eta, epochs=epochs, batches_count=batches_count, seed=seed,
                         embedding_model_params=embedding_model_params,
                         optimizer=optimizer, optimizer_params=optimizer_params,
                         loss=loss, loss_params=loss_params,
                         regularizer=regularizer, regularizer_params=regularizer_params,
                         verbose=verbose)

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

        return tf.negative(
            tf.norm(e_s + e_p - e_o, ord=self.embedding_model_params.get('norm', DEFAULT_NORM_TRANSE), axis=1))

    def fit(self, X, early_stopping=False, early_stopping_params={}):
        """Train an Translating Embeddings model.

            The model is trained on a training set X using the training protocol
            described in :cite:`trouillon2016complex`.

        Parameters
        ----------
        X : ndarray, shape [n, 3]
            The training triples
        early_stopping: bool
            Flag to enable early stopping (default:``False``)
        early_stopping_params: dictionary
            Dictionary of hyperparameters for the early stopping heuristics.

            The following string keys are supported:

                - **'x_valid'**: ndarray, shape [n, 3] : Validation set to be used for early stopping.
                - **'criteria'**: string : criteria for early stopping 'hits10', 'hits3', 'hits1' or 'mrr'(default).
                - **'x_filter'**: ndarray, shape [n, 3] : Positive triples to use as filter if a 'filtered' early stopping criteria is desired (i.e. filtered-MRR if 'criteria':'mrr'). Note this will affect training time (no filter by default).
                - **'burn_in'**: int : Number of epochs to pass before kicking in early stopping (default: 100).
                - **check_interval'**: int : Early stopping interval after burn-in (default:10).
                - **'stop_interval'**: int : Stop if criteria is performing worse over n consecutive checks (default: 3)
                - **'corruption_entities'**: List of entities to be used for corruptions. If 'all', it uses all entities (default: 'all')
                - **'corrupt_side'**: Specifies which side to corrupt. 's', 'o', 's+o' (default)

                Example: ``early_stopping_params={x_valid=X['valid'], 'criteria': 'mrr'}``


        """
        super().fit(X, early_stopping, early_stopping_params)

    def predict(self, X, from_idx=False, get_ranks=False):
        """Predict the scores of triples using a trained embedding model.

             The function returns raw scores generated by the model.

             .. note::

                 To obtain probability estimates, use a logistic sigmoid: ::

                     >>> model.fit(X)
                     >>> y_pred = model.predict(np.array([['f', 'y', 'e'], ['b', 'y', 'd']]))
                     >>> print(y_pred)
                     array([1.2052395, 1.5818497], dtype=float32)
                     >>> from scipy.special import expit
                     >>> expit(y_pred)
                     array([0.7694556 , 0.82946634], dtype=float32)


         Parameters
         ----------
         X : ndarray, shape [n, 3]
             The triples to score.
         from_idx : bool
             If True, will skip conversion to internal IDs. (default: False).
         get_ranks : bool
             Flag to compute ranks by scoring against corruptions (default: False).

         Returns
         -------
         scores_predict : ndarray, shape [n]
             The predicted scores for input triples X.

         rank : ndarray, shape [n]
             Ranks of the triples (only returned if ``get_ranks=True``.

        """
        return super().predict(X, from_idx=from_idx, get_ranks=get_ranks)


@register_model("DistMult", ["normalize_ent_emb", "negative_corruption_entities"])
class DistMult(EmbeddingModel):
    """The DistMult model

        The model as described in :cite:`yang2014embedding`.

        The bilinear diagonal DistMult model uses the trilinear dot product as scoring function:

        .. math::

            f_{DistMult}=\langle \mathbf{r}_p, \mathbf{e}_s, \mathbf{e}_o \\rangle

        where :math:`\mathbf{e}_{s}` is the embedding of the subject, :math:`\mathbf{r}_{p}` the embedding
        of the predicate and :math:`\mathbf{e}_{o}` the embedding of the object.


        Examples
        --------
        >>> import numpy as np
        >>> from ampligraph.latent_features import DistMult
        >>> model = DistMult(batches_count=1, seed=555, epochs=20, k=10, loss='pairwise',
        >>>         loss_params={'margin':5})
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
        [3.29703, -3.543957]
        >>> model.get_embeddings(['f','e'], type='entity')
        array([[-0.7101061 , -0.35752687,  0.5337027 , -0.612499  , -0.34532365,
        -0.7219143 , -0.07083285,  0.19323194,  1.0108972 ,  0.42850104],
        [-1.2280471 , -0.22018537,  0.17179069,  0.757755  , -0.05845603,
         0.94373196, -0.14994079, -0.929564  ,  1.0907435 ,  0.20400602]],
        dtype=float32)

    """

    def __init__(self, 
                 k=DEFAULT_EMBEDDING_SIZE, 
                 eta=DEFAULT_ETA, 
                 epochs=DEFAULT_EPOCH, 
                 batches_count=DEFAULT_BATCH_COUNT, 
                 seed=DEFAULT_SEED,
                 embedding_model_params={'normalize_ent_emb':DEFAULT_NORMALIZE_EMBEDDINGS,
                                         'negative_corruption_entities':DEFAULT_CORRUPTION_ENTITIES},
                 optimizer=DEFAULT_OPTIM, 
                 optimizer_params={'lr':DEFAULT_LR},
                 loss=DEFAULT_LOSS, 
                 loss_params={},
                 regularizer=DEFAULT_REGULARIZER, 
                 regularizer_params={},
                 verbose=DEFAULT_VERBOSE):
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
            DistMult-specific hyperparams, passed to the model as a dictionary.

            Supported keys:

            - **'normalize_ent_emb'** (bool): flag to indicate whether to normalize entity embeddings after each batch update (default: False).
            - **'negative_corruption_entities'** - Entities to be used for generation of corruptions while training. It can take the following values : ``all`` (default: all entities), ``batch`` (entities present in each batch), list of entities or an int (which indicates how many entities that should be used for corruption generation).

            Example: ``embedding_model_params={'norm': 1, 'normalize_ent_emb': False}``

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

            - 'pairwise'  the model will use pairwise margin-based loss function.
            - 'nll' the model will use negative loss likelihood.
            - 'absolute_margin' the model will use absolute margin likelihood.
            - 'self_adversarial' the model will use adversarial sampling loss function.

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

        verbose : bool
            Verbose mode
        """
        super().__init__(k=k, eta=eta, epochs=epochs, batches_count=batches_count, seed=seed,
                         embedding_model_params=embedding_model_params,
                         optimizer=optimizer, optimizer_params=optimizer_params,
                         loss=loss, loss_params=loss_params,
                         regularizer=regularizer, regularizer_params=regularizer_params,
                         verbose=verbose)

    def _fn(self, e_s, e_p, e_o):
        """DistMult

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
            The operation corresponding to the DistMult scoring function.

        """

        return tf.reduce_sum(e_s * e_p * e_o, axis=1)

    def fit(self, X, early_stopping=False, early_stopping_params={}):
        """Train an DistMult.

            The model is trained on a training set X using the training protocol
            described in :cite:`trouillon2016complex`.

        Parameters
        ----------
        X : ndarray, shape [n, 3]
            The training triples
        early_stopping: bool
            Flag to enable early stopping (default:``False``)
        early_stopping_params: dictionary
            Dictionary of hyperparameters for the early stopping heuristics.

            The following string keys are supported:

                - **'x_valid'**: ndarray, shape [n, 3] : Validation set to be used for early stopping.
                - **'criteria'**: string : criteria for early stopping 'hits10', 'hits3', 'hits1' or 'mrr'(default).
                - **'x_filter'**: ndarray, shape [n, 3] : Positive triples to use as filter if a 'filtered' early stopping criteria is desired (i.e. filtered-MRR if 'criteria':'mrr'). Note this will affect training time (no filter by default).
                - **'burn_in'**: int : Number of epochs to pass before kicking in early stopping (default: 100).
                - **check_interval'**: int : Early stopping interval after burn-in (default:10).
                - **'stop_interval'**: int : Stop if criteria is performing worse over n consecutive checks (default: 3)
                - **'corruption_entities'**: List of entities to be used for corruptions. If 'all', it uses all entities (default: 'all')
                - **'corrupt_side'**: Specifies which side to corrupt. 's', 'o', 's+o' (default)

                Example: ``early_stopping_params={x_valid=X['valid'], 'criteria': 'mrr'}``

        """
        super().fit(X, early_stopping, early_stopping_params)

    def predict(self, X, from_idx=False, get_ranks=False):
        """Predict the scores of triples using a trained embedding model.

            The function returns raw scores generated by the model.

            .. note::

                To obtain probability estimates, use a logistic sigmoid: ::

                    >>> model.fit(X)
                    >>> y_pred = model.predict(np.array([['f', 'y', 'e'], ['b', 'y', 'd']]))
                    >>> print(y_pred)
                    array([1.2052395, 1.5818497], dtype=float32)
                    >>> from scipy.special import expit
                    >>> expit(y_pred)
                    array([0.7694556 , 0.82946634], dtype=float32)


        Parameters
        ----------
        X : ndarray, shape [n, 3]
            The triples to score.
        from_idx : bool
            If True, will skip conversion to internal IDs. (default: False).
        get_ranks : bool
            Flag to compute ranks by scoring against corruptions (default: False).

        Returns
        -------
        scores_predict : ndarray, shape [n]
            The predicted scores for input triples X.
            
        rank : ndarray, shape [n]
            Ranks of the triples (only returned if ``get_ranks=True``.

        """
        return super().predict(X, from_idx=from_idx, get_ranks=get_ranks)


@register_model("ComplEx", ["negative_corruption_entities"])
class ComplEx(EmbeddingModel):
    """ Complex embeddings (ComplEx)

        The ComplEx model :cite:`trouillon2016complex` is an extension of
        the :class:`ampligraph.latent_features.DistMult` bilinear diagonal model
        . ComplEx scoring function is based on the trilinear Hermitian dot product in :math:`\mathcal{C}`:

        .. math::

            f_{ComplEx}=Re(\langle \mathbf{r}_p, \mathbf{e}_s, \overline{\mathbf{e}_o}  \\rangle)

        Note that because embeddings are in :math:`\mathcal{C}`, ComplEx uses twice as many parameters as
        :class:`ampligraph.latent_features.DistMult`.

        Examples
        --------
        >>> import numpy as np
        >>> from ampligraph.latent_features import ComplEx
        >>>
        >>> model = ComplEx(batches_count=1, seed=555, epochs=20, k=10, 
        >>>             loss='pairwise', loss_params={'margin':1}, 
        >>>             regularizer='LP', regularizer_params={'lambda':0.1})
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
        [0.96325016, -0.17629346]
        >>> model.get_embeddings(['f','e'], type='entity')
        array([[-0.11257   , -0.09226837,  0.2829331 , -0.02094189,  0.02826234,
        -0.3068198 , -0.41022655, -0.23714773, -0.00084166,  0.22521858,
        -0.48155236,  0.29627186,  0.29841757,  0.16540456,  0.45836073,
         0.14025007, -0.03458257, -0.03813137,  0.35438442, -0.4733188 ],
        [ 0.06088537,  0.13615245, -0.20476362,  0.20391239,  0.22199424,
         0.5762486 , -0.01087974,  0.39070424, -0.1372974 ,  0.39998057,
        -0.5944237 ,  0.506474  ,  0.1255992 , -0.06021457, -0.26678884,
        -0.18713273,  0.36862013,  0.07165384, -0.00845572, -0.16494963]],
        dtype=float32)

    """
    def __init__(self, 
                 k=DEFAULT_EMBEDDING_SIZE, 
                 eta=DEFAULT_ETA, 
                 epochs=DEFAULT_EPOCH, 
                 batches_count=DEFAULT_BATCH_COUNT, 
                 seed=DEFAULT_SEED,
                 embedding_model_params={'negative_corruption_entities':DEFAULT_CORRUPTION_ENTITIES},
                 optimizer=DEFAULT_OPTIM, 
                 optimizer_params={'lr':DEFAULT_LR},
                 loss=DEFAULT_LOSS, 
                 loss_params={},
                 regularizer=DEFAULT_REGULARIZER, 
                 regularizer_params={},
                 verbose=DEFAULT_VERBOSE):
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
            ComplEx-specific hyperparams:
            
            - **'negative_corruption_entities'** - Entities to be used for generation of corruptions while training. It can take the following values : ``all`` (default: all entities), ``batch`` (entities present in each batch), list of entities or an int (which indicates how many entities that should be used for corruption generation).

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

            - 'pairwise'  the model will use pairwise margin-based loss function.
            - 'nll' the model will use negative loss likelihood.
            - 'absolute_margin' the model will use absolute margin likelihood.
            - 'self_adversarial' the model will use adversarial sampling loss function.

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
        verbose : bool
            Verbose mode
        """
        super().__init__(k=k, eta=eta, epochs=epochs, batches_count=batches_count, seed=seed,
                         embedding_model_params=embedding_model_params,
                         optimizer=optimizer, optimizer_params=optimizer_params,
                         loss=loss, loss_params=loss_params,
                         regularizer=regularizer, regularizer_params=regularizer_params,
                         verbose=verbose)

    def _initialize_parameters(self):
        """ Initialize the complex embeddings.
        """
        self.ent_emb = tf.get_variable('ent_emb', shape=[len(self.ent_to_idx), self.k * 2],
                                       initializer=self.initializer)
        self.rel_emb = tf.get_variable('rel_emb', shape=[len(self.rel_to_idx), self.k * 2],
                                       initializer=self.initializer)

    def _fn(self, e_s, e_p, e_o):
        """ComplEx scoring function.

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
            The operation corresponding to the ComplEx scoring function.

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
        X : ndarray, shape [n, 3]
            The training triples
        early_stopping: bool
            Flag to enable early stopping (default:``False``)
        early_stopping_params: dictionary
            Dictionary of hyperparameters for the early stopping heuristics.

            The following string keys are supported:

                - **'x_valid'**: ndarray, shape [n, 3] : Validation set to be used for early stopping.
                - **'criteria'**: string : criteria for early stopping 'hits10', 'hits3', 'hits1' or 'mrr'(default).
                - **'x_filter'**: ndarray, shape [n, 3] : Positive triples to use as filter if a 'filtered' early stopping criteria is desired (i.e. filtered-MRR if 'criteria':'mrr'). Note this will affect training time (no filter by default).
                - **'burn_in'**: int : Number of epochs to pass before kicking in early stopping (default: 100).
                - **check_interval'**: int : Early stopping interval after burn-in (default:10).
                - **'stop_interval'**: int : Stop if criteria is performing worse over n consecutive checks (default: 3)
                - **'corruption_entities'**: List of entities to be used for corruptions. If 'all', it uses all entities (default: 'all')
                - **'corrupt_side'**: Specifies which side to corrupt. 's', 'o', 's+o' (default)

                Example: ``early_stopping_params={x_valid=X['valid'], 'criteria': 'mrr'}``

        """
        super().fit(X, early_stopping, early_stopping_params)

    def predict(self, X, from_idx=False, get_ranks=False):
        """Predict the scores of triples using a trained embedding model.

             The function returns raw scores generated by the model.

             .. note::

                 To obtain probability estimates, use a logistic sigmoid: ::

                     >>> model.fit(X)
                     >>> y_pred = model.predict(np.array([['f', 'y', 'e'], ['b', 'y', 'd']]))
                     >>> print(y_pred)
                     array([1.2052395, 1.5818497], dtype=float32)
                     >>> from scipy.special import expit
                     >>> expit(y_pred)
                     array([0.7694556 , 0.82946634], dtype=float32)


         Parameters
         ----------
         X : ndarray, shape [n, 3]
             The triples to score.
         from_idx : bool
             If True, will skip conversion to internal IDs. (default: False).
         get_ranks : bool
             Flag to compute ranks by scoring against corruptions (default: False).

         Returns
         -------
         scores_predict : ndarray, shape [n]
             The predicted scores for input triples X.

         rank : ndarray, shape [n]
             Ranks of the triples (only returned if ``get_ranks=True``.

        """
        return super().predict(X, from_idx=from_idx, get_ranks=get_ranks)


@register_model("HolE", ["negative_corruption_entities"])
class HolE(ComplEx):
    """ Holographic Embeddings

    The HolE model :cite:`nickel2016holographic` as re-defined by Hayashi et al. :cite:`HayashiS17`:

    .. math::

        f_{HolE}= \\frac{2}{n} \, f_{ComplEx}

    Examples
    --------
    >>> import numpy as np
    >>> from ampligraph.latent_features import HolE
    >>> model = HolE(batches_count=1, seed=555, epochs=20, k=10,
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
    >>> model.predict(np.array([['f', 'y', 'e'], ['b', 'y', 'd']]), get_ranks=True)
    [0.3046168, -0.0379385]
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
    def __init__(self, 
                 k=DEFAULT_EMBEDDING_SIZE, 
                 eta=DEFAULT_ETA, 
                 epochs=DEFAULT_EPOCH, 
                 batches_count=DEFAULT_BATCH_COUNT, 
                 seed=DEFAULT_SEED,
                 embedding_model_params={'negative_corruption_entities':DEFAULT_CORRUPTION_ENTITIES},
                 optimizer=DEFAULT_OPTIM, 
                 optimizer_params={'lr':DEFAULT_LR},
                 loss=DEFAULT_LOSS, 
                 loss_params={},
                 regularizer=DEFAULT_REGULARIZER, 
                 regularizer_params={},
                 verbose=DEFAULT_VERBOSE):
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
            
            - **negative_corruption_entities** - Entities to be used for generation of corruptions while training. It can take the following values : ``all`` (default: all entities), ``batch`` (entities present in each batch), list of entities or an int (which indicates how many entities that should be used for corruption generation).
            
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

            - 'pairwise'  the model will use pairwise margin-based loss function.
            - 'nll' the model will use negative loss likelihood.
            - 'absolute_margin' the model will use absolute margin likelihood.
            - 'self_adversarial' the model will use adversarial sampling loss function.

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
        verbose : bool
            Verbose mode
        """
        super().__init__(k=k, eta=eta, epochs=epochs, batches_count=batches_count, seed=seed,
                         embedding_model_params=embedding_model_params,
                         optimizer=optimizer, optimizer_params=optimizer_params,
                         loss=loss, loss_params=loss_params,
                         regularizer=regularizer, regularizer_params=regularizer_params,
                         verbose=verbose)

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

    def fit(self, X, early_stopping=False, early_stopping_params={}):
        """Train a HolE model.

            The model is trained on a training set X using the training protocol
            described in :cite:`nickel2016holographic`.

        Parameters
        ----------
        X : ndarray, shape [n, 3]
            The training triples
        early_stopping: bool
            Flag to enable early stopping (default:``False``)
        early_stopping_params: dictionary
            Dictionary of hyperparameters for the early stopping heuristics.

            The following string keys are supported:

                - **'x_valid'**: ndarray, shape [n, 3] : Validation set to be used for early stopping.
                - **'criteria'**: string : criteria for early stopping 'hits10', 'hits3', 'hits1' or 'mrr'(default).
                - **'x_filter'**: ndarray, shape [n, 3] : Positive triples to use as filter if a 'filtered' early stopping criteria is desired (i.e. filtered-MRR if 'criteria':'mrr'). Note this will affect training time (no filter by default).
                - **'burn_in'**: int : Number of epochs to pass before kicking in early stopping (default: 100).
                - **check_interval'**: int : Early stopping interval after burn-in (default:10).
                - **'stop_interval'**: int : Stop if criteria is performing worse over n consecutive checks (default: 3)
                - **'corruption_entities'**: List of entities to be used for corruptions. If 'all', it uses all entities (default: 'all')
                - **'corrupt_side'**: Specifies which side to corrupt. 's', 'o', 's+o' (default)

                Example: ``early_stopping_params={x_valid=X['valid'], 'criteria': 'mrr'}``

        """
        super().fit(X, early_stopping, early_stopping_params)

    def predict(self, X, from_idx=False, get_ranks=False):
        """Predict the scores of triples using a trained embedding model.

             The function returns raw scores generated by the model.

             .. note::

                 To obtain probability estimates, use a logistic sigmoid: ::

                     >>> model.fit(X)
                     >>> y_pred = model.predict(np.array([['f', 'y', 'e'], ['b', 'y', 'd']]))
                     >>> print(y_pred)
                     array([1.2052395, 1.5818497], dtype=float32)
                     >>> from scipy.special import expit
                     >>> expit(y_pred)
                     array([0.7694556 , 0.82946634], dtype=float32)


         Parameters
         ----------
         X : ndarray, shape [n, 3]
             The triples to score.
         from_idx : bool
             If True, will skip conversion to internal IDs. (default: False).
         get_ranks : bool
             Flag to compute ranks by scoring against corruptions (default: False).

         Returns
         -------
         scores_predict : ndarray, shape [n]
             The predicted scores for input triples X.

         rank : ndarray, shape [n]
             Ranks of the triples (only returned if ``get_ranks=True``.

        """
        return super().predict(X, from_idx=from_idx, get_ranks=get_ranks)
