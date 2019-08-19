# Copyright 2019 The AmpliGraph Authors. All Rights Reserved.
#
# This file is Licensed under the Apache License, Version 2.0.
# A copy of the Licence is available in LICENCE, or at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
import numpy as np
import tensorflow as tf
from sklearn.utils import check_random_state
import abc
from tqdm import tqdm
import logging
from ampligraph.latent_features.loss_functions import LOSS_REGISTRY
from ampligraph.latent_features.regularizers import REGULARIZER_REGISTRY
from ampligraph.latent_features.optimizers import OPTIMIZER_REGISTRY, SGDOptimizer
from ampligraph.latent_features.initializers import INITIALIZER_REGISTRY, DEFAULT_XAVIER_IS_UNIFORM
from ampligraph.evaluation import generate_corruptions_for_fit, to_idx, generate_corruptions_for_eval, \
    hits_at_n_score, mrr_score
from ampligraph.datasets import AmpligraphDatasetAdapter, NumpyDatasetAdapter
from functools import partial
from ampligraph.latent_features import constants as constants

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

MODEL_REGISTRY = {}

ENTITY_THRESHOLD = 5e5


def set_entity_threshold(threshold):
    """Sets the entity threshold (threshold after which large graph mode is initiated)
    """
    global ENTITY_THRESHOLD
    ENTITY_THRESHOLD = threshold


def reset_entity_threshold():
    """Resets the entity threshold
    """
    global ENTITY_THRESHOLD
    ENTITY_THRESHOLD = 5e5


def register_model(name, external_params=None, class_params=None):
    if external_params is None:
        external_params = []
    if class_params is None:
        class_params = {}

    def insert_in_registry(class_handle):
        MODEL_REGISTRY[name] = class_handle
        class_handle.name = name
        MODEL_REGISTRY[name].external_params = external_params
        MODEL_REGISTRY[name].class_params = class_params
        return class_handle

    return insert_in_registry


class EmbeddingModel(abc.ABC):
    """Abstract class for embedding models

    AmpliGraph neural knowledge graph embeddings models extend this class and
    its core methods.

    """

    def __init__(self,
                 k=constants.DEFAULT_EMBEDDING_SIZE,
                 eta=constants.DEFAULT_ETA,
                 epochs=constants.DEFAULT_EPOCH,
                 batches_count=constants.DEFAULT_BATCH_COUNT,
                 seed=constants.DEFAULT_SEED,
                 embedding_model_params={},
                 optimizer=constants.DEFAULT_OPTIM,
                 optimizer_params={'lr': constants.DEFAULT_LR},
                 loss=constants.DEFAULT_LOSS,
                 loss_params={},
                 regularizer=constants.DEFAULT_REGULARIZER,
                 regularizer_params={},
                 initializer=constants.DEFAULT_INITIALIZER,
                 initializer_params={'uniform': DEFAULT_XAVIER_IS_UNIFORM},
                 verbose=constants.DEFAULT_VERBOSE):
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
            Model-specific hyperparams, passed to the model as a dictionary.
            Refer to model-specific documentation for details.

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

            - ``pairwise``  the model will use pairwise margin-based loss function.
            - ``nll`` the model will use negative loss likelihood.
            - ``absolute_margin`` the model will use absolute margin likelihood.
            - ``self_adversarial`` the model will use adversarial sampling loss function.
            - ``multiclass_nll`` the model will use multiclass nll loss. Switch to multiclass loss defined in
              :cite:`chen2015` by passing 'corrupt_sides' as ['s','o'] to embedding_model_params.
              To use loss defined in :cite:`kadlecBK17` pass 'corrupt_sides' as 'o' to embedding_model_params.

        loss_params : dict
            Dictionary of loss-specific hyperparameters. See :ref:`loss
            functions <loss>`
            documentation for additional details.

            Example: ``optimizer_params={'lr': 0.01}`` if ``loss='pairwise'``.

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
        self.seed = seed
        self.rnd = check_random_state(self.seed)
        tf.random.set_random_seed(seed)

        self.is_filtered = False
        self.loss_params = loss_params

        self.embedding_model_params = embedding_model_params

        self.k = k
        self.internal_k = k
        self.epochs = epochs
        self.eta = eta
        self.regularizer_params = regularizer_params
        self.batches_count = batches_count

        self.dealing_with_large_graphs = False

        if batches_count == 1:
            logger.warning(
                'All triples will be processed in the same batch (batches_count=1). '
                'When processing large graphs it is recommended to batch the input knowledge graph instead.')

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

        try:
            self.optimizer = OPTIMIZER_REGISTRY[optimizer](self.optimizer_params,
                                                           self.batches_count,
                                                           verbose)
        except KeyError:
            msg = 'Unsupported optimizer: {}'.format(optimizer)
            logger.error(msg)
            raise ValueError(msg)

        self.verbose = verbose

        self.initializer_params = initializer_params

        try:
            self.initializer = INITIALIZER_REGISTRY[initializer](self.initializer_params,
                                                                 verbose,
                                                                 self.rnd)
        except KeyError:
            msg = 'Unsupported initializer: {}'.format(initializer)
            logger.error(msg)
            raise ValueError(msg)

        self.verbose = verbose

        self.initializer_params = initializer_params

        try:
            self.initializer = INITIALIZER_REGISTRY[initializer](self.initializer_params,
                                                                 verbose,
                                                                 self.rnd)
        except KeyError:
            msg = 'Unsupported initializer: {}'.format(initializer)
            logger.error(msg)
            raise ValueError(msg)

        self.tf_config = tf.ConfigProto(allow_soft_placement=True)
        self.tf_config.gpu_options.allow_growth = True
        self.sess_train = None
        self.sess_predict = None
        self.trained_model_params = []
        self.is_fitted = False
        self.eval_config = {}
        self.eval_dataset_handle = None
        self.train_dataset_handle = None

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
        """Save the model parameters in the dictionary.

        Parameters
        ----------
        output_dict : dictionary
            Dictionary of saved params. 
            It's the duty of the model to save all the variables correctly, so that it can be used for restoring later.
        
        """
        output_dict['model_params'] = self.trained_model_params
        output_dict['large_graph'] = self.dealing_with_large_graphs

    def restore_model_params(self, in_dict):
        """Load the model parameters from the input dictionary.
        
        Parameters
        ----------
        in_dict : dictionary
            Dictionary of saved params. It's the duty of the model to load the variables correctly.
        """

        self.trained_model_params = in_dict['model_params']
        # Try catch is for backward compatibility
        try:
            self.dealing_with_large_graphs = in_dict['large_graph']
        except KeyError:
            # For backward compatibility
            self.dealing_with_large_graphs = False

    def _save_trained_params(self):
        """After model fitting, save all the trained parameters in trained_model_params in some order. 
        The order would be useful for loading the model. 
        This method must be overridden if the model has any other parameters (apart from entity-relation embeddings).
        """
        if not self.dealing_with_large_graphs:
            self.trained_model_params = self.sess_train.run([self.ent_emb, self.rel_emb])
        else:
            self.trained_model_params = [self.ent_emb_cpu, self.sess_train.run(self.rel_emb)]

    def _load_model_from_trained_params(self):
        """Load the model from trained params. 
        While restoring make sure that the order of loaded parameters match the saved order.
        It's the duty of the embedding model to load the variables correctly.
        This method must be overridden if the model has any other parameters (apart from entity-relation embeddings).
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
            self.ent_emb = tf.Variable(self.trained_model_params[0], dtype=tf.float32)
        else:
            self.ent_emb_cpu = self.trained_model_params[0]
            self.ent_emb = tf.Variable(np.zeros((self.batch_size, self.internal_k)), dtype=tf.float32)

        self.rel_emb = tf.Variable(self.trained_model_params[1], dtype=tf.float32)

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
            emb_list = self.trained_model_params[0]
            lookup_dict = self.ent_to_idx
        elif embedding_type == 'relation':
            emb_list = self.trained_model_params[1]
            lookup_dict = self.rel_to_idx
        else:
            msg = 'Invalid entity type: {}'.format(embedding_type)
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
        e_s = self._entity_lookup(x[:, 0])
        e_p = tf.nn.embedding_lookup(self.rel_emb, x[:, 1], name='embedding_lookup_predicate')
        e_o = self._entity_lookup(x[:, 2])
        return e_s, e_p, e_o        

    def _entity_lookup(self, entity):
        """Get the embeddings for entities.
           Remaps the entity indices to corresponding variables in the GPU memory when dealing with large graphs.

        Parameters
        ----------
        entity : nd-tensor, shape [n, 1]
        Returns
        -------
        emb : Tensor
            A Tensor that includes the embeddings of the entities.
        """

        if self.dealing_with_large_graphs:
            remapping = self.sparse_mappings.lookup(entity)
        else:
            remapping = entity

        emb = tf.nn.embedding_lookup(self.ent_emb, remapping)
        return emb

    def _initialize_parameters(self):
        """Initialize parameters of the model.

            This function creates and initializes entity and relation embeddings (with size k).
            If the graph is large, then it loads only the required entity embeddings (max:batch_size*2)
            and all relation embeddings.
            Overload this function if the parameters needs to be initialized differently.
        """
        if not self.dealing_with_large_graphs:

            self.ent_emb = tf.get_variable('ent_emb', shape=[len(self.ent_to_idx), self.internal_k],
                                           initializer=self.initializer.get_tf_initializer(), dtype=tf.float32)
            self.rel_emb = tf.get_variable('rel_emb', shape=[len(self.rel_to_idx), self.internal_k],
                                           initializer=self.initializer.get_tf_initializer(), dtype=tf.float32)
        else:

            self.ent_emb = tf.get_variable('ent_emb', shape=[self.batch_size * 2, self.internal_k],
                                           initializer=self.initializer.get_tf_initializer(), dtype=tf.float32)
            self.rel_emb = tf.get_variable('rel_emb', shape=[self.batch_size * 2, self.internal_k],
                                           initializer=self.initializer.get_tf_initializer(), dtype=tf.float32)

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
        # get the train triples of the batch, unique entities and the corresponding embeddings
        # the latter 2 variables are passed only for large graphs.
        x_pos_tf, self.unique_entities, ent_emb_batch = dataset_iterator.get_next()

        # list of dependent ops that need to be evaluated before computing the loss
        dependencies = []

        # if the graph is large
        if self.dealing_with_large_graphs:
            # Create a dependency to load the embeddings of the batch entities dynamically
            init_ent_emb_batch = self.ent_emb.assign(ent_emb_batch, use_locking=True)
            dependencies.append(init_ent_emb_batch)

            # create a lookup dependency(to remap the entity indices to the corresponding indices of variables in memory
            self.sparse_mappings = tf.contrib.lookup.MutableDenseHashTable(key_dtype=tf.int32, value_dtype=tf.int32,
                                                                           default_value=-1, empty_key=-2,
                                                                           deleted_key=-1)

            insert_lookup_op = self.sparse_mappings.insert(self.unique_entities,
                                                           tf.reshape(tf.range(tf.shape(self.unique_entities)[0],
                                                                      dtype=tf.int32), (-1, 1)))

            dependencies.append(insert_lookup_op)

        # run the dependencies
        with tf.control_dependencies(dependencies):
            entities_size = 0
            entities_list = None

            x_pos = x_pos_tf

            e_s_pos, e_p_pos, e_o_pos = self._lookup_embeddings(x_pos)
            scores_pos = self._fn(e_s_pos, e_p_pos, e_o_pos)

            if self.loss.get_state('require_same_size_pos_neg'):
                logger.debug('Requires the same size of postive and negative')
                scores_pos = tf.reshape(tf.tile(scores_pos, [self.eta]), [tf.shape(scores_pos)[0] * self.eta])

            # look up embeddings from input training triples
            negative_corruption_entities = self.embedding_model_params.get('negative_corruption_entities',
                                                                           constants.DEFAULT_CORRUPTION_ENTITIES)

            if negative_corruption_entities == 'all':
                '''
                if number of entities are large then in this case('all'),
                the corruptions would be generated from batch entities and and additional random entities that
                are selected from all entities (since a total of batch_size*2 entity embeddings are loaded in memory)
                '''
                logger.debug('Using all entities for generation of corruptions during training')
                if self.dealing_with_large_graphs:
                    entities_list = tf.squeeze(self.unique_entities)
                else:
                    entities_size = tf.shape(self.ent_emb)[0]
            elif negative_corruption_entities == 'batch':
                # default is batch (entities_size=0 and entities_list=None)
                logger.debug('Using batch entities for generation of corruptions during training')
            elif isinstance(negative_corruption_entities, list):
                logger.debug('Using the supplied entities for generation of corruptions during training')
                entities_list = tf.squeeze(tf.constant(np.asarray([idx for uri, idx in self.ent_to_idx.items()
                                                                   if uri in negative_corruption_entities]),
                                                       dtype=tf.int32))
            elif isinstance(negative_corruption_entities, int):
                logger.debug('Using first {} entities for generation of corruptions during \
                             training'.format(negative_corruption_entities))
                entities_size = negative_corruption_entities

            loss = 0
            corruption_sides = self.embedding_model_params.get('corrupt_sides', constants.DEFAULT_CORRUPT_SIDE_TRAIN)
            if not isinstance(corruption_sides, list):
                corruption_sides = [corruption_sides]

            for side in corruption_sides:
                # Generate the corruptions
                x_neg_tf = generate_corruptions_for_fit(x_pos_tf,
                                                        entities_list=entities_list,
                                                        eta=self.eta,
                                                        corrupt_side=side,
                                                        entities_size=entities_size,
                                                        rnd=self.seed)

                # compute corruption scores
                e_s_neg, e_p_neg, e_o_neg = self._lookup_embeddings(x_neg_tf)
                scores_neg = self._fn(e_s_neg, e_p_neg, e_o_neg)

                # Apply the loss function
                loss += self.loss.apply(scores_pos, scores_neg)

            if self.regularizer is not None:
                # Apply the regularizer
                loss += self.regularizer.apply([self.ent_emb, self.rel_emb])

            return loss

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

            elif isinstance(self.x_valid, AmpligraphDatasetAdapter):
                # this assumes that the validation data has already been set in the adapter
                self.eval_dataset_handle = self.x_valid
            else:
                msg = 'Invalid type for input X. Expected ndarray/AmpligraphDataset object, \
                       got {}'.format(type(self.x_valid))
                logger.error(msg)
                raise ValueError(msg)
        except KeyError:
            msg = 'x_valid must be passed for early fitting.'
            logger.error(msg)
            raise KeyError(msg)

        self.early_stopping_criteria = self.early_stopping_params.get(
            'criteria', constants.DEFAULT_CRITERIA_EARLY_STOPPING)
        if self.early_stopping_criteria not in ['hits10', 'hits1', 'hits3',
                                                'mrr']:
            msg = 'Unsupported early stopping criteria.'
            logger.error(msg)
            raise ValueError(msg)

        self.eval_config['corruption_entities'] = self.early_stopping_params.get('corruption_entities',
                                                                                 constants.DEFAULT_CORRUPTION_ENTITIES)

        if isinstance(self.eval_config['corruption_entities'], list):
            # convert from list of raw triples to entity indices
            logger.debug('Using the supplied entities for generation of corruptions for early stopping')
            self.eval_config['corruption_entities'] = np.asarray([idx for uri, idx in self.ent_to_idx.items()
                                                                  if uri in self.eval_config['corruption_entities']])
        elif self.eval_config['corruption_entities'] == 'all':
            logger.debug('Using all entities for generation of corruptions for early stopping')
        elif self.eval_config['corruption_entities'] == 'batch':
            logger.debug('Using batch entities for generation of corruptions for early stopping')

        self.eval_config['corrupt_side'] = self.early_stopping_params.get('corrupt_side', 
                                                                          constants.DEFAULT_CORRUPT_SIDE_EVAL)

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
        self._initialize_eval_graph("valid")

    def _perform_early_stopping_test(self, epoch):
        """Performs regular validation checks and stop early if the criteria is achieved.

        Parameters
        ----------
        epoch : int
            current training epoch.
        Returns
        -------
        stopped: bool
            Flag to indicate if the early stopping criteria is achieved.
        """

        if epoch >= self.early_stopping_params.get('burn_in',
                                                   constants.DEFAULT_BURN_IN_EARLY_STOPPING) \
                and epoch % self.early_stopping_params.get('check_interval',
                                                           constants.DEFAULT_CHECK_INTERVAL_EARLY_STOPPING) == 0:
            # compute and store test_loss
            ranks = []

            # Get each triple and compute the rank for that triple
            for x_test_triple in range(self.eval_dataset_handle.get_size("valid")):
                rank_triple = self.sess_train.run(self.rank)
                ranks.append(rank_triple)

            if self.early_stopping_criteria == 'hits10':
                current_test_value = hits_at_n_score(ranks, 10)
            elif self.early_stopping_criteria == 'hits3':
                current_test_value = hits_at_n_score(ranks, 3)
            elif self.early_stopping_criteria == 'hits1':
                current_test_value = hits_at_n_score(ranks, 1)
            elif self.early_stopping_criteria == 'mrr':
                current_test_value = mrr_score(ranks)

            if self.early_stopping_best_value is None:  # First validation iteration
                self.early_stopping_best_value = current_test_value
                self.early_stopping_first_value = current_test_value
            elif self.early_stopping_best_value >= current_test_value:
                self.early_stopping_stop_counter += 1
                if self.early_stopping_stop_counter == self.early_stopping_params.get(
                        'stop_interval', constants.DEFAULT_STOP_INTERVAL_EARLY_STOPPING):

                    # If the best value for the criteria has not changed from
                    #  initial value then
                    # save the model before early stopping
                    if self.early_stopping_best_value == self.early_stopping_first_value:
                        self._save_trained_params()

                    if self.verbose:
                        msg = 'Early stopping at epoch:{}'.format(epoch)
                        logger.info(msg)
                        msg = 'Best {}: {:10f}'.format(
                            self.early_stopping_criteria,
                            self.early_stopping_best_value)
                        logger.info(msg)
                    return True
            else:
                self.early_stopping_best_value = current_test_value
                self.early_stopping_stop_counter = 0
                self._save_trained_params()

            if self.verbose:
                msg = 'Current best:{}'.format(self.early_stopping_best_value)
                logger.debug(msg)
                msg = 'Current:{}'.format(current_test_value)
                logger.debug(msg)

        return False

    def _end_training(self):
        """Performs clean up tasks after training.
        """
        # Reset this variable as it is reused during evaluation phase
        if self.is_filtered and self.eval_dataset_handle is not None:
            # cleanup the evaluation data (deletion of tables
            self.eval_dataset_handle.cleanup()
            self.eval_dataset_handle = None

        if self.train_dataset_handle is not None:
            self.train_dataset_handle.cleanup()
            self.train_dataset_handle = None

        self.is_filtered = False
        self.eval_config = {}

        # close the tf session
        if self.sess_train is not None:
            self.sess_train.close()

        # set is_fitted to true to indicate that the model fitting is completed
        self.is_fitted = True

    def _training_data_generator(self):
        """Generates the training data.
           If we are dealing with large graphs, then along with the training triples (of the batch),
           this method returns the idx of the entities present in the batch (along with filler entities
           sampled randomly from the rest(not in batch) to load batch_size*2 entities on the GPU) and their embeddings.
        """

        all_ent = np.int32(np.arange(len(self.ent_to_idx)))
        unique_entities = all_ent.reshape(-1, 1)
        # generate empty embeddings for smaller graphs - as all the entity embeddings will be loaded on GPU
        entity_embeddings = np.empty(shape=(0, self.internal_k), dtype=np.float32)
        # create iterator to iterate over the train batches
        batch_iterator = iter(self.train_dataset_handle.get_next_train_batch(self.batch_size, "train"))
        for i in range(self.batches_count):
            out = next(batch_iterator)
            # If large graph, load batch_size*2 entities on GPU memory
            if self.dealing_with_large_graphs:
                # find the unique entities - these HAVE to be loaded
                unique_entities = np.int32(np.unique(np.concatenate([out[:, 0], out[:, 2]], axis=0)))
                # Load the remaining entities by randomly selecting from the rest of the entities
                self.leftover_entities = self.rnd.permutation(np.setdiff1d(all_ent, unique_entities))
                needed = (self.batch_size * 2 - unique_entities.shape[0])
                '''
                #this is for debugging
                large_number = np.zeros((self.batch_size-unique_entities.shape[0],
                                             self.ent_emb_cpu.shape[1]), dtype=np.float32) + np.nan

                entity_embeddings = np.concatenate((self.ent_emb_cpu[unique_entities,:],
                                                    large_number), axis=0)
                '''
                unique_entities = np.int32(np.concatenate([unique_entities, self.leftover_entities[:needed]], axis=0))
                entity_embeddings = self.ent_emb_cpu[unique_entities, :]

                unique_entities = unique_entities.reshape(-1, 1)

            yield out, unique_entities, entity_embeddings

    def fit(self, X, early_stopping=False, early_stopping_params={}):
        """Train an EmbeddingModel (with optional early stopping).

        The model is trained on a training set X using the training protocol
        described in :cite:`trouillon2016complex`.

        Parameters
        ----------
        X : ndarray (shape [n, 3]) or object of AmpligraphDatasetAdapter
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
                self.train_dataset_handle = NumpyDatasetAdapter()
                self.train_dataset_handle.set_data(X, "train")
            elif isinstance(X, AmpligraphDatasetAdapter):
                self.train_dataset_handle = X
            else:
                msg = 'Invalid type for input X. Expected ndarray/AmpligraphDataset object, got {}'.format(type(X))
                logger.error(msg)
                raise ValueError(msg)

            # create internal IDs mappings
            self.rel_to_idx, self.ent_to_idx = self.train_dataset_handle.generate_mappings()
            prefetch_batches = 1

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
                    raise Exception("This mode works well only with SGD optimizer with decay (read docs for details).\
 Kindly change the optimizer and restart the experiment")

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
            # dataset = tf.data.Dataset.from_tensor_slices(X).repeat().batch(batch_size).prefetch(2)

            if len(self.ent_to_idx) > ENTITY_THRESHOLD:
                logger.warning('Only {} embeddings would be loaded in memory per batch...'.format(batch_size * 2))

            self.batch_size = batch_size
            self._initialize_parameters()

            dataset = tf.data.Dataset.from_generator(self._training_data_generator,
                                                     output_types=(tf.int32, tf.int32, tf.float32),
                                                     output_shapes=((None, 3), (None, 1), (None, self.internal_k)))

            dataset = dataset.repeat().prefetch(prefetch_batches)

            dataset_iterator = dataset.make_one_shot_iterator()
            # init tf graph/dataflow for training
            # init variables (model parameters to be learned - i.e. the embeddings)

            if self.loss.get_state('require_same_size_pos_neg'):
                batch_size = batch_size * self.eta

            loss = self._get_model_loss(dataset_iterator)

            train = self.optimizer.minimize(loss)

            # Entity embeddings normalization
            normalize_ent_emb_op = self.ent_emb.assign(tf.clip_by_norm(self.ent_emb, clip_norm=1, axes=1))

            self.early_stopping_params = early_stopping_params

            # early stopping
            if early_stopping:
                self._initialize_early_stopping()

            self.sess_train.run(tf.tables_initializer())
            self.sess_train.run(tf.global_variables_initializer())

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
                    if self.dealing_with_large_graphs:
                        loss_batch, unique_entities, _ = self.sess_train.run([loss, self.unique_entities, train],
                                                                             feed_dict=feed_dict)
                        self.ent_emb_cpu[np.squeeze(unique_entities), :] = \
                            self.sess_train.run(self.ent_emb)[:unique_entities.shape[0], :]
                    else:
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
                    if self._perform_early_stopping_test(epoch):
                        self._end_training()
                        return

            self._save_trained_params()
            self._end_training()
        except BaseException as e:
            self._end_training()
            raise e

    def set_filter_for_eval(self):
        """Configures to use filter
        """
        self.is_filtered = True

    def configure_evaluation_protocol(self, config=None):
        """Set the configuration for evaluation

        Parameters
        ----------
        config : dictionary
            Dictionary of parameters for evaluation configuration. Can contain following keys:

            - **corruption_entities**: List of entities to be used for corruptions.
              If ``all``, it uses all entities (default: ``all``)
            - **corrupt_side**: Specifies which side to corrupt. ``s``, ``o``, ``s+o`` (default)
            - **default_protocol**: Boolean flag to indicate whether to use default protocol for evaluation.
              This computes scores for corruptions of subjects and objects and ranks them separately.
              This could have been done by evaluating s and o separately and then
              ranking but it slows down the performance.
              Hence this mode is used where s+o corruptions are generated at once but ranked separately for speed up
              (default: False).

        """
        if config is None:
            config = {'corruption_entities': constants.DEFAULT_CORRUPTION_ENTITIES,
                      'corrupt_side': constants.DEFAULT_CORRUPT_SIDE_EVAL,
                      'default_protocol': constants.DEFAULT_PROTOCOL_EVAL}
        self.eval_config = config
        if self.eval_config['default_protocol']:
            self.eval_config['corrupt_side'] = 's+o'

    def _test_generator(self, mode):
        """Generates the test/validation data. If filter_triples are passed, then it returns the False Negatives
           that could be present in the generated corruptions.

           If we are dealing with large graphs, then along with the above, this method returns the idx of the
           entities present in the batch and their embeddings.
        """
        if self.is_filtered:
            test_generator = partial(self.eval_dataset_handle.get_next_batch_with_filter,
                                     batch_size=1, dataset_type=mode)
        else:
            test_generator = partial(self.eval_dataset_handle.get_next_eval_batch, batch_size=1, dataset_type=mode)

        batch_iterator = iter(test_generator())
        indices_obj = np.empty(shape=(0, 1), dtype=np.int32)
        indices_sub = np.empty(shape=(0, 1), dtype=np.int32)
        unique_ent = np.empty(shape=(0, 1), dtype=np.int32)
        entity_embeddings = np.empty(shape=(0, self.internal_k), dtype=np.float32)
        for i in range(self.eval_dataset_handle.get_size(mode)):
            if self.is_filtered:
                out, indices_obj, indices_sub = next(batch_iterator)
            else:
                out = next(batch_iterator)

            if self.dealing_with_large_graphs:
                # since we are dealing with only one triple (2 entities)
                unique_ent = np.unique(np.array([out[0, 0], out[0, 2]]))
                needed = (self.batch_size - unique_ent.shape[0])
                large_number = np.zeros((needed, self.ent_emb_cpu.shape[1]), dtype=np.float32) + np.nan
                entity_embeddings = np.concatenate((self.ent_emb_cpu[unique_ent, :], large_number), axis=0)
                unique_ent = unique_ent.reshape(-1, 1)

            yield out, indices_obj, indices_sub, entity_embeddings, unique_ent

    def _generate_corruptions_for_large_graphs(self):
        """Corruption generator for large graphs.
           It generates corruptions in batches and also yields the corresponding entity embeddings.
        """

        corruption_entities = self.eval_config.get('corruption_entities', constants.DEFAULT_CORRUPTION_ENTITIES)

        if corruption_entities == 'all':
            all_entities_np = np.arange(len(self.ent_to_idx))
            corruption_entities = all_entities_np
        elif isinstance(corruption_entities, np.ndarray):
            corruption_entities = corruption_entities
        else:
            msg = 'Invalid type for corruption entities.'
            logger.error(msg)
            raise ValueError(msg)

        entity_embeddings = np.empty(shape=(0, self.internal_k), dtype=np.float32)

        for i in range(self.corr_batches_count):
            all_ent = corruption_entities[i * self.batch_size:(i + 1) * self.batch_size]
            if self.dealing_with_large_graphs:
                needed = (self.batch_size - all_ent.shape[0])
                large_number = np.zeros((needed, self.ent_emb_cpu.shape[1]), dtype=np.float32) + np.nan
                entity_embeddings = np.concatenate((self.ent_emb_cpu[all_ent, :], large_number), axis=0)

            all_ent = all_ent.reshape(-1, 1)
            yield all_ent, entity_embeddings

    def _initialize_eval_graph(self, mode="test"):
        """Initialize the evaluation graph.

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

        use_default_protocol = self.eval_config.get('default_protocol', constants.DEFAULT_PROTOCOL_EVAL)
        corrupt_side = self.eval_config.get('corrupt_side', constants.DEFAULT_CORRUPT_SIDE_EVAL)
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

            corruption_entities = self.eval_config.get('corruption_entities', constants.DEFAULT_CORRUPTION_ENTITIES)

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

            corrupt_side = self.eval_config.get('corrupt_side', constants.DEFAULT_CORRUPT_SIDE_EVAL)
            # Generate corruptions
            self.out_corr = generate_corruptions_for_eval(self.X_test_tf,
                                                          self.corruption_entities_tf,
                                                          corrupt_side)

            # Compute scores for negatives
            e_s, e_p, e_o = self._lookup_embeddings(self.out_corr)
            self.scores_predict = self._fn(e_s, e_p, e_o)

            # Compute scores for positive
            e_s, e_p, e_o = self._lookup_embeddings(self.X_test_tf)
            self.score_positive = tf.squeeze(self._fn(e_s, e_p, e_o))

            use_default_protocol = self.eval_config.get('default_protocol', constants.DEFAULT_PROTOCOL_EVAL)

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
                                               constants.DEFAULT_CORRUPTION_ENTITIES), np.ndarray):
                corruption_entities = self.eval_config.get('corruption_entities',
                                                           constants.DEFAULT_CORRUPTION_ENTITIES).astype(np.int32)
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

    def end_evaluation(self):
        """End the evaluation and close the Tensorflow session.
        """

        if self.is_filtered and self.eval_dataset_handle is not None:
            self.eval_dataset_handle.cleanup()
            self.eval_dataset_handle = None

        if self.sess_predict is not None:
            self.sess_predict.close()

        self.sess_predict = None
        self.is_filtered = False

        self.eval_config = {}

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

        self.eval_dataset_handle = dataset_handle

        # build tf graph for predictions
        if self.sess_predict is None:
            tf.reset_default_graph()
            self.rnd = check_random_state(self.seed)
            tf.random.set_random_seed(self.seed)
            # load the parameters
            self._load_model_from_trained_params()
            # build the eval graph
            self._initialize_eval_graph()

            sess = tf.Session()
            sess.run(tf.tables_initializer())
            sess.run(tf.global_variables_initializer())
            self.sess_predict = sess

        ranks = []

        for i in tqdm(range(self.eval_dataset_handle.get_size('test'))):
            rank = self.sess_predict.run(self.rank)
            if self.eval_config.get('default_protocol', constants.DEFAULT_PROTOCOL_EVAL):
                ranks.append(list(rank))
            else:
                ranks.append(rank)

        return ranks

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
        # adapt the data with numpy adapter for internal use
        dataset_handle = NumpyDatasetAdapter()
        dataset_handle.use_mappings(self.rel_to_idx, self.ent_to_idx)
        dataset_handle.set_data(X, "test", mapped_status=from_idx)

        self.eval_dataset_handle = dataset_handle

        # build tf graph for predictions
        if self.sess_predict is None:
            tf.reset_default_graph()
            self.rnd = check_random_state(self.seed)
            tf.random.set_random_seed(self.seed)
            # load the parameters
            self._load_model_from_trained_params()
            # build the eval graph
            self._initialize_eval_graph()

            sess = tf.Session()
            sess.run(tf.tables_initializer())
            sess.run(tf.global_variables_initializer())
            self.sess_predict = sess

        scores = []

        for i in tqdm(range(self.eval_dataset_handle.get_size('test'))):
            score = self.sess_predict.run([self.score_positive])
            if self.eval_config.get('default_protocol', constants.DEFAULT_PROTOCOL_EVAL):
                scores.extend(list(score))
            else:
                scores.append(score)

        return scores

    def is_fitted_on(self, X):
        """ Determine heuristically if a model was fitted on the given triples.
        Parameters
        ----------
        X : ndarray, shape [n, 3]
             The triples to score.
        Returns
        -------
        bool : True if the number of unique entities and relations in X and
        the model match.
        """

        if not self.is_fitted:
            msg = 'Model has not been fitted.'
            logger.error(msg)
            raise RuntimeError(msg)

        unique_ent = np.unique(np.concatenate((X[:, 0], X[:, 2])))
        unique_rel = np.unique(X[:, 1])

        if not len(unique_ent) == len(self.ent_to_idx.keys()):
            return False
        elif not len(unique_rel) == len(self.rel_to_idx.keys()):
            return False

        return True
