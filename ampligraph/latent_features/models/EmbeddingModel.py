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
                 large_graphs=False,
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
              :cite:`chen2015` by passing 'corrupt_side' as ['s','o'] to embedding_model_params.
              To use loss defined in :cite:`kadlecBK17` pass 'corrupt_side' as 'o' to embedding_model_params.

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

        large_graphs : bool
            Avoid loading entire dataset onto GPU when dealing with large graphs.

        verbose : bool
            Verbose mode.
        """
        if (loss == "bce") ^ (self.name == "ConvE"):
            raise ValueError('Invalid Model - Loss combination. '
                             'ConvE model can be used with BCE loss only and vice versa.')

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
                'initializer': initializer,
                'initializer_params': initializer_params,
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

        self.dealing_with_large_graphs = large_graphs

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

        self.tf_config = tf.ConfigProto(allow_soft_placement=True)
        self.tf_config.gpu_options.allow_growth = True
        self.sess_train = None
        self.trained_model_params = []
        self.is_fitted = False
        self.eval_config = {}
        self.eval_dataset_handle = None
        self.train_dataset_handle = None
        self.is_calibrated = False
        self.calibration_parameters = []

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

    def get_hyperparameter_dict(self):
        """Returns hyperparameters of the model.

        Returns
        -------
        hyperparam_dict : dict
            Dictionary of hyperparameters that were used for training.

        """
        return self.all_params

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
        output_dict['calibration_parameters'] = self.calibration_parameters

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
            self.calibration_parameters = in_dict['calibration_parameters']
        except KeyError:
            # For backward compatibility
            self.calibration_parameters = []

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
            # (We use tf.variable for future - to load and continue training)
            self.ent_emb = tf.Variable(self.trained_model_params[0], dtype=tf.float32)
        else:
            # Embeddings of all the corruptions entities will not fit on GPU.
            # During training we loaded batch_size*2 embeddings on GPU as only 2* batch_size unique
            # entities can be present in one batch.
            # During corruption generation in eval mode, one side(s/o) is fixed and only the other side varies.
            # Hence we use a batch size of 2 * training_batch_size for corruption generation i.e. those many
            # corruption embeddings would be loaded per batch on the GPU. In other words, those corruptions
            # would be processed as a batch.

            self.corr_batch_size = self.batch_size * 2

            # Load the entity embeddings on the cpu
            self.ent_emb_cpu = self.trained_model_params[0]
            # (We use tf.variable for future - to load and continue training)
            # create empty variable on GPU.
            # we initialize it with zeros because the actual embeddings will be loaded on the fly.
            self.ent_emb = tf.Variable(np.zeros((self.corr_batch_size, self.internal_k)), dtype=tf.float32)

        # (We use tf.variable for future - to load and continue training)
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
        x : tensor, shape [n, k]
            A tensor of k-dimensional embeddings

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
            self.rel_emb = tf.get_variable('rel_emb', shape=[len(self.rel_to_idx), self.internal_k],
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
            corruption_sides = self.embedding_model_params.get('corrupt_side', constants.DEFAULT_CORRUPT_SIDE_TRAIN)
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
        self.early_stopping_epoch = None

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
                if self.eval_config.get('corrupt_side', constants.DEFAULT_CORRUPT_SIDE_EVAL) == 's,o':
                    ranks.append(list(rank_triple))
                else:
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

                    self.early_stopping_epoch = epoch

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
        batch_iterator = iter(self.train_dataset_handle.get_next_batch(self.batches_count, "train"))
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
                - **'corrupt_side'**: Specifies which side to corrupt. 's', 'o', 's+o', 's,o' (default)

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

                logger.warning('Your graph has a large number of distinct entities. '
                               'Found {} distinct entities'.format(len(self.ent_to_idx)))

                logger.warning('Changing the variable initialization strategy.')
                logger.warning('Changing the strategy to use lazy loading of variables...')

                if early_stopping:
                    raise Exception('Early stopping not supported for large graphs')

                if not isinstance(self.optimizer, SGDOptimizer):
                    raise Exception("This mode works well only with SGD optimizer with decay (read docs for details).\
 Kindly change the optimizer and restart the experiment")

            if self.dealing_with_large_graphs:
                prefetch_batches = 0
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

            dataset_iterator = tf.data.make_one_shot_iterator(dataset)
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
            try:
                self.sess_train.run(self.set_training_true)
            except AttributeError:
                pass

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

                    try:
                        self.sess_train.run(self.set_training_false)
                    except AttributeError:
                        pass

                    if self._perform_early_stopping_test(epoch):
                        self._end_training()
                        return

                    try:
                        self.sess_train.run(self.set_training_true)
                    except AttributeError:
                        pass

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
            - **corrupt_side**: Specifies which side to corrupt. ``s``, ``o``, ``s+o``, ``s,o`` (default)
              In 's,o' mode subject and object corruptions are generated at once but ranked separately
              for speed up (default: False).

        """
        if config is None:
            config = {'corruption_entities': constants.DEFAULT_CORRUPTION_ENTITIES,
                      'corrupt_side': constants.DEFAULT_CORRUPT_SIDE_EVAL}
        self.eval_config = config

    def _test_generator(self, mode):
        """Generates the test/validation data. If filter_triples are passed, then it returns the False Negatives
           that could be present in the generated corruptions.

           If we are dealing with large graphs, then along with the above, this method returns the idx of the
           entities present in the batch and their embeddings.
        """
        test_generator = partial(self.eval_dataset_handle.get_next_batch,
                                 dataset_type=mode,
                                 use_filter=self.is_filtered)

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
                needed = (self.corr_batch_size - unique_ent.shape[0])
                large_number = np.zeros((needed, self.ent_emb_cpu.shape[1]), dtype=np.float32) + np.nan
                entity_embeddings = np.concatenate((self.ent_emb_cpu[unique_ent, :], large_number), axis=0)
                unique_ent = unique_ent.reshape(-1, 1)

            yield out, indices_obj, indices_sub, entity_embeddings, unique_ent

    def _generate_corruptions_for_large_graphs(self):
        """Corruption generator for large graph mode only.
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
            all_ent = corruption_entities[i * self.corr_batch_size:(i + 1) * self.corr_batch_size]
            needed = (self.corr_batch_size - all_ent.shape[0])
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
        dataset_iter = tf.data.make_one_shot_iterator(dataset)
        self.X_test_tf, indices_obj, indices_sub, entity_embeddings, unique_ent = dataset_iter.get_next()

        corrupt_side = self.eval_config.get('corrupt_side', constants.DEFAULT_CORRUPT_SIDE_EVAL)

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
                self.corr_batches_count = int(np.ceil(len(self.ent_to_idx) / (self.corr_batch_size)))

                # Corruption generator -
                # returns corruptions and their corresponding embeddings that need to be loaded on the GPU
                corruption_generator = tf.data.Dataset.from_generator(self._generate_corruptions_for_large_graphs,
                                                                      output_types=(tf.int32, tf.float32),
                                                                      output_shapes=((None, 1),
                                                                                     (None, self.internal_k)))

                corruption_generator = corruption_generator.repeat()
                corruption_generator = corruption_generator.prefetch(0)

                corruption_iter = tf.data.make_one_shot_iterator(corruption_generator)

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
                        if 's' in corrupt_side:
                            # compute and store the scores batch wise
                            scores_predict_s_c = self._fn(emb_corr, e_p, e_o)
                            scores_predict_s_corruptions_in = \
                                scores_predict_s_corruptions_in.scatter(tf.squeeze(corr_batch),
                                                                        tf.squeeze(scores_predict_s_c))

                        if 'o' in corrupt_side:
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

                if 's' in corrupt_side:
                    subj_corruption_scores = scores_predict_s_corr_out.stack()

                if 'o' in corrupt_side:
                    obj_corruption_scores = scores_predict_o_corr_out.stack()

                if corrupt_side == 's+o' or corrupt_side == 's,o':
                    self.scores_predict = tf.concat([obj_corruption_scores, subj_corruption_scores], axis=0)
                elif corrupt_side == 'o':
                    self.scores_predict = obj_corruption_scores
                else:
                    self.scores_predict = subj_corruption_scores

        else:

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

            if corrupt_side == 's,o':
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
            if corrupt_side == 's,o':
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
            if 'o' in corrupt_side:
                positives_among_obj_corruptions_ranked_higher = tf.reduce_sum(
                    tf.cast(scores_pos_obj >= self.score_positive, tf.int32))
            if 's' in corrupt_side:
                positives_among_sub_corruptions_ranked_higher = tf.reduce_sum(
                    tf.cast(scores_pos_sub >= self.score_positive, tf.int32))

        # compute the rank of the test triple and subtract the positives(from corruptions) that are ranked higher
        if corrupt_side == 's,o':
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
        ranks : ndarray, shape [n] or [n,2] depending on the value of corrupt_side.
                An array of ranks of test triples.
        """
        if not self.is_fitted:
            msg = 'Model has not been fitted.'
            logger.error(msg)
            raise RuntimeError(msg)

        self.eval_dataset_handle = dataset_handle

        # build tf graph for predictions
        tf.reset_default_graph()
        self.rnd = check_random_state(self.seed)
        tf.random.set_random_seed(self.seed)
        # load the parameters
        self._load_model_from_trained_params()
        # build the eval graph
        self._initialize_eval_graph()

        with tf.Session(config=self.tf_config) as sess:
            sess.run(tf.tables_initializer())
            sess.run(tf.global_variables_initializer())

            try:
                sess.run(self.set_training_false)
            except AttributeError:
                pass

            ranks = []

            for _ in tqdm(range(self.eval_dataset_handle.get_size('test')), disable=(not self.verbose)):
                rank = sess.run(self.rank)
                if self.eval_config.get('corrupt_side', constants.DEFAULT_CORRUPT_SIDE_EVAL) == 's,o':
                    ranks.append(list(rank))
                else:
                    ranks.append(rank)

            return ranks

    def predict(self, X, from_idx=False):
        """
        Predict the scores of triples using a trained embedding model.
        The function returns raw scores generated by the model.

        .. note::
            To obtain probability estimates, calibrate the model with :func:`~EmbeddingModel.calibrate`,
            then call :func:`~EmbeddingModel.predict_proba`.


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

        if type(X) is not np.ndarray:
            X = np.array(X)

        if not self.dealing_with_large_graphs:
            if not from_idx:
                X = to_idx(X, ent_to_idx=self.ent_to_idx, rel_to_idx=self.rel_to_idx)
            x_tf = tf.Variable(X, dtype=tf.int32, trainable=False)

            e_s, e_p, e_o = self._lookup_embeddings(x_tf)
            scores = self._fn(e_s, e_p, e_o)

            with tf.Session(config=self.tf_config) as sess:
                sess.run(tf.global_variables_initializer())
                return sess.run(scores)
        else:
            dataset_handle = NumpyDatasetAdapter()
            dataset_handle.use_mappings(self.rel_to_idx, self.ent_to_idx)
            dataset_handle.set_data(X, "test", mapped_status=from_idx)

            self.eval_dataset_handle = dataset_handle

            # build tf graph for predictions
            self.rnd = check_random_state(self.seed)
            tf.random.set_random_seed(self.seed)
            # load the parameters
            # build the eval graph
            self._initialize_eval_graph()

            with tf.Session(config=self.tf_config) as sess:
                sess.run(tf.tables_initializer())
                sess.run(tf.global_variables_initializer())

                try:
                    sess.run(self.set_training_false)
                except AttributeError:
                    pass

                scores = []

                for _ in tqdm(range(self.eval_dataset_handle.get_size('test')), disable=(not self.verbose)):
                    score = sess.run(self.score_positive)
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

    def _calibrate_with_corruptions(self, X_pos, batches_count):
        """
        Calibrates model with corruptions. The corruptions are hard-coded to be subject and object ('s,o')
        with all available entities.

        Parameters
        ----------
        X_pos : ndarray (shape [n, 3])
            Numpy array of positive triples.

        batches_count: int
            Number of batches to complete one epoch of the Platt scaling training.

        Returns
        -------
        scores_pos: tf.Tensor
            Tensor with positive scores.

        scores_neg: tf.Tensor
            Tensor with negative scores (generated by the corruptions).

        dataset_handle: NumpyDatasetAdapter
            Dataset handle (only used for clean-up).

        """
        dataset_handle = NumpyDatasetAdapter()
        dataset_handle.use_mappings(self.rel_to_idx, self.ent_to_idx)

        dataset_handle.set_data(X_pos, "pos")

        gen_fn = partial(dataset_handle.get_next_batch, batches_count=batches_count, dataset_type="pos")
        dataset = tf.data.Dataset.from_generator(gen_fn,
                                                 output_types=tf.int32,
                                                 output_shapes=(None, 3))
        dataset = dataset.repeat().prefetch(1)
        dataset_iter = tf.data.make_one_shot_iterator(dataset)

        x_pos_tf = dataset_iter.get_next()

        e_s, e_p, e_o = self._lookup_embeddings(x_pos_tf)
        scores_pos = self._fn(e_s, e_p, e_o)

        x_neg_tf = generate_corruptions_for_fit(x_pos_tf,
                                                entities_list=None,
                                                eta=1,
                                                corrupt_side='s,o',
                                                entities_size=len(self.ent_to_idx),
                                                rnd=self.seed)

        e_s_neg, e_p_neg, e_o_neg = self._lookup_embeddings(x_neg_tf)
        scores_neg = self._fn(e_s_neg, e_p_neg, e_o_neg)

        return scores_pos, scores_neg, dataset_handle

    def _calibrate_with_negatives(self, X_pos, X_neg):
        """
        Calibrates model with two datasets, one with positive triples and another with negative triples.

        Parameters
        ----------
        X_pos : ndarray (shape [n, 3])
            Numpy array of positive triples.

        X_neg : ndarray (shape [n, 3])
            Numpy array of negative triples.

        Returns
        -------
        scores_pos: tf.Tensor
            Tensor with positive scores.

        scores_neg: tf.Tensor
            Tensor with negative scores.

        """
        x_neg = to_idx(X_neg, ent_to_idx=self.ent_to_idx, rel_to_idx=self.rel_to_idx)
        x_neg_tf = tf.Variable(x_neg, dtype=tf.int32, trainable=False)

        x_pos = to_idx(X_pos, ent_to_idx=self.ent_to_idx, rel_to_idx=self.rel_to_idx)
        x_pos_tf = tf.Variable(x_pos, dtype=tf.int32, trainable=False)

        e_s, e_p, e_o = self._lookup_embeddings(x_neg_tf)
        scores_neg = self._fn(e_s, e_p, e_o)

        e_s, e_p, e_o = self._lookup_embeddings(x_pos_tf)
        scores_pos = self._fn(e_s, e_p, e_o)

        return scores_pos, scores_neg

    def calibrate(self, X_pos, X_neg=None, positive_base_rate=None, batches_count=100, epochs=50):
        """Calibrate predictions

        The method implements the heuristics described in :cite:`calibration`,
        using Platt scaling :cite:`platt1999probabilistic`.

        The calibrated predictions can be obtained with :meth:`predict_proba`
        after calibration is done.

        Ideally, calibration should be performed on a validation set that was not used to train the embeddings.

        There are two modes of operation, depending on the availability of negative triples:

        #. Both positive and negative triples are provided via ``X_pos`` and ``X_neg`` respectively. \
        The optimization is done using a second-order method (limited-memory BFGS), \
        therefore no hyperparameter needs to be specified.

        #. Only positive triples are provided, and the negative triples are generated by corruptions \
        just like it is done in training or evaluation. The optimization is done using a first-order method (ADAM), \
        therefore ``batches_count`` and ``epochs`` must be specified.


        Calibration is highly dependent on the base rate of positive triples.
        Therefore, for mode (2) of operation, the user is required to provide the ``positive_base_rate`` argument.
        For mode (1), that can be inferred automatically by the relative sizes of the positive and negative sets,
        but the user can override that by providing a value to ``positive_base_rate``.

        Defining the positive base rate is the biggest challenge when calibrating without negatives. That depends on
        the user choice of which triples will be evaluated during test time.
        Let's take WN11 as an example: it has around 50% positives triples on both the validation set and test set,
        so naturally the positive base rate is 50%. However, should the user resample it to have 75% positives
        and 25% negatives, its previous calibration will be degraded. The user must recalibrate the model now with a
        75% positive base rate. Therefore, this parameter depends on how the user handles the dataset and
        cannot be determined automatically or a priori.

        .. Note ::
            Incompatible with large graph mode (i.e. if ``self.dealing_with_large_graphs=True``).

        .. Note ::
            :cite:`calibration` `calibration experiments available here
            <https://github.com/Accenture/AmpliGraph/tree/paper/ICLR-20/experiments/ICLR-20>`_.


        Parameters
        ----------
        X_pos : ndarray (shape [n, 3])
            Numpy array of positive triples.
        X_neg : ndarray (shape [n, 3])
            Numpy array of negative triples.

            If `None`, the negative triples are generated via corruptions
            and the user must provide a positive base rate instead.
        positive_base_rate: float
            Base rate of positive statements.

            For example, if we assume there is a fifty-fifty chance of any query to be true, the base rate would be 50%.

            If ``X_neg`` is provided and this is `None`, the relative sizes of ``X_pos`` and ``X_neg`` will be used to
            determine the base rate. For example, if we have 50 positive triples and 200 negative triples,
            the positive base rate will be assumed to be 50/(50+200) = 1/5 = 0.2.

            This must be a value between 0 and 1.
        batches_count: int
            Number of batches to complete one epoch of the Platt scaling training.
            Only applies when ``X_neg`` is  `None`.
        epochs: int
            Number of epochs used to train the Platt scaling model.
            Only applies when ``X_neg`` is  `None`.

        Examples
        -------

        >>> import numpy as np
        >>> from sklearn.metrics import brier_score_loss, log_loss
        >>> from scipy.special import expit
        >>>
        >>> from ampligraph.datasets import load_wn11
        >>> from ampligraph.latent_features.models import TransE
        >>>
        >>> X = load_wn11()
        >>> X_valid_pos = X['valid'][X['valid_labels']]
        >>> X_valid_neg = X['valid'][~X['valid_labels']]
        >>>
        >>> model = TransE(batches_count=64, seed=0, epochs=500, k=100, eta=20,
        >>>                optimizer='adam', optimizer_params={'lr':0.0001},
        >>>                loss='pairwise', verbose=True)
        >>>
        >>> model.fit(X['train'])
        >>>
        >>> # Raw scores
        >>> scores = model.predict(X['test'])
        >>>
        >>> # Calibrate with positives and negatives
        >>> model.calibrate(X_valid_pos, X_valid_neg, positive_base_rate=None)
        >>> probas_pos_neg = model.predict_proba(X['test'])
        >>>
        >>> # Calibrate with just positives and base rate of 50%
        >>> model.calibrate(X_valid_pos, positive_base_rate=0.5)
        >>> probas_pos = model.predict_proba(X['test'])
        >>>
        >>> # Calibration evaluation with the Brier score loss (the smaller, the better)
        >>> print("Brier scores")
        >>> print("Raw scores:", brier_score_loss(X['test_labels'], expit(scores)))
        >>> print("Positive and negative calibration:", brier_score_loss(X['test_labels'], probas_pos_neg))
        >>> print("Positive only calibration:", brier_score_loss(X['test_labels'], probas_pos))
        Brier scores
        Raw scores: 0.4925058891371126
        Positive and negative calibration: 0.20434617882733366
        Positive only calibration: 0.22597599585144656

        """
        if not self.is_fitted:
            msg = 'Model has not been fitted.'
            logger.error(msg)
            raise RuntimeError(msg)

        if self.dealing_with_large_graphs:
            msg = "Calibration is incompatible with large graph mode."
            logger.error(msg)
            raise ValueError(msg)

        if positive_base_rate is not None and (positive_base_rate <= 0 or positive_base_rate >= 1):
            msg = "positive_base_rate must be a value between 0 and 1."
            logger.error(msg)
            raise ValueError(msg)

        dataset_handle = None

        try:
            tf.reset_default_graph()
            self.rnd = check_random_state(self.seed)
            tf.random.set_random_seed(self.seed)

            self._load_model_from_trained_params()

            if X_neg is not None:
                if positive_base_rate is None:
                    positive_base_rate = len(X_pos) / (len(X_pos) + len(X_neg))
                scores_pos, scores_neg = self._calibrate_with_negatives(X_pos, X_neg)
            else:
                if positive_base_rate is None:
                    msg = "When calibrating with randomly generated negative corruptions, " \
                          "`positive_base_rate` must be set to a value between 0 and 1."
                    logger.error(msg)
                    raise ValueError(msg)
                scores_pos, scores_neg, dataset_handle = self._calibrate_with_corruptions(X_pos, batches_count)

            n_pos = len(X_pos)
            n_neg = len(X_neg) if X_neg is not None else n_pos

            scores_tf = tf.concat([scores_pos, scores_neg], axis=0)
            labels = tf.concat([tf.cast(tf.fill(tf.shape(scores_pos), (n_pos + 1.0) / (n_pos + 2.0)), tf.float32),
                                tf.cast(tf.fill(tf.shape(scores_neg), 1 / (n_neg + 2.0)), tf.float32)],
                               axis=0)

            # Platt scaling model
            w = tf.get_variable('w', initializer=0.0, dtype=tf.float32)
            b = tf.get_variable('b', initializer=np.log((n_neg + 1.0) / (n_pos + 1.0)).astype(np.float32),
                                dtype=tf.float32)
            logits = -(w * tf.stop_gradient(scores_tf) + b)

            # Sample weights make sure the given positive_base_rate will be achieved irrespective of batch sizes
            weigths_pos = tf.size(scores_neg) / tf.size(scores_pos)
            weights_neg = (1.0 - positive_base_rate) / positive_base_rate
            weights = tf.concat([tf.cast(tf.fill(tf.shape(scores_pos), weigths_pos), tf.float32),
                                 tf.cast(tf.fill(tf.shape(scores_neg), weights_neg), tf.float32)], axis=0)

            loss = tf.losses.sigmoid_cross_entropy(labels, logits, weights=weights)

            if X_neg is not None:
                optimizer = tf.contrib.opt.ScipyOptimizerInterface(loss)
            else:
                optimizer = tf.train.AdamOptimizer()
                train = optimizer.minimize(loss)

            with tf.Session(config=self.tf_config) as sess:
                sess.run(tf.global_variables_initializer())

                if X_neg is not None:
                    optimizer.minimize(sess)
                else:
                    epoch_iterator_with_progress = tqdm(range(1, epochs + 1), disable=(not self.verbose), unit='epoch')
                    for _ in epoch_iterator_with_progress:
                        losses = []
                        for batch in range(batches_count):
                            loss_batch, _ = sess.run([loss, train])
                            losses.append(loss_batch)
                        if self.verbose:
                            msg = 'Calibration Loss: {:10f}'.format(sum(losses) / batches_count)
                            logger.debug(msg)
                            epoch_iterator_with_progress.set_description(msg)

                self.calibration_parameters = sess.run([w, b])
            self.is_calibrated = True
        finally:
            if dataset_handle is not None:
                dataset_handle.cleanup()

    def predict_proba(self, X):
        """
        Predicts probabilities using the Platt scaling model (after calibration).

        Model must be calibrated beforehand with the ``calibrate`` method.

        Parameters
        ----------
        X: ndarray (shape [n, 3])
            Numpy array of triples to be evaluated.

        Returns
        -------
        probas: ndarray (shape [n])
            Probability of each triple to be true according to the Platt scaling calibration.

        """
        if not self.is_calibrated:
            msg = "Model has not been calibrated. Please call `model.calibrate(...)` before predicting probabilities."
            logger.error(msg)
            raise RuntimeError(msg)

        tf.reset_default_graph()

        self._load_model_from_trained_params()

        w = tf.Variable(self.calibration_parameters[0], dtype=tf.float32, trainable=False)
        b = tf.Variable(self.calibration_parameters[1], dtype=tf.float32, trainable=False)

        x_idx = to_idx(X, ent_to_idx=self.ent_to_idx, rel_to_idx=self.rel_to_idx)
        x_tf = tf.Variable(x_idx, dtype=tf.int32, trainable=False)

        e_s, e_p, e_o = self._lookup_embeddings(x_tf)
        scores = self._fn(e_s, e_p, e_o)
        logits = -(w * scores + b)
        probas = tf.sigmoid(logits)

        with tf.Session(config=self.tf_config) as sess:
            sess.run(tf.global_variables_initializer())
            return sess.run(probas)
