# Copyright 2019-2020 The AmpliGraph Authors. All Rights Reserved.
#
# This file is Licensed under the Apache License, Version 2.0.
# A copy of the Licence is available in LICENCE, or at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
import tensorflow as tf
import abc
import logging
import six
from tensorflow.python.eager import def_function
from tensorflow.python.keras import metrics as metrics_mod
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.ops import math_ops

LOSS_REGISTRY = {}

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Default margin used by pairwise and absolute margin loss
DEFAULT_MARGIN = 1

# default sampling temperature used by adversarial loss
DEFAULT_ALPHA_ADVERSARIAL = 0.5

# Default margin used by margin based adversarial loss
DEFAULT_MARGIN_ADVERSARIAL = 3

# Min score below which the values will be clipped before applying exponential
DEFAULT_CLIP_EXP_LOWER = -75.0

# Max score above which the values will be clipped before applying exponential
DEFAULT_CLIP_EXP_UPPER = 75.0

# Default label smoothing for ConvE
DEFAULT_LABEL_SMOOTHING = None

# Default label weighting for ConvE
DEFAULT_LABEL_WEIGHTING = False


def register_loss(name, external_params=None):
    if external_params is None:
        external_params = []

    def insert_in_registry(class_handle):
        LOSS_REGISTRY[name] = class_handle
        class_handle.name = name
        LOSS_REGISTRY[name].external_params = external_params
        return class_handle

    return insert_in_registry


def clip_before_exp(value):
    """Clip the value for stability of exponential
    """
    return tf.clip_by_value(value,
                            clip_value_min=DEFAULT_CLIP_EXP_LOWER,
                            clip_value_max=DEFAULT_CLIP_EXP_UPPER)


class Loss(abc.ABC):
    """Abstract class for loss function.
    """

    name = ""
    external_params = []
    class_params = {}

    def __init__(self, hyperparam_dict={}, verbose=False):
        """Initialize Loss.

        Parameters
        ----------
        hyperparam_dict : dict
            dictionary of hyperparams.
            (Keys are described in the hyperparameters section)
        """
        self._loss_parameters = {}
        self._dependencies = []
        self._user_losses = self.name
        
        self._loss_metric = metrics_mod.Mean(name='loss')  # Total loss.

        # perform check to see if all the required external hyperparams are passed
        try:
            self._init_hyperparams(hyperparam_dict)
            if verbose:
                logger.info('\n--------- Loss ---------')
                logger.info('Name : {}'.format(self.name))
                for key, value in self._loss_parameters.items():
                    logger.info('{} : {}'.format(key, value))
        except KeyError as e:
            msg = 'Some of the hyperparams for loss were not passed to the loss function.\n{}'.format(e)
            logger.error(msg)
            raise Exception(msg)

    @property
    def metrics(self):
        """Per-output loss metrics."""
        return [self._loss_metric] 

    def _init_hyperparams(self, hyperparam_dict):
        """Initializes the hyperparameters needed by the algorithm.

        Parameters
        ----------
        hyperparam_dict : dictionary
            Consists of key value pairs. The Loss will check the keys to get the corresponding params.
        """
        msg = 'This function is a placeholder in an abstract class.'
        logger.error(msg)
        raise NotImplementedError(msg)

    @tf.function(experimental_relax_shapes=True)
    def _apply_loss(self, scores_pos, scores_neg, eta):
        """Interface to external world.
        This function does the input checks, preprocesses input and finally applies loss function.

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
        msg = 'This function is a placeholder in an abstract class.'
        logger.error(msg)
        raise NotImplementedError(msg)
        
    def _broadcast_score_pos(self, scores_pos, eta):
        """Broadcast the score_pos to be as same size as the number of corruptions
        
        Parameters
        ----------
        scores_pos : tf.Tensor
            A tensor of scores assigned to positive statements.
        eta : tf.Tensor
            Number of corruptions
        Returns
        -------
        scores_pos : tf.Tensor
            Broadcasted score_pos
        """
        scores_pos = tf.reshape(tf.tile(scores_pos, [eta]), [tf.shape(scores_pos)[0] * eta])
        return scores_pos

        
    def __call__(self, scores_pos, scores_neg, eta, regularization_losses=None):
        """Interface to external world.
        This function does the input checks, preprocesses input and finally applies loss function.

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
        
        loss_values = []
        loss = self._apply_loss(scores_pos, scores_neg, eta)
        loss_values.append(loss)
        
        if regularization_losses:
            regularization_losses = losses_utils.cast_losses_to_common_dtype(regularization_losses)
            reg_loss = math_ops.add_n(regularization_losses)
            loss_values.append(reg_loss)
            
        loss_values = losses_utils.cast_losses_to_common_dtype(loss_values)
        total_loss = math_ops.add_n(loss_values)
        self._loss_metric.update_state(total_loss)
        return total_loss
        


@register_loss("pairwise", ['margin'])
class PairwiseLoss(Loss):
    r"""Pairwise, max-margin loss.

    Introduced in :cite:`bordes2013translating`.

    .. math::

        \mathcal{L}(\Theta) = \sum_{t^+ \in \mathcal{G}}\sum_{t^- \in \mathcal{C}}max(0, [\gamma + f_{model}(t^-;\Theta)
         - f_{model}(t^+;\Theta)])

    where :math:`\gamma` is the margin, :math:`\mathcal{G}` is the set of positives,
    :math:`\mathcal{C}` is the set of corruptions, :math:`f_{model}(t;\Theta)` is the model-specific scoring function.

    """

    def __init__(self, loss_params={}, verbose=False):
        """Initialize Loss.

        Parameters
        ----------
        loss_params : dict
            Dictionary of loss-specific hyperparams:

            - **'margin'**: (float). Margin to be used in pairwise loss computation (default: 1)

            Example: ``loss_params={'margin': 1}``
        """
        super().__init__(loss_params, verbose)

    def _init_hyperparams(self, hyperparam_dict={}):
        """Verifies and stores the hyperparameters needed by the algorithm.

        Parameters
        ----------
        hyperparam_dict : dictionary
            Consists of key value pairs. The Loss will check the keys to get the corresponding params

            - **margin** - Margin to be used in pairwise loss computation(default:1)
        """
        self._loss_parameters['margin'] = hyperparam_dict.get('margin', DEFAULT_MARGIN)

    @tf.function(experimental_relax_shapes=True)
    def _apply_loss(self, scores_pos, scores_neg, eta):
        """Apply the loss function.

        Parameters
        ----------
        scores_pos : tf.Tensor, shape [n, 1]
            A tensor of scores assigned to positive statements.
        scores_neg : tf.Tensor, shape [n, 1]
            A tensor of scores assigned to negative statements.

        Returns
        -------
        loss : tf.Tensor
            The loss value that must be minimized.

        """
        scores_pos = self._broadcast_score_pos(scores_pos, eta)
        margin = tf.constant(self._loss_parameters['margin'], dtype=tf.float32, name='margin')
        loss = tf.reduce_sum(tf.maximum(margin - scores_pos + scores_neg, 0))
        return loss


@register_loss("nll")
class NLLLoss(Loss):
    r"""Negative log-likelihood loss.

    As described in :cite:`trouillon2016complex`.

    .. math::

        \mathcal{L}(\Theta) = \sum_{t \in \mathcal{G} \cup \mathcal{C}}log(1 + exp(-y \, f_{model}(t;\Theta)))

    where :math:`y \in {-1, 1}` is the label of the statement, :math:`\mathcal{G}` is the set of positives,
    :math:`\mathcal{C}` is the set of corruptions, :math:`f_{model}(t;\Theta)` is the model-specific scoring function.

    """

    def __init__(self, loss_params={}, verbose=False):
        """Initialize Loss.

        Parameters
        ----------
        loss_params : dict
            Dictionary of hyperparams. No hyperparameters are required for this loss.
        """
        super().__init__(loss_params, verbose)

    def _init_hyperparams(self, hyperparam_dict={}):
        """Initializes the hyperparameters needed by the algorithm.

        Parameters
        ----------
        hyperparam_dict : dictionary
            Consists of key value pairs. The Loss will check the keys to get the corresponding params.
        """
        return

    @tf.function(experimental_relax_shapes=True)
    def _apply_loss(self, scores_pos, scores_neg, eta):
        """Apply the loss function.

        Parameters
        ----------
        scores_pos : tf.Tensor, shape [n, 1]
            A tensor of scores assigned to positive statements.
        scores_neg : tf.Tensor, shape [n, 1]
            A tensor of scores assigned to negative statements.

        Returns
        -------
        loss : tf.Tensor
            The loss value that must be minimized.

        """
        scores_neg = clip_before_exp(scores_neg)
        scores_pos = clip_before_exp(scores_pos)
        
        scores_pos = self._broadcast_score_pos(scores_pos, eta)
        
        scores = tf.concat([-scores_pos, scores_neg], 0)
        return tf.reduce_sum(tf.math.log(1 + tf.exp(scores)))


@register_loss("absolute_margin", ['margin'])
class AbsoluteMarginLoss(Loss):
    r"""Absolute margin , max-margin loss.

    Introduced in :cite:`Hamaguchi2017`.

    .. math::

        \mathcal{L}(\Theta) = \sum_{t^+ \in \mathcal{G}}\sum_{t^- \in \mathcal{C}} f_{model}(t^-;\Theta)
        - max(0, [\gamma - f_{model}(t^+;\Theta)])

    where :math:`\gamma` is the margin, :math:`\mathcal{G}` is the set of positives, :math:`\mathcal{C}` is the
    set of corruptions, :math:`f_{model}(t;\Theta)` is the model-specific scoring function.

    """

    def __init__(self, loss_params={}, verbose=False):
        """Initialize Loss

        Parameters
        ----------
        loss_params : dict
            Dictionary of loss-specific hyperparams:

            - **'margin'**: float. Margin to be used in pairwise loss computation (default:1)

            Example: ``loss_params={'margin': 1}``
        """
        super().__init__(loss_params, verbose)

    def _init_hyperparams(self, hyperparam_dict={}):
        """Initializes the hyperparameters needed by the algorithm.

        Parameters
        ----------
        hyperparam_dict : dict
           Consists of key value pairs. The Loss will check the keys to get the corresponding params.

           **margin** - Margin to be used in loss computation(default:1)

        Returns
        -------
        """
        self._loss_parameters['margin'] = hyperparam_dict.get('margin', DEFAULT_MARGIN)

    @tf.function(experimental_relax_shapes=True)
    def _apply_loss(self, scores_pos, scores_neg, eta):
        """Apply the loss function.

        Parameters
        ----------
        scores_pos : tf.Tensor, shape [n, 1]
           A tensor of scores assigned to positive statements.
        scores_neg : tf.Tensor, shape [n, 1]
           A tensor of scores assigned to negative statements.

        Returns
        -------
        loss : tf.Tensor
           The loss value that must be minimized.

        """
        margin = tf.constant(self._loss_parameters['margin'], dtype=tf.float32, name='margin')
        scores_pos = self._broadcast_score_pos(scores_pos, eta)
        loss = tf.reduce_sum(tf.maximum(margin + scores_neg, 0) - scores_pos)
        return loss


@register_loss("self_adversarial", ['margin', 'alpha'])#, {'require_same_size_pos_neg': False})
class SelfAdversarialLoss(Loss):
    r"""Self adversarial sampling loss.

    Introduced in :cite:`sun2018rotate`.

    .. math::

        \mathcal{L} = -log\, \sigma(\gamma + f_{model} (\mathbf{s},\mathbf{o}))
        - \sum_{i=1}^{n} p(h_{i}^{'}, r, t_{i}^{'} ) \ log \
        \sigma(-f_{model}(\mathbf{s}_{i}^{'},\mathbf{o}_{i}^{'}) - \gamma)

    where :math:`\mathbf{s}, \mathbf{o} \in \mathcal{R}^k` are the embeddings of the subject
    and object of a triple :math:`t=(s,r,o)`, :math:`\gamma` is the margin, :math:`\sigma` the sigmoid function,
    and :math:`p(s_{i}^{'}, r, o_{i}^{'} )` is the negatives sampling distribution which is defined as:

    .. math::

        p(s'_j, r, o'_j | \{(s_i, r_i, o_i)\}) = \frac{\exp \alpha \, f_{model}(\mathbf{s'_j}, \mathbf{o'_j})}
        {\sum_i \exp \alpha \, f_{model}(\mathbf{s'_i}, \mathbf{o'_i})}

    where :math:`\alpha` is the temperature of sampling, :math:`f_{model}` is the scoring function of
    the desired embeddings model.


    """

    def __init__(self, loss_params={}, verbose=False):
        """Initialize Loss

        Parameters
        ----------
        loss_params : dict
            Dictionary of loss-specific hyperparams:

            - **'margin'**: (float). Margin to be used for loss computation (default: 1)
            - **'alpha'** : (float). Temperature of sampling (default:0.5)

            Example: ``loss_params={'margin': 1, 'alpha': 0.5}``

        """
        super().__init__(loss_params, verbose)

    def _init_hyperparams(self, hyperparam_dict={}):
        """Initializes the hyperparameters needed by the algorithm.

        Parameters
        ----------
        hyperparam_dict : dictionary
            Consists of key value pairs. The Loss will check the keys to get the corresponding params

            - **margin** - Margin to be used in adversarial loss computation (default:3)

            - **alpha** - Temperature of sampling (default:0.5)
        """
        self._loss_parameters['margin'] = hyperparam_dict.get('margin', DEFAULT_MARGIN_ADVERSARIAL)
        self._loss_parameters['alpha'] = hyperparam_dict.get('alpha', DEFAULT_ALPHA_ADVERSARIAL)

    @tf.function(experimental_relax_shapes=True)
    def _apply_loss(self, scores_pos, scores_neg, eta):
        """Apply the loss function.

       Parameters
       ----------
       scores_pos : tf.Tensor, shape [n, 1]
           A tensor of scores assigned to positive statements.
       scores_neg : tf.Tensor, shape [n*negative_count, 1]
           A tensor of scores assigned to negative statements.

       Returns
       -------
       loss : tf.Tensor
           The loss value that must be minimized.

       """
        margin = tf.constant(self._loss_parameters['margin'], dtype=tf.float32, name='margin')
        alpha = tf.constant(self._loss_parameters['alpha'], dtype=tf.float32, name='alpha')

        # Compute p(neg_samples) based on eq 4
        scores_neg_reshaped = tf.reshape(scores_neg, [eta, tf.shape(scores_pos)[0]])
        p_neg = tf.nn.softmax(alpha * scores_neg_reshaped, axis=0)

        # Compute Loss based on eg 5
        loss = tf.reduce_sum(-tf.math.log_sigmoid(margin - tf.negative(scores_pos))) - tf.reduce_sum(
            tf.multiply(p_neg, tf.math.log_sigmoid(tf.negative(scores_neg_reshaped) - margin)))

        return loss


@register_loss("multiclass_nll", [])#, {'require_same_size_pos_neg': False})
class NLLMulticlass(Loss):
    r"""Multiclass NLL Loss.

    Introduced in :cite:`chen2015` where both the subject and objects are corrupted (to use it in this way pass
    corrupt_sides = ['s', 'o'] to embedding_model_params) .

    This loss was re-engineered in :cite:`kadlecBK17` where only the object was corrupted to get improved
    performance (to use it in this way pass corrupt_sides = 'o' to embedding_model_params).

    .. math::

        \mathcal{L(X)} = -\sum_{x_{e_1,e_2,r_k} \in X} log\,p(e_2|e_1,r_k)
         -\sum_{x_{e_1,e_2,r_k} \in X} log\,p(e_1|r_k, e_2)

    Examples
    --------
    >>> from ampligraph.latent_features import TransE
    >>> model = TransE(batches_count=1, seed=555, epochs=20, k=10,
    >>>                embedding_model_params={'corrupt_sides':['s', 'o']},
    >>>                loss='multiclass_nll', loss_params={})

    """
    def __init__(self, loss_params={}, verbose=False):
        """Initialize Loss

        Parameters
        ----------
        loss_params : dict
            Dictionary of loss-specific hyperparams:

        """
        super().__init__(loss_params, verbose)

    def _init_hyperparams(self, hyperparam_dict={}):
        """Verifies and stores the hyperparameters needed by the algorithm.

        Parameters
        ----------
        hyperparam_dict : dictionary
            Consists of key value pairs. The Loss will check the keys to get the corresponding params
        """
        pass

    @tf.function(experimental_relax_shapes=True)
    def _apply_loss(self, scores_pos, scores_neg, eta):
        """Apply the loss function.

       Parameters
       ----------
       scores_pos : tf.Tensor, shape [n, 1]
           A tensor of scores assigned to positive statements.
       scores_neg : tf.Tensor, shape [n*negative_count, 1]
           A tensor of scores assigned to negative statements.

       Returns
       -------
       loss : float
           The loss value that must be minimized.

       """
        # Fix for numerical instability of multiclass loss
        scores_pos = clip_before_exp(scores_pos)
        scores_neg = clip_before_exp(scores_neg)

        scores_neg_reshaped = tf.reshape(scores_neg, [eta, tf.shape(scores_pos)[0]])
        neg_exp = tf.exp(scores_neg_reshaped)
        pos_exp = tf.exp(scores_pos)
        softmax_score = pos_exp / (tf.reduce_sum(neg_exp, axis=0) + pos_exp)

        loss = -tf.reduce_sum(tf.math.log(softmax_score))
        return loss
    

class LossFunctionWrapper(Loss):
    """Wraps a loss function in the `Loss` class."""

    def __init__(self,
                 user_defined_loss,
                 name=None):
        super(LossFunctionWrapper, self).__init__()
        self._user_losses = user_defined_loss
        self.name = name
        
    def _init_hyperparams(self, hyperparam_dict={}):
        """Verifies and stores the hyperparameters needed by the algorithm.

        Parameters
        ----------
        hyperparam_dict : dictionary
            Consists of key value pairs. The Loss will check the keys to get the corresponding params
        """
        pass
    
    @tf.function(experimental_relax_shapes=True)
    def _apply_loss(self, scores_pos, scores_neg, eta):
        return self._user_losses(scores_pos, scores_neg, eta)

    
def get(identifier):
    if isinstance(identifier, Loss):
        return identifier
    elif isinstance(identifier, six.string_types):
        if identifier not in LOSS_REGISTRY.keys():
            raise ValueError('Could not interpret loss identifier:', identifier)
        return LOSS_REGISTRY.get(identifier)()
    elif callable(identifier):
        loss_name = identifier.__name__
        wrapped_callable = LossFunctionWrapper(identifier, loss_name)
        return wrapped_callable
    else:
        raise ValueError('Could not interpret loss identifier:', identifier)