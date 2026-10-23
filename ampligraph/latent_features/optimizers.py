# Copyright 2019-2026 The AmpliGraph Authors. All Rights Reserved.
#
# This file is Licensed under the Apache License, Version 2.0.
# A copy of the Licence is available in LICENCE, or at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
import abc
import inspect
import logging

import tensorflow as tf
import tf_keras

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# TF 2.16+ uses Keras 3 by default; tf.keras.optimizers.legacy is a broken stub there.
# tf_keras.optimizers.legacy provides the true Keras 2 optimizers with get_weights/set_weights
# that AmpliGraph needs for partitioned training persistence.
_legacy_optimizers = tf_keras.optimizers.legacy

# Accept Keras 3, tf_keras new-style, and tf_keras legacy optimizer instances from callers.
_OPTIMIZER_CLASSES = (
    tf.keras.optimizers.Optimizer,
    tf_keras.optimizers.Optimizer,
    tf_keras.optimizers.legacy.Optimizer,
)


def _optimizer_classes(name):
    """Return the (new, legacy) optimizer classes for a given optimizer name.

    Used for ``isinstance`` checks so that they work regardless of whether the wrapped
    optimizer is a new-style or legacy Keras optimizer.
    """
    classes = ()
    new_cls = getattr(tf.keras.optimizers, name, None)
    if new_cls is not None:
        classes += (new_cls,)
    if _legacy_optimizers is not None:
        legacy_cls = getattr(_legacy_optimizers, name, None)
        if legacy_cls is not None:
            classes += (legacy_cls,)
    return classes


# Pre-computed class tuples for the optimizers that need special handling.
_ADAM_CLASSES = _optimizer_classes("Adam")
_GPU_WORKAROUND_CLASSES = (
    _optimizer_classes("Adadelta")
    + _optimizer_classes("Adagrad")
    + _optimizer_classes("Ftrl")
)


def _to_legacy_optimizer(identifier, **hyperparams):
    """Resolve a string identifier to a *legacy* Keras optimizer instance when possible.

    Falls back to ``tf.keras.optimizers.get`` if the legacy module or the named optimizer is
    not available (e.g. on TF versions without the ``legacy`` namespace).
    """
    for attr in dir(_legacy_optimizers):
        if attr.lower() == identifier.lower():
            cls = getattr(_legacy_optimizers, attr)
            if isinstance(cls, type):
                return cls(**hyperparams)
    return tf_keras.optimizers.legacy.get(identifier)


def _ensure_legacy_optimizer(optimizer):
    """Rebuild a Keras 3 optimizer as its tf_keras (Keras 2) equivalent.

    AmpliGraph needs ``get_weights``/``set_weights`` for partitioned model persistence.
    Keras 3 optimizers dropped these, so any Keras 3 instance is converted to the
    matching tf_keras optimizer. tf_keras instances are returned unchanged.
    """
    if isinstance(optimizer, tf_keras.optimizers.legacy.Optimizer):
        return optimizer  # already a Keras 2 legacy optimizer with get_weights/set_weights

    legacy_cls = getattr(_legacy_optimizers, type(optimizer).__name__, None)
    if legacy_cls is None:
        return optimizer

    valid_kwargs = set(
        inspect.signature(legacy_cls.__init__).parameters
    )
    config = {
        k: v for k, v in optimizer.get_config().items() if k in valid_kwargs
    }
    try:
        return legacy_cls(**config)
    except (TypeError, ValueError):
        return optimizer


class OptimizerWrapper(abc.ABC):
    """Wrapper around tensorflow optimizer."""

    def __init__(self, optimizer=None):
        """Initialize the tensorflow Optimizer and wraps it so that it can be used with graph partitioning.

        Parameters
        ----------
        optimizer: str (name of optimizer) or optimizer instance.
            See `tf.keras.optimizers <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers>`.
        """
        self.optimizer = optimizer
        self.num_optimized_vars = 0
        # number of optimizer hpyerparams - adam has 2 if amsgrad is false
        self.number_hyperparams = 1
        self.is_partitioned_training = False

        # workaround for Adagrad/Adadelta/Ftrl optimizers to work on gpu
        self.gpu_workaround = False
        if isinstance(self.optimizer, _GPU_WORKAROUND_CLASSES):
            self.gpu_workaround = True

        if isinstance(self.optimizer, _ADAM_CLASSES):
            self.number_hyperparams = 2

    def apply_gradients(self, grads_and_vars):
        """Wrapper around apply_gradients.

        See `tf.keras.optimizers <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers>` for more details.
        """
        # tf_keras legacy optimizers expect tf.Variable (ResourceVariable) with _unique_id / device.
        # Keras 3 layers produce keras.Variable which wraps tf.Variable at ._value.
        # Unwrap so the legacy optimizer can slot correctly.
        unwrapped = [
            (g, v._value if isinstance(getattr(v, "_value", None), tf.Variable) else v)
            for g, v in grads_and_vars
        ]
        self.optimizer.apply_gradients(unwrapped)

    def set_partitioned_training(self, value=True):
        self.is_partitioned_training = value

    def minimize(self, loss, ent_emb, rel_emb, gradient_tape, other_vars=[]):
        """Minimizes the loss with respect to entity and relation embeddings and other trainable variables.

        Parameters
        ----------
        loss: tf.Tensor
            Model Loss.
        ent_emb: tf.Variable
            Entity embedding.
        rel_emb: tf.Variable
            Relation embedding.
        gradient tape: tf.GradientTape
            Gradient tape under which the loss computation was tracked.
        other_vars: list
            List of all the other trainable variables.
        """
        all_trainable_vars = [ent_emb, rel_emb]
        all_trainable_vars.extend(other_vars)
        # Total number of trainable variables in the graph
        self.num_optimized_vars = len(all_trainable_vars)

        if self.gpu_workaround:
            # workaround - see the issue:
            # https://github.com/tensorflow/tensorflow/issues/28090
            with gradient_tape:
                loss += 0.0000 * (
                    tf.reduce_sum(ent_emb) + tf.reduce_sum(rel_emb)
                )

        # Compute gradient of loss wrt trainable vars
        gradients = gradient_tape.gradient(loss, all_trainable_vars)
        # update the trainable params
        self.apply_gradients(zip(gradients, all_trainable_vars))

        # Compute the number of hyperparameters related to the optimizer
        # if self.is_partitioned_training and self.number_hyperparams == -1:
        #    optim_weights = self.optimizer.get_weights()
        #    self.number_hyperparams = 0
        #    for i in range(1, len(optim_weights), self.num_optimized_vars):
        #        self.number_hyperparams += 1

    def get_hyperparam_count(self):
        """Number of hyperparams of the optimizer being used.

        E.g., `adam` has `beta1` and `beta2`; if we use the `amsgrad` argument then it has also a third.
        """
        return self.number_hyperparams

    def get_entity_relation_hyperparams(self):
        """Get optimizer hyperparams related to entity and relation embeddings (for partitioned training).

        Returns
        -------
        ent_hyperparams: np.array
            Entity embedding related optimizer hyperparameters.
        rel_hyperparams: np.array
            Relation embedding related optimizer hyperparameters.
        """
        optim_weights = self.get_weights()
        ent_hyperparams = []
        rel_hyperparams = []
        for i in range(1, len(optim_weights), self.num_optimized_vars):
            ent_hyperparams.append(optim_weights[i])
            rel_hyperparams.append(optim_weights[i + 1])

        return ent_hyperparams, rel_hyperparams

    def set_entity_relation_hyperparams(
        self, ent_hyperparams, rel_hyperparams
    ):
        """Sets optimizer hyperparams related to entity and relation embeddings (for partitioned training).

        Parameters
        ----------
        ent_hyperparams: np.array
            Entity embedding related optimizer hyperparameters.
        rel_hyperparams: np.array
            Relation embedding related optimizer hyperparameters.
        """
        optim_weights = self.get_weights()
        for i, j in zip(
            range(1, len(optim_weights), self.num_optimized_vars),
            range(len(ent_hyperparams)),
        ):
            optim_weights[i] = ent_hyperparams[j]
            optim_weights[i + 1] = rel_hyperparams[j]
        self.set_weights(optim_weights)

    def get_weights(self):
        """Wrapper around get weights.

        See `tf.keras.optimizers <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers>` for more details.
        """
        return self.optimizer.get_weights()

    def set_weights(self, weights):
        """Wrapper around set weights.

        See `tf.keras.optimizers <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers>` for more details.
        """
        self.optimizer.set_weights(weights)

    def get_iterations(self):
        return self.optimizer.iterations.numpy()

    def get_config(self):
        return self.optimizer.get_config()

    @classmethod
    def from_config(cls, config):
        new_config = {}
        new_config["class_name"] = config["name"]

        del config["name"]
        new_config["config"] = config
        optimizer = tf.keras.optimizers.get(new_config)
        return optimizer


def get(identifier, hyperparams={}):
    """
    Get the optimizer specified by the identifier.

    Parameters
    ----------
    identifier: str or tf.optimizers.Optimizer instance
        Name of the optimizer to use (with default parameters) or instance of the class `tf.optimizers.Optimizer`.
    hyperparams: dict
        Dictionary containing the hyperparameters of the optimizer (learning rate...) if identifier is a string.
        Refer to tf.keras.optimizers args for the list of valid keys.

    Returns
    -------
    optimizer: OptimizerWrapper
        Instance of `tf.optimizers.Optimizer` wrapped around by `OptimizerWrapper` so that graph partitioning
        is supported.

    Example
    -------
    >>> from ampligraph.latent_features.optimizers import get as get_optimizer
    >>> optim = get_optimizer('adam', {'learning_rate': 5e-3})

    """
    if isinstance(identifier, _OPTIMIZER_CLASSES):
        return OptimizerWrapper(_ensure_legacy_optimizer(identifier))
    elif isinstance(identifier, OptimizerWrapper):
        return identifier
    elif isinstance(identifier, str):
        learning_rate = 0.001 if "learning_rate" not in hyperparams else hyperparams.pop("learning_rate")
        optimizer = _to_legacy_optimizer(identifier, **hyperparams)
        optimizer.learning_rate = learning_rate
        return OptimizerWrapper(optimizer)
    else:
        raise ValueError(
            "Could not interpret optimizer identifier: ", identifier
        )
