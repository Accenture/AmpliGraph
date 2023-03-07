# Copyright 2019-2023 The AmpliGraph Authors. All Rights Reserved.
#
# This file is Licensed under the Apache License, Version 2.0.
# A copy of the Licence is available in LICENCE, or at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
import glob
import logging
import os
import pickle
import shutil
from time import gmtime, strftime

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorboard.plugins import projector

"""This module contains utility functions for neural knowledge graph embedding models.
"""

DEFAULT_MODEL_NAMES = "{0}.model.pkl"

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def save_model(model, model_name_path=None):
    """Save a trained model to disk.

    Example
    -------
    >>> import numpy as np
    >>> from ampligraph.latent_features import ComplEx
    >>> from ampligraph.utils import save_model
    >>> model = ComplEx(batches_count=2, seed=555, epochs=20, k=10)
    >>> X = np.array([['a', 'y', 'b'],
    >>>               ['b', 'y', 'a'],
    >>>               ['a', 'y', 'c'],
    >>>               ['c', 'y', 'a'],
    >>>               ['a', 'y', 'd'],
    >>>               ['c', 'y', 'd'],
    >>>               ['b', 'y', 'c'],
    >>>               ['f', 'y', 'e']])
    >>> model.fit(X)
    >>> y_pred_before = model.predict(np.array([['f', 'y', 'e'], ['b', 'y', 'd']]))
    >>> example_name = 'helloworld.pkl'
    >>> save_model(model, model_name_path=example_name)
    >>> print(y_pred_before)
    [-0.29721245, 0.07865551]

    Parameters
    ----------
    model: EmbeddingModel
        A trained neural knowledge graph embedding model.
        The model must be an instance of TransE, DistMult, ComplEx, or HolE.
    model_name_path: str
        The name of the model to be saved.
        If not specified, a default name with current datetime is selected and the model is saved
        to the working directory.

    """
    model.data_shape = tf.Variable(
        model.data_shape, trainable=False
    )  # Redefine the attribute for saving it
    if model_name_path is None:
        model_name_path = "{0}".format(strftime("%Y_%m_%d-%H_%M_%S", gmtime()))
    if os.path.exists(model_name_path):
        print(
            "The path {} already exists. This save operation will overwrite the model \
                at the specified path.".format(
                model_name_path
            )
        )
        shutil.rmtree(model_name_path)
    if model.is_backward:
        model = model.model
    tf.keras.models.save_model(model, model_name_path)
    model.save_metadata(filedir=model_name_path)


def restore_model(model_name_path=None):
    """Restore a trained model from disk.

    Parameters
    ----------
    model_name_path : str
        Name of the path to the model.
    """
    from ampligraph.compat.models import BACK_COMPAT_MODELS
    from ampligraph.latent_features import ScoringBasedEmbeddingModel
    from ampligraph.latent_features.layers.encoding import EmbeddingLookupLayer
    from ampligraph.latent_features.loss_functions import LOSS_REGISTRY
    from ampligraph.latent_features.optimizers import OptimizerWrapper

    if model_name_path is None:
        logger.warning(
            "There is no model name specified. \
                        We will try to lookup \
                        the latest default saved model..."
        )
        default_models = glob.glob("*.ampkl")
        if len(default_models) == 0:
            raise Exception(
                "No default model found. Please specify \
                             model_name_path..."
            )

    try:
        custom_objects = {
            "ScoringBasedEmbeddingModel": ScoringBasedEmbeddingModel,
            "OptimizerWrapper": OptimizerWrapper,
            "embedding_lookup_layer": EmbeddingLookupLayer,
        }
        custom_objects.update(LOSS_REGISTRY)

        model = tf.keras.models.load_model(
            model_name_path, custom_objects=custom_objects
        )
        model.load_metadata(filedir=model_name_path)
        if model.is_backward:
            model = BACK_COMPAT_MODELS.get(model.scoring_type)(model)
    except pickle.UnpicklingError as e:
        msg = "Error loading model {} : {}.".format(model_name_path, e)
        logger.debug(msg)
        raise Exception(msg)
    except (IOError, FileNotFoundError):
        msg = "No model found: {}.".format(model_name_path)
        logger.debug(msg)
        raise FileNotFoundError(msg)
    return model


def create_tensorboard_visualizations(
    model,
    loc,
    entities_subset="all",
    labels=None,
    write_metadata=True,
    export_tsv_embeddings=True,
):
    """Export embeddings to Tensorboard.

    This function exports embeddings to disk in a format used by
    `TensorBoard <https://www.tensorflow.org/tensorboard>`_ and
    `TensorBoard Embedding Projector <https://projector.tensorflow.org>`_.
    The function exports:

    * A number of checkpoint and graph embedding files in the provided location that will allow
      the visualization of the embeddings using Tensorboard. This is generally for use with a
      `local Tensorboard instance <https://www.tensorflow.org/tensorboard/r1/overview>`_.
    * A tab-separated file of embeddings named `embeddings_projector.tsv`. This is generally used to
      visualize embeddings by uploading to `TensorBoard Embedding Projector <https://projector.tensorflow.org>`_.
    * Embeddings metadata (i.e., the embedding labels from the original knowledge graph) saved to in a file named
        `metadata.tsv``. Such file can be used in TensorBoard or uploaded to TensorBoard Embedding Projector.

    The content of ``loc`` will look like: ::

        tensorboard_files/
            ├── checkpoint
            ├── embeddings_projector.tsv
            ├── graph_embedding.ckpt.data-00000-of-00001
            ├── graph_embedding.ckpt.index
            ├── graph_embedding.ckpt.meta
            ├── metadata.tsv
            └── projector_config.pbtxt

    .. Note ::
        A TensorBoard guide is available `here <https://www.tensorflow.org/tensorboard/r1/overview>`_.

    .. Note ::
        Uploading `embeddings_projector.tsv` and `metadata.tsv` to
        `TensorBoard Embedding Projector <https://projector.tensorflow.org>`_ will give a result
        similar to the picture below:

        .. image:: ../img/embeddings_projector.png

    Example
    -------
    >>> # create model and compile using user defined optimizer settings and user defined settings of an existing loss
    >>> from ampligraph.latent_features import ScoringBasedEmbeddingModel
    >>> from ampligraph.latent_features.loss_functions import SelfAdversarialLoss
    >>> import tensorflow as tf
    >>> optim = tf.optimizers.Adam(learning_rate=0.01)
    >>> loss = SelfAdversarialLoss({'margin': 0.1, 'alpha': 5, 'reduction': 'sum'})
    >>> model = ScoringBasedEmbeddingModel(eta=5,
    >>>                                    k=300,
    >>>                                    scoring_type='ComplEx',
    >>>                                    seed=0)
    >>> model.compile(optimizer=optim, loss=loss)
    >>> model.fit('./fb15k-237/train.txt',
    >>>           batch_size=10000,
    >>>           epochs=5)
    Epoch 1/5
    29/29 [==============================] - 2s 67ms/step - loss: 13101.9443
    Epoch 2/5
    29/29 [==============================] - 1s 20ms/step - loss: 11907.5771
    Epoch 3/5
    29/29 [==============================] - 1s 21ms/step - loss: 10890.3447
    Epoch 4/5
    29/29 [==============================] - 1s 20ms/step - loss: 9520.3994
    Epoch 5/5
    29/29 [==============================] - 1s 20ms/step - loss: 8314.7529
    >>> from ampligraph.utils import create_tensorboard_visualizations
    >>> create_tensorboard_visualizations(model,
                                          entities_subset='all',
                                          loc = './full_embeddings_vis')
    >>> # On terminal run: tensorboard --logdir='./full_embeddings_vis' --port=8891
    >>> # Open the browser and go to the following URL: http://127.0.0.1:8891/#projector


    Parameters
    ----------
    model: EmbeddingModel
        A trained neural knowledge graph embedding model, the model must be an instance of TransE,
        DistMult, ComplEx, or HolE.
    loc: str
        Directory where the files are written.
    entities_subset: list
        List of entities whose embeddings have to be visualized.
    labels: pd.DataFrame
        Label(s) for each embedding point in the Tensorboard visualization.
        Default behaviour is to use the embedding labels included in the model.
    export_tsv_embeddings: bool
         If `True` (default), will generate a tab-separated file of embeddings at the given path.
         This is generally used to visualize embeddings by uploading to
         `TensorBoard Embedding Projector <https://projector.tensorflow.org>`_.
    write_metadata: bool
        If `True` (default), will write a file named `'metadata.tsv'` in the same directory as path.

    """

    # Create loc if it doesn't exist
    if not os.path.exists(loc):
        logger.debug("Creating Tensorboard visualization directory: %s" % loc)
        os.mkdir(loc)

    if not model.is_fit():
        raise ValueError("Cannot write embeddings if model is not fitted.")

    if entities_subset != "all":
        assert isinstance(
            entities_subset, list
        ), "Please pass a list of entities of entities_subset!"

    if entities_subset == "all":
        entities_index = np.arange(model.get_count("e"))

        entities_label = list(
            model.get_indexes(entities_index, type_of="e", order="ind2raw")
        )
    else:
        entities_index = model.get_indexes(
            entities_subset, type_of="e", order="raw2ind"
        )
        entities_label = entities_subset

    if labels is not None:
        # Check if the lengths of the supplied labels is equal to the number of embeddings retrieved
        if len(labels) != len(entities_label):
            raise ValueError(
                "Label data rows must equal number of embeddings."
            )
    else:
        # If no label data supplied, use model ent_to_idx keys as labels
        labels = entities_label

    if write_metadata:
        logger.debug("Writing metadata.tsv to: %s" % loc)
        write_metadata_tsv(loc, labels)

    embeddings = model.get_embeddings(entities_label)

    if export_tsv_embeddings:
        tsv_filename = "embeddings_projector.tsv"
        logger.info(
            "Writing embeddings tsv to: %s" % os.path.join(loc, tsv_filename)
        )
        np.savetxt(os.path.join(loc, tsv_filename), embeddings, delimiter="\t")

    # Create a checkpoint with the embeddings only
    embeddings = tf.Variable(embeddings, name="graph_embeddings")
    checkpoint = tf.train.Checkpoint(KGE_embeddings=embeddings)
    checkpoint.save(os.path.join(loc, "graph_embeddings.ckpt"))

    # create a config to display the embeddings in the checkpoint
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = "KGE_embeddings/.ATTRIBUTES/VARIABLE_VALUE"
    embedding.metadata_path = "metadata.tsv"
    projector.visualize_embeddings(loc, config)


def write_metadata_tsv(loc, data):
    """Write Tensorboard `"metadata.tsv"` file.

    Parameters
    ----------
    loc: str
        Directory where the file is written.
    data: list of strings or pd.DataFrame
        Label(s) for each embedding point in the Tensorboard visualization.
        If ``data`` is a list of strings then no header will be written. If it is a `pandas DataFrame` with multiple
        columns, then the headers will be written.
    """

    # Write metadata.tsv
    metadata_path = os.path.join(loc, "metadata.tsv")

    if isinstance(data, list):
        with open(metadata_path, "w+", encoding="utf8") as metadata_file:
            for row in data:
                metadata_file.write("%s\n" % row)

    elif isinstance(data, pd.DataFrame):
        data.to_csv(metadata_path, sep="\t", index=False)

    else:
        raise ValueError("Labels must be passed as a list or a dataframe")


def dataframe_to_triples(X, schema):
    """Convert DataFrame into triple format.

    Parameters
    ----------
    X: pd.DataFrame with headers
    schema: list of tuples
        List of (subject, relation_name, object) tuples where subject and object are in the headers of the data frame.

    Example
    -------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from ampligraph.utils.model_utils import dataframe_to_triples
    >>>
    >>> X = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv')
    >>>
    >>> schema = [['species', 'has_sepal_length', 'sepal_length']]
    >>>
    >>> dataframe_to_triples(X, schema)[0]
    array(['setosa', 'has_sepal_length', '5.1'], dtype='<U16')
    """
    triples = []
    request_headers = set(np.delete(np.array(schema), 1, 1).flatten())
    diff = request_headers.difference(set(X.columns))
    if len(diff) > 0:
        raise Exception(
            "Subject/Object {} are not in data frame headers".format(diff)
        )
    for s, p, o in schema:
        triples.extend([[si, p, oi] for si, oi in zip(X[s], X[o])])
    return np.array(triples)


def preprocess_focusE_weights(data, weights, normalize=True):
    """Preprocessing of focusE weights.

    Extract weights from data, remove `NaNs`, average weights and normalize them
    if ``self.focusE_params['normalize_numeric_values']==True``.

    Parameters
    ----------
    data: array-like, shape (n,m)
        Array of shape (n,m) with :math:`m=4`. If ``weights=None``, data contains triples
        and weights (:math:`m>3`). If ``weights`` is passed, ``data`` only contains triples (:math:`m=3`).
    weights: array-like
        If not `None`, ``weights`` has shape (n, m-3), with m>0.
    normalize : bool
        Specify whether to normalize the weights into the [0,1] range (default: `True`).

    Returns
    -------
    processed_weights: np.array, shape (n, 1)
        An array of weights properly preprocessed and averaged into a unique vector if more than one vector of
        weights were given.
    """
    if weights.ndim == 1:
        weights = weights.reshape(-1, 1)
    logger.debug("focusE normalizing weights")
    unique_relations = np.unique(data[:, 1])
    for reln in unique_relations:
        for col_idx in range(weights.shape[1]):
            # here nans signify unknown numeric values
            suma = np.sum(pd.isna(weights[data[:, 1] == reln, col_idx]))
            if suma != weights[data[:, 1] == reln, col_idx].shape[0]:
                min_val = np.nanmin(
                    weights[data[:, 1] == reln, col_idx].astype(np.float32)
                )
                max_val = np.nanmax(
                    weights[data[:, 1] == reln, col_idx].astype(np.float32)
                )
                if min_val == max_val:
                    weights[data[:, 1] == reln, col_idx] = 1.0
                    continue
                # Normalization of the weights
                if normalize:
                    val = (
                        weights[data[:, 1] == reln, col_idx].astype(float)
                        - min_val
                    ) / (max_val - min_val)
                    weights[data[:, 1] == reln, col_idx] = val
            else:
                pass  # all the weights are nans
    weights = np.mean(weights, axis=1).reshape(-1, 1)
    return weights
