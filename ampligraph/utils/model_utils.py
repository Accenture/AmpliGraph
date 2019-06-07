# Copyright 2019 The AmpliGraph Authors. All Rights Reserved.
#
# This file is Licensed under the Apache License, Version 2.0.
# A copy of the Licence is available in LICENCE, or at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
import os
import pickle
import importlib
from time import gmtime, strftime
import glob
import logging

import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import numpy as np
import pandas as pd

"""This module contains utility functions for neural knowledge graph embedding models.
"""

DEFAULT_MODEL_NAMES = "{0}.model.pkl"

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def save_model(model, model_name_path=None):
    """ Save a trained model to disk.

        Examples
        --------
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
        >>> save_model(model, model_name_path = example_name)
        >>> print(y_pred_before)
        [-0.29721245, 0.07865551]

        Parameters
        ----------
        model: EmbeddingModel
            A trained neural knowledge graph embedding model,
            the model must be an instance of TransE,
            DistMult, ComplEx, or HolE.
        model_name_path: string
            The name of the model to be saved.
            If not specified, a default name model
            with current datetime is named
            and saved to the working directory

    """

    logger.debug('Saving model {}.'.format(model.__class__.__name__))

    obj = {
        'class_name': model.__class__.__name__,
        'hyperparams': model.all_params,
        'is_fitted': model.is_fitted,
        'ent_to_idx': model.ent_to_idx,
        'rel_to_idx': model.rel_to_idx,
    }

    model.get_embedding_model_params(obj)

    logger.debug('Saving hyperparams:{}\n\tis_fitted: \
                 {}'.format(model.all_params, model.is_fitted))

    if model_name_path is None:
        model_name_path = DEFAULT_MODEL_NAMES.format(strftime("%Y_%m_%d-%H_%M_%S", gmtime()))

    with open(model_name_path, 'wb') as fw:
        pickle.dump(obj, fw)
        # dump model tf


def restore_model(model_name_path=None):
    """ Restore a saved model from disk.

        See also :meth:`save_model`.

        Examples
        --------
        >>> from ampligraph.utils import restore_model
        >>> import numpy as np
        >>> example_name = 'helloworld.pkl'
        >>> restored_model = restore_model(model_name_path = example_name)
        >>> y_pred_after = restored_model.predict(np.array([['f', 'y', 'e'], ['b', 'y', 'd']]))
        >>> print(y_pred_after)
        [-0.29721245, 0.07865551]

        Parameters
        ----------
        model_name_path: string
            The name of saved model to be restored. If not specified,
            the library will try to find the default model in the working directory.

        Returns
        -------
        model: EmbeddingModel
            the neural knowledge graph embedding model restored from disk.

    """
    if model_name_path is None:
        logger.warn("There is no model name specified. \
                     We will try to lookup \
                     the latest default saved model...")
        default_models = glob.glob("*.model.pkl")
        if len(default_models) == 0:
            raise Exception("No default model found. Please specify \
                             model_name_path...")
        else:
            model_name_path = default_models[len(default_models) - 1]
            logger.info("Will will load the model: {0} in your \
                         current dir...".format(model_name_path))

    model = None
    logger.info('Will load model {}.'.format(model_name_path))
    restored_obj = None

    with open(model_name_path, 'rb') as fr:
        restored_obj = pickle.load(fr)

    if restored_obj:
        logger.debug('Restoring model...')
        module = importlib.import_module("ampligraph.latent_features.models")
        class_ = getattr(module, restored_obj['class_name'])
        model = class_(**restored_obj['hyperparams'])
        model.is_fitted = restored_obj['is_fitted']
        model.ent_to_idx = restored_obj['ent_to_idx']
        model.rel_to_idx = restored_obj['rel_to_idx']
        model.restore_model_params(restored_obj)
    else:
        logger.debug('No model found.')
    return model


def create_tensorboard_visualizations(model, loc, labels=None, write_metadata=True, export_tsv_embeddings=True):
    """ Export embeddings to Tensorboard.

        This function exports embeddings to disk in a format used by
        `TensorBoard <https://www.tensorflow.org/tensorboard>`_ and
        `TensorBoard Embedding Projector <https://projector.tensorflow.org>`_.
        The function exports:

        * A number of checkpoint and graph embedding files in the provided location that will allow
          you to visualize embeddings using Tensorboard. This is generally for use with a
          `local Tensorboard instance <https://www.tensorflow.org/tensorboard/r1/overview>`_.
        * a tab-separated file of embeddings ``embeddings_projector.tsv``. This is generally used to
          visualize embeddings by uploading to `TensorBoard Embedding Projector <https://projector.tensorflow.org>`_.
        * embeddings metadata (i.e. the embeddings labels from the original knowledge graph), saved to ``metadata.tsv``.
          Such file can be used in TensorBoard or uploaded to TensorBoard Embedding Projector.

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
            A TensorBoard guide is available at `this address <https://www.tensorflow.org/tensorboard/r1/overview>`_.

        .. Note ::
            Uploading ``embeddings_projector.tsv`` and ``metadata.tsv`` to
            `TensorBoard Embedding Projector <https://projector.tensorflow.org>`_ will give a result
            similar to the picture below:

            .. image:: ../img/embeddings_projector.png

        Examples
        --------
        >>> import numpy as np
        >>> from ampligraph.latent_features import TransE
        >>> from ampligraph.utils import create_tensorboard_visualizations
        >>>
        >>> X = np.array([['a', 'y', 'b'],
        >>>               ['b', 'y', 'a'],
        >>>               ['a', 'y', 'c'],
        >>>               ['c', 'y', 'a'],
        >>>               ['a', 'y', 'd'],
        >>>               ['c', 'y', 'd'],
        >>>               ['b', 'y', 'c'],
        >>>               ['f', 'y', 'e']])
        >>>
        >>> model = TransE(batches_count=1, seed=555, epochs=20, k=10, loss='pairwise',
        >>>                loss_params={'margin':5})
        >>> model.fit(X)
        >>>
        >>> create_tensorboard_visualizations(model, 'tensorboard_files')


        Parameters
        ----------
        model: EmbeddingModel
            A trained neural knowledge graph embedding model, the model must be an instance of TransE,
            DistMult, ComplEx, or HolE.
        loc: string
            Directory where the files are written.
        labels: pd.DataFrame
            Label(s) for each embedding point in the Tensorboard visualization.
            Default behaviour is to use the embeddings labels included in the model.
        export_tsv_embeddings: bool (Default: True
             If True, will generate a tab-separated file of embeddings at the given path. This is generally used to
             visualize embeddings by uploading to `TensorBoard Embedding Projector <https://projector.tensorflow.org>`_.
        write_metadata: bool (Default: True)
            If True will write a file named 'metadata.tsv' in the same directory as path.

    """

    # Create loc if it doesn't exist
    if not os.path.exists(loc):
        logger.debug('Creating Tensorboard visualization directory: %s' % loc)
        os.mkdir(loc)

    if not model.is_fitted:
        raise ValueError('Cannot write embeddings if model is not fitted.')

    # If no label data supplied, use model ent_to_idx keys as labels
    if labels is None:

        logger.info('Using model entity dictionary to create Tensorboard metadata.tsv')
        labels = list(model.ent_to_idx.keys())
    else:
        if len(labels) != len(model.ent_to_idx):
            raise ValueError('Label data rows must equal number of embeddings.')

    if write_metadata:
        logger.debug('Writing metadata.tsv to: %s' % loc)
        write_metadata_tsv(loc, labels)

    if export_tsv_embeddings:
        tsv_filename = "embeddings_projector.tsv"
        logger.info('Writing embeddings tsv to: %s' % os.path.join(loc, tsv_filename))
        np.savetxt(os.path.join(loc, tsv_filename), model.trained_model_params[0], delimiter='\t')

    checkpoint_path = os.path.join(loc, 'graph_embedding.ckpt')

    # Create embeddings Variable
    embedding_var = tf.Variable(model.trained_model_params[0], name='graph_embedding')

    with tf.Session() as sess:
        saver = tf.train.Saver([embedding_var])

        sess.run(embedding_var.initializer)

        saver.save(sess, checkpoint_path)

        config = projector.ProjectorConfig()

        # One can add multiple embeddings.
        embedding = config.embeddings.add()
        embedding.tensor_name = embedding_var.name

        # Link this tensor to its metadata file (e.g. labels).
        embedding.metadata_path = 'metadata.tsv'

        # Saves a config file that TensorBoard will read during startup.
        projector.visualize_embeddings(tf.summary.FileWriter(loc), config)


def write_metadata_tsv(loc, data):
    """ Write Tensorboard metadata.tsv file.

        Parameters
        ----------
        loc: string
            Directory where the file is written.
        data: list of strings, or pd.DataFrame
            Label(s) for each embedding point in the Tensorboard visualization.
            If data is a list of strings then no header will be written. If it is a pandas DataFrame with multiple
            columns then headers will be written.
    """

    # Write metadata.tsv
    metadata_path = os.path.join(loc, 'metadata.tsv')

    if isinstance(data, list):
        with open(metadata_path, 'w+', encoding='utf8') as metadata_file:
            for row in data:
                metadata_file.write('%s\n' % row)

    elif isinstance(data, pd.DataFrame):
        data.to_csv(metadata_path, sep='\t', index=False)
