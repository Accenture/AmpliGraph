Utils
=====
.. currentmodule:: ampligraph.utils

.. automodule:: ampligraph.utils


Saving/Restoring Models
-----------------------

Models can be saved and restored from disk. This is useful to avoid re-training a model.


.. autosummary::
    :toctree: generated
    :template: function.rst

    save_model
    restore_model


Visualization
-------------

Functions to visualize embeddings. 

.. autosummary::
    :toctree: generated
    :template: function.rst

    create_tensorboard_visualizations

Others
-------------

Function to convert a pandas DataFrame with headers into triples. 

.. autosummary::
    :toctree: generated
    :template: function.rst

    dataframe_to_triples