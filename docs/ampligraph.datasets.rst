KG Loaders
===========
.. currentmodule:: ampligraph.datasets

.. automodule:: ampligraph.datasets


Dataset-Specific Loaders
^^^^^^^^^^^^^^^^^^^^^^^^

Use these helpers functions to load datasets used in graph representation learning literature.


.. autosummary::
    :toctree: generated
    :template: function.rst

    load_wn18
    load_fb15k
    load_fb15k_237
    load_wn11
    load_fb13
    load_yago3_10
    load_wn18rr

Generic Loaders
^^^^^^^^^^^^^^^

Functions to load custom knowledge graphs from disk.

.. note:: The environment variable ``AMPLIGRAPH_DATA_HOME`` must be set
    and input graphs must be stored at the path indicated.
    

.. autosummary::
    :toctree: generated
    :template: function.rst

    load_from_csv
    load_from_ntriples
    load_from_rdf
