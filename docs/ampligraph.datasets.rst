Datasets
========
.. currentmodule:: ampligraph.datasets

.. automodule:: ampligraph.datasets

.. note::
    It is recommended to set the ``AMPLIGRAPH_DATA_HOME`` environment variable::

        export AMPLIGRAPH_DATA_HOME=/YOUR/PATH/TO/datasets

    When attempting to load a dataset, the module will first check if ``AMPLIGRAPH_DATA_HOME`` is set.
    If it is, it will search this location for the required dataset.
    If the dataset is not found it will be downloaded and placed in this directory.

    If ``AMPLIGRAPH_DATA_HOME`` has not been set the databases will be saved in the following directory::

        ~/ampligraph_datasets

Dataset-Specific Loaders
^^^^^^^^^^^^^^^^^^^^^^^^

Use these helpers functions to load datasets used in graph representation learning literature.
The functions will **automatically download** the datasets if they are not present in ``~/ampligraph_datasets`` or
at the location set in ``AMPLIGRAPH_DATA_HOME``.

.. role:: red

.. autosummary::
    :toctree: generated
    :template: function.rst

    load_wn18
    load_fb15k
    load_fb15k_237
    load_yago3_10
    load_wn18rr



**Datasets Summary**

========= ========= ======= ======= ============ ===========
 Dataset  Train     Valid   Test    Entities     Relations
========= ========= ======= ======= ============ ===========
FB15K-237 272,115   17,535  20,466  14,541        237
WN18RR    86,835    3,034   3,134   40,943        11
FB15K     483,142   50,000  59,071  14,951        1,345
WN18      141,442   5,000   5,000   40,943        18
YAGO3-10  1,079,040 5,000   5,000   123,182       37
========= ========= ======= ======= ============ ===========


.. warning:: FB15K-237's validation set contains 8 unseen entities over 9 triples. The test set has 29 unseen entities,
        distributed over 28 triples. WN18RR's validation set contains 198 unseen entities over 210 triples. The test set
        has 209 unseen entities, distributed over 210 triples.

Generic Loaders
^^^^^^^^^^^^^^^

Functions to load custom knowledge graphs from disk.

.. autosummary::
    :toctree: generated
    :template: function.rst

    load_from_csv
    load_from_ntriples
    load_from_rdf
