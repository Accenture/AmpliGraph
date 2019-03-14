Input
=====
.. currentmodule:: ampligraph.datasets

.. automodule:: ampligraph.datasets

Dataset-Specific Loaders
^^^^^^^^^^^^^^^^^^^^^^^^

Use these helpers functions to load datasets used in graph representation learning literature.

.. role:: red

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



**Dataset Summary**

========= ========= ======= ======= ============ ===========
 Dataset  Train     Valid   Test    Entities     Relations
========= ========= ======= ======= ============ ===========
FB15K-237 272,115   17,535  20,466  14,541        237
WN18RR    86,835    3,034   3,134   40,943        11
FB15K     483,142   50,000  59,071  14,951        1,345
WN18      141,442   5,000   5,000   40,943        18
WN11      110,361   5,215   21,035  38,588        11
FB13      316,232   11,816  47,464  75,043        13
YAGO3-10  1,079,040 5,000   5,000   123,182       37
========= ========= ======= ======= ============ ===========

These datasets are originated from: `FB15K-237 <https://www.microsoft.com/en-us/download/details.aspx?id=52312>`_, 
`WN18RR <https://github.com/TimDettmers/ConvE/blob/master/WN18RR.tar.gz>`_, 
`FB15K <https://www.hds.utc.fr/everest/doku.php?id=en:transe>`_, 
`WN18 <https://www.hds.utc.fr/everest/doku.php?id=en:transe>`_, 
`YAGO3-10 <https://github.com/TimDettmers/ConvE/blob/master/YAGO3-10.tar.gz>`_

.. warning:: FB15K-237 contains 8 unseen entities inside 9 triples in the validation set and 29 inside 28 triples in the test set.
             WN18RR contains 198 unseen entities inside 210 triples in the validation set and 209 inside 210 triples in the test set.

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
