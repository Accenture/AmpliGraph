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

========= ======= ====== ====== ============ ===========
 Dataset  Train   Valid  Test   Dist. Ents   Dist. Rels  
========= ======= ====== ====== ============ ===========
WN18      141442  5000    5000   40943       18
FB15K     483142  50000  59071  14951        1345 
FB15K-237 272115  17535  20466  14541        237
WN11      110361  5215   21035  38588        11
FB13      316232  11816  47464  75043        13
YAGO3-10  1079040 5000   5000   123182       37
WN18RR    86835   3034   3134   40943        11
========= ======= ====== ====== ============ ===========

.. warning:: FB15K-237 contains (8, 29) unseen entities in (validation, test) sets. WN18RR contains (198,209) unseen entities in (validation, test) sets

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
