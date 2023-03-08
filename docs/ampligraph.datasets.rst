Datasets
========
.. currentmodule:: ampligraph.datasets

.. automodule:: ampligraph.datasets


Loaders for Custom Knowledge Graphs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These are functions to load custom knowledge graphs from disk. They load the data from the specified files and store it
as a numpy array. These loaders are recommended when the datasets to load are small in size (approx 1M entities and
millions of triples), i.e., as long as they can be stored in memory. In case the dataset is too big to fit in memory,
use the :class:`GraphDataLoader` class instead (see the
:ref:`Advanced Topics <advanced_topics>` section for more).

.. autosummary::
    :toctree: generated
    :template: function.rst

    load_from_csv
    load_from_ntriples
    load_from_rdf
    
    
Benchmark Datasets Loaders
^^^^^^^^^^^^^^^^^^^^^^^^^^

| The following helper functions allow to load datasets used in graph representation learning literature as benchmarks.
| Among the various datasets, some include additional content to the usual triples. *WN11* and *FB13* provide true and negative labels for the triples in the validation and tests sets. CODEX-M contains also ground truths negative triples for test and validation sets (more information about the dataset in :cite:`safavi_codex_2020`).
| Finally, even though some of them are nowadays deprecated (*WN18* and *FB15k*), they are kept in the library as these were the first benchmarks to be used in literature.

.. role:: red

.. autosummary::
    :toctree: generated
    :template: function.rst

    load_fb15k_237
    load_wn18rr
    load_yago3_10
    load_wn11
    load_fb13
    load_codex
    load_fb15k
    load_wn18


**Datasets Summary**

========= ========= ======= ======= ============ ===========
 Dataset  Train     Valid   Test    Entities     Relations
========= ========= ======= ======= ============ ===========
FB15K-237 272,115   17,535  20,466  14,541        237
WN18RR    86,835    3,034   3,134   40,943        11
YAGO3-10  1,079,040 5,000   5,000   123,182       37
WN11      110,361   5,215   21,035  38,194        11
FB13      316,232   11,816  47,464  75,043        13
CODEX-M   185,584   10,310  10,311  17,050        51
FB15K     483,142   50,000  59,071  14,951        1,345
WN18      141,442   5,000   5,000   40,943        18
========= ========= ======= ======= ============ ===========

.. hint::
    It is recommended to set the ``AMPLIGRAPH_DATA_HOME`` environment variable::

        export AMPLIGRAPH_DATA_HOME=/YOUR/PATH/TO/datasets

    | When attempting to load a dataset, the module will first check if ``AMPLIGRAPH_DATA_HOME`` is set. If so, it will search this location for the required dataset. If not, the dataset will be downloaded and placed in this directory.
    | If ``AMPLIGRAPH_DATA_HOME`` is not set, the datasets will be saved in the ``~/ampligraph_datasets`` directory.


.. warning::
        | *FB15K-237*'s validation set contains 8 unseen entities over 9 triples. The test set has 29 unseen entities, distributed over 28 triples.
        | *WN18RR*'s validation set contains 198 unseen entities over 210 triples. The test set has 209 unseen entities, distributed over 210 triples.

.. _numeric-enriched-edges-loaders:

Benchmark Datasets With Numeric Values on Edges Loader
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These helper functions load benchmark datasets **with numeric values on edges** (the figure below shows a toy example).
Numeric values associated to edges of a knowledge graph have been used to represent uncertainty, edge importance, and
even out-of-band knowledge in a growing number of scenarios, ranging from genetic data to social networks.

.. image:: img/kg_eg.png
    :scale: 50 %
    :align: center

.. hint::
    To process a knowledge graphs with numeric values associated to edges, enable the
    FocusE layer :cite:`pai2021learning` when training a knowledge graph embedding model.

The functions will **automatically download** the datasets if they are not present in ``~/ampligraph_datasets`` or
at the location set in the ``AMPLIGRAPH_DATA_HOME``.

.. autosummary::
    :toctree: generated
    :template: function.rst

    load_onet20k
    load_ppi5k
    load_nl27k
    load_cn15k
    

**Datasets Summary**

========= ========= ======= ========= =========== =========
 Dataset  Train     Valid   Test      Entities    Relations
========= ========= ======= ========= =========== =========
O*NET20K  461,932   138      2,000    20,643      19
PPI5K     230,929   19,017   21,720   4,999       7
NL27K     149,100   12,274   14,026   27,221      405
CN15K     199,417   16,829   19,224   15,000      36
========= ========= ======= ========= =========== =========
