Advanced Topics
===============
.. currentmodule:: ampligraph.datasets

.. automodule:: ampligraph.datasets

.. _advanced_topics:

This section is meant as a brief introduction to AmpliGraph's data pipeline. Advanced users may use it as a starting
point to understand how to train their models on custom datasets which are extremely large and do not fit either on CPU
or GPU memory.

The first element of AmpliGraph's data pipeline is a data handler, that leverages the
:class:`GraphDataLoader` class to load large datasets. This data loader takes data from a source
and stores it in a certain backend. If, when initializing the :class:`GraphDataLoader`, we specify as
argument ``backend=DummyBackend`` (default), we opt for storing data in memory, i.e., we are not using any backend.
If, on the other hand, we set ``backend=SQLiteAdapter``, then we initialize a backend that relies on
`SQLite <https://www.sqlite.org/index.html>`_. In this case, data is persisted on disk and is later loaded in memory in
chunks, so to avoid overloading the RAM. This is the option to choose for handling massive datasets.

The instantiation of a backend is not by itself sufficient. Indeed, it is capital to specify how the chunks
we load in memory are defined. This is equivalent to tackle the problem of graph partitioning.
Partitioning a graph amounts to split its nodes into :math:`P` partitions sized to fit in memory.
When loading the data, partitions are created and singularly persisted on disk. Then, during training, single partitions
are loaded in memory and the model is trained on it. Once the model finishes operating on one partition, it unloads it
and loads the next one.

There are many possible strategies to partition a graph, but in AmpliGraph we recommend to use the default
option, the :class:`BucketGraphPartitioner` strategy, as its runtime performance are much better than
the others baselines.

For more details about the data pipeline components see the API below:

.. autosummary::
    :toctree:
    :template: class.rst

    GraphDataLoader
    BucketGraphPartitioner
