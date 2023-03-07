# Copyright 2019-2023 The AmpliGraph Authors. All Rights Reserved.
#
# This file is Licensed under the Apache License, Version 2.0.
# A copy of the Licence is available in LICENCE, or at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
"""
Graph partitioning strategies.

This module contains several graph partitioning strategies both based on vertices split and edges split.

Attributes
----------
PARTITION_ALGO_REGISTRY : dict
    Dictionary containing the names of the strategies as key and reference to the strategy class as a value.

"""
import logging
import os
import shelve
import tempfile
from abc import ABC
from datetime import datetime

import numpy as np

from ampligraph.utils.profiling import timing_and_memory

from .graph_data_loader import GraphDataLoader

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
PARTITION_ALGO_REGISTRY = {}


def register_partitioning_strategy(name, manager):
    """Decorator responsible for registering partition in the partition registry.

    Parameters
    ----------
    name: str
         Name of the new partition strategy.
    manager: str
         Name of the partitioning manager that will handle this partitioning strategy during training.

    Example
    -------
    >>>@register_partitioning_strategy("NewStrategyName")
    >>>class NewPartitionStrategy(AbstractGraphPartitioner):
    >>>... pass
    """

    def insert_in_registry(class_handle):
        """Checks if partition already exists and if not registers it."""
        if name in PARTITION_ALGO_REGISTRY.keys():
            msg = "Partitioning Strategy with name {} already exists!".format(
                name
            )
            logger.error(msg)
            raise Exception(msg)

        PARTITION_ALGO_REGISTRY[name] = class_handle
        class_handle.name = name
        class_handle.manager = manager

        return class_handle

    return insert_in_registry


def get_number_of_partitions(n):
    """Calculates number of partitions for Bucket Partitioner.

    Parameters
    ----------
    n: int
         Number of buckets with vertices.

    Returns
    -------
    n_partitions: int
         Number of partitions.
    """
    return int(n * (n + 1) / 2)


class AbstractGraphPartitioner(ABC):
    """Meta class defining interface for graph partitioning algorithms."""

    def __init__(self, data, k=2, seed=None, root_dir=None, **kwargs):
        """Initialise the AbstractGraphPartitioner.

        Parameters
        ----------
        data: GraphDataLoader
             Input data provided as a GraphDataLoader.
        k: int
             Number of partitions or buckets to split data into.
        """
        self.files = []
        self.partitions = []
        self._data = data
        self._k = k
        if root_dir is None:
            self.root_dir = tempfile.gettempdir()
        else:
            self.root_dir = root_dir
        self._split(seed=seed, batch_size=data.batch_size, **kwargs)
        self.reload()

    def __iter__(self):
        """Function needed to be used as an iterator."""
        return self

    def reload(self):
        """Reload the partition."""
        self.generator = self.partitions_generator()

    def get_data(self):
        """Get the underlying data handler."""
        return self._data

    def partitions_generator(self):
        """Generates partitions.

        Yields
        ------
        next_partition : GraphDataLoader
             Next partition as a GraphDataLoader object.
        """
        for partition in self.partitions:
            partition.reload()
            yield partition

    def get_partitions_iterator(self):
        """Re-instantiate partitions generator.

        Returns
        -------
            Partitions generator
        """
        return self.partitions_generator()

    def get_partitions_list(self):
        """Returns handler for partitions list."""
        for partition in self.partitions:
            partition.reload()
        return self.partitions

    def __next__(self):
        """Function needed to be used as an iterator."""
        return next(self.generator)

    def _split(self, seed=None, **kwargs):
        """Split data into `k` equal size partitions.

        Parameters
        ----------
        seed: int
             Seed to be used for repeatability purposes, it is only used when certain randomization is required.

        Returns
        -------
        Partitions:
            Partitions in which the entities are divided.
        """
        pass

    def clean(self):
        """Remove the temporary files created for the partitions."""
        for partition in self.partitions:
            partition.clean()
        for f in self.files:
            if f.split(".")[-1] != "shf":
                os.remove(f)
            else:
                try:
                    os.remove(f + ".bak")
                    os.remove(f + ".dir")
                    os.remove(f + ".dat")
                except Exception:
                    if os.path.exists(f + ".db"):
                        os.remove(f + ".db")


@register_partitioning_strategy("Bucket", "BucketPartitionDataManager")
class BucketGraphPartitioner(AbstractGraphPartitioner):
    """Bucket-based partition strategy.

    This strategy first splits entities into :math:`k` buckets and creates:

    + `k` partitions where the `i`-th includes triples such that subject and object belong to the `i`-th partition.
    + :math:`\\frac{(k^2-k)}{2}` partitions indexed by :math:`(i,j)` with :math:`i,j=1,...,k`, :math:`i \\neq j` where
      the  :math:`(i,j)`-th partition contains triples such that the subject belongs to the :math:`i`-th partition
      and the object to the :math:`j`-th partition or viceversa.

    Example
    -------
    >>> from ampligraph.datasets import load_fb15k_237, GraphDataLoader, BucketGraphPartitioner
    >>> from ampligraph.datasets.sqlite_adapter import SQLiteAdapter
    >>> from ampligraph.latent_features import ScoringBasedEmbeddingModel
    >>> dataset = load_fb15k_237()
    >>> dataset_loader = GraphDataLoader(dataset['train'],
    >>>                                  backend=SQLiteAdapter, # Type of backend to use
    >>>                                  batch_size=1000,       # Batch size to use while iterating over the dataset
    >>>                                  dataset_type='train',  # Dataset type
    >>>                                  use_filter=False,      # Whether to use filter or not
    >>>                                  use_indexer=True)      # indicates that the data needs to be mapped to index
    >>> partitioner = BucketGraphPartitioner(dataset_loader, k=2)
    >>> # create and compile a model as usual
    >>> partitioned_model = ScoringBasedEmbeddingModel(eta=2, k=50, scoring_type='DistMult')
    >>> partitioned_model.compile(optimizer='adam', loss='multiclass_nll')
    >>> partitioned_model.fit(partitioner,       # The partitioner object generate data for the model during training
    >>>                       epochs=10)         # Number of epochs

    Example
    -------
    >>> import numpy as np
    >>> from ampligraph.datasets import GraphDataLoader, BucketGraphPartitioner
    >>> d = np.array([[1,1,2], [1,1,3],[1,1,4],[5,1,3],[5,1,2],[6,1,3],[6,1,2],[6,1,4],[6,1,7]])
    >>> data = GraphDataLoader(d, batch_size=1, dataset_type="test")
    >>> partitioner = BucketGraphPartitioner(data, k=2)
    >>> for i, partition in enumerate(partitioner):
    >>>    print("partition ", i)
    >>>    for batch in partition:
    >>>        print(batch)
    partition  0
    [['0,0,1']]
    [['0,0,2']]
    [['0,0,3']]
    partition  1
    [['4,0,1']]
    [['4,0,2']]
    [['5,0,1']]
    [['5,0,2']]
    [['5,0,3']]
    partition  2
    [['5,0,6']]

    """

    def __init__(self, data, k=2, **kwargs):
        """Initialise the BucketGraphPartitioner.

        Parameters
        ----------
        data: GraphDataLoader
            Input data as a GraphDataLoader.
        k: int
            Number of buckets to split entities (i.e., vertices) into.

        """

        self.partitions = []
        super().__init__(data, k, **kwargs)

    def create_single_partition(
        self, ind1, ind2, timestamp, partition_nb, batch_size=1
    ):
        """Creates partition based on the two given indices of buckets.

        It appends created partition to the list of partitions (self.partitions).

        Parameters
        ----------
        ind1: int
             Index of the first bucket needed to create partition.
        ind2: int
             Index of the second bucket needed to create partition.
        timestamp: str
             Date and time string that the files are created with (shelves).
        partition_nb: int
             Assigned number of partitions.

        """
        # logger.debug("------------------------------------------------")
        # logger.debug("Creating partition nb: {}".format(partition_nb))

        fname = "bucket_{}_{}.shf".format(ind1, timestamp)
        with shelve.open(
            os.path.join(self.root_dir, fname), writeback=True
        ) as bucket_partition_1:
            indexes_1 = bucket_partition_1["indexes"]
        fname = "bucket_{}_{}.shf".format(ind2, timestamp)
        with shelve.open(
            os.path.join(self.root_dir, fname), writeback=True
        ) as bucket_partition_2:
            indexes_2 = bucket_partition_2["indexes"]

        # logger.debug("indexes 1: {}".format(ind1, indexes_1))
        # logger.debug("indexes 2: {}".format(ind2, indexes_2))

        triples_1_2 = np.array(
            self._data.get_triples(subjects=indexes_1, objects=indexes_2)
        )[:, :3]
        triples_2_1 = np.array(
            self._data.get_triples(subjects=indexes_2, objects=indexes_1)
        )[:, :3]

        logger.debug("triples 1-2: {}".format(triples_1_2))
        logger.debug("triples 2-1: {}".format(triples_2_1))
        triples = np.vstack([triples_1_2, triples_2_1]).astype(np.int32)
        # logger.debug(triples)
        if triples.size != 0:
            triples = np.unique(triples, axis=0)
            # logger.debug("unique triples: {}".format(triples))
            fname = "partition_{}_{}.csv".format(partition_nb, timestamp)
            fname = os.path.join(self.root_dir, fname)
            self.files.append(fname)
            np.savetxt(fname, triples, delimiter="\t", fmt="%d")
            # special case of GraphDataLoader to create partition datasets:
            # with remapped indexes (0, size_of_partition),
            # persisted, with partition number to look up remappings
            partition_loader = GraphDataLoader(
                fname,
                use_indexer=False,
                batch_size=batch_size,
                remap=True,
                parent=self._data,
                name="partition_{}_buckets_{}-{}".format(
                    partition_nb, ind1, ind2
                ),
            )
            self.partitions.append(partition_loader)
            return 0  # status everything went ok
        else:
            return 1  # status not ok, no partition created

    @timing_and_memory
    def _split(self, seed=None, verbose=False, batch_size=1, **kwargs):
        """Split data into `self.k` buckets based on unique entities and assign
        accordingly triples to `k` partitions and intermediate partitions.

        """
        timestamp = datetime.now().strftime("%d-%m-%Y_%I-%M-%S_%p")
        self.ents_size = self._data.backend.mapper.get_entities_count()
        logger.debug(self.ents_size)
        self.bucket_size = int(np.ceil(self.ents_size / self._k))
        self.buckets_generator = (
            self._data.backend.mapper.get_entities_in_batches(
                batch_size=self.bucket_size
            )
        )

        for i, bucket in enumerate(self.buckets_generator):
            # dump entities in partition shelve/file
            fname = "bucket_{}_{}.shf".format(i, timestamp)
            fname = os.path.join(self.root_dir, fname)
            self.files.append(fname)
            with shelve.open(fname, writeback=True) as bucket_partition:
                bucket_partition["indexes"] = bucket
            # logger.debug(bucket)

        partition_nb = 0
        # ensure that the "same" bucket partitions are generated first
        for i in range(self._k):
            # condition that excludes duplicated partitions
            # from k x k possibilities, partition 0-1 and 1-0 is the same - not
            # needed
            status_not_ok = self.create_single_partition(
                i, i, timestamp, partition_nb, batch_size=batch_size
            )
            if status_not_ok:
                continue
            partition_nb += 1

        # Now generate across bucket partitions
        for i in range(self._k):
            for j in range(self._k):
                if j > i:
                    # condition that excludes duplicated partitions
                    # from k x k possibilities, partition 0-1 and 1-0 are the
                    # same - not needed
                    status_not_ok = self.create_single_partition(
                        i, j, timestamp, partition_nb, batch_size=batch_size
                    )
                    if status_not_ok:
                        continue
                    partition_nb += 1


@register_partitioning_strategy(
    "RandomVertices", "GeneralPartitionDataManager"
)
class RandomVerticesGraphPartitioner(AbstractGraphPartitioner):
    """Partitioning strategy that splits vertices into equal sized buckets of random entities from the graph.

    Example
    -------
    >>> from ampligraph.datasets imoprt load_fb15k_237
    >>> from ampligraph.datasets import GraphDataLoader
    >>> from ampligraph.datasets.sqlite_adapter import SQLiteAdapter
    >>> from ampligraph.datasets.graph_partitioner import RandomVerticesGraphPartitioner
    >>> from ampligraph.latent_features import ScoringBasedEmbeddingModel
    >>> dataset = load_fb15k_237()
    >>> dataset_loader = GraphDataLoader(dataset['train'],
    >>>                                  backend=SQLiteAdapter, # type of backend to use
    >>>                                  batch_size=2,          # batch size to use while iterating over this dataset
    >>>                                  dataset_type='train',  # dataset type
    >>>                                  use_filter=False,      # whether to use filter or not
    >>>                                  use_indexer=True)      # indicates that the data needs to be mapped to index
    >>> partitioner = RandomVerticesGraphPartitioner(dataset_loader, k=2)
    >>> # create and compile a model as usual
    >>> partitioned_model = ScoringBasedEmbeddingModel(eta=2,
    >>>                                                k=50,
    >>>                                                scoring_type='DistMult')
    >>>
    >>> partitioned_model.compile(optimizer='adam', loss='multiclass_nll')
    >>>
    >>> partitioned_model.fit(partitioner,            # pass the partitioner object as input to the fit function this will generate data for the model during training
    >>>                       epochs=10)              # number of epochs

    Example
    -------
    >>> import numpy as np
    >>> from ampligraph.datasets.graph_partitioner import GraphDataLoader, RandomVerticesGraphPartitioner
    >>> d = np.array([[1,1,2], [1,1,3],[1,1,4],[5,1,3],[5,1,2],[6,1,3],[6,1,2],[6,1,4],[6,1,7]])
    >>> data = GraphDataLoader(d, batch_size=1, dataset_type="test")
    >>> partitioner = RandomVerticesGraphPartitioner(data, k=2)
    >>> for i, partition in enumerate(partitioner):
    >>>    print("partition ", i)
    >>>    for batch in partition:
    >>>        print(batch)

    """

    def __init__(self, data, k=2, seed=None, **kwargs):
        """Initialise the RandomVerticesGraphPartitioner.

        Parameters
        ----------
        data: GraphDataLoader
            Input data provided as a GraphDataLoader.
        k: int
            Number of buckets to split entities (i.e., vertices) into.
        seed: int
            Seed to be used during partitioning.

        """
        self._data = data
        self._k = k
        self.partitions = []
        super().__init__(data, k, **kwargs)

    @timing_and_memory
    def _split(self, seed=None, batch_size=1, **kwargs):
        """Split data into `k` equal size partitions by randomly drawing subset of vertices
        of partition size and retrieving triples associated with these vertices.

        """
        timestamp = datetime.now().strftime("%d-%m-%Y_%I-%M-%S_%p")
        self.ents_size = self._data.backend.mapper.get_entities_count()
        # logger.debug(self.ents_size)
        # logger.debug(backend.mapper.max_ents_index)
        self.partition_size = int(np.ceil(self.ents_size / self._k))
        # logger.debug(self.partition_size)
        self.buckets_generator = (
            self._data.backend.mapper.get_entities_in_batches(
                batch_size=self.partition_size, random=True, seed=seed
            )
        )

        for partition_nb, partition in enumerate(self.buckets_generator):
            # logger.debug(partition)
            tmp = np.array(self._data.backend._get_triples(entities=partition))
            # tmp_subj = np.array(self._data.backend._get_triples(subjects=partition))
            # tmp_obj = np.array(self._data.backend._get_triples(objects=partition))
            # tmp = np.unique(np.concatenate([tmp_subj, tmp_obj], axis=0), axis=0)

            if tmp.size != 0:
                triples = tmp[:, :3].astype(np.int32)
                # logger.debug("unique triples: {}".format(triples))
                fname = "partition_{}_{}.csv".format(partition_nb, timestamp)
                fname = os.path.join(self.root_dir, fname)
                self.files.append(fname)
                np.savetxt(fname, triples, delimiter="\t", fmt="%d")
                # special case of GraphDataLoader to create partition datasets:
                # with remapped indexes (0, size_of_partition),
                # persisted, with partition number to look up remappings
                partition_loader = GraphDataLoader(
                    fname,
                    use_indexer=False,
                    batch_size=batch_size,
                    remap=True,
                    parent=self._data,
                    name="partition_{}".format(partition_nb),
                )
                self.partitions.append(partition_loader)
            else:
                logger.debug("Partition has no triples, skipping!")


class EdgeBasedGraphPartitioner(AbstractGraphPartitioner):
    """Template for edge-based partitioning strategy that splits edges
    into partitions.

    To be inherited to create different edge-based strategies.

    Example
    -------
    >>> from ampligraph.datasets imoprt load_fb15k_237
    >>> from ampligraph.datasets import GraphDataLoader
    >>> from ampligraph.datasets.sqlite_adapter import SQLiteAdapter
    >>> from ampligraph.datasets.graph_partitioner import EdgeBasedGraphPartitioner
    >>> from ampligraph.latent_features import ScoringBasedEmbeddingModel
    >>> dataset = load_fb15k_237()
    >>> dataset_loader = GraphDataLoader(dataset['train'],
    >>>                                  backend=SQLiteAdapter, # type of backend to use
    >>>                                  batch_size=2,          # batch size to use while iterating over this dataset
    >>>                                  dataset_type='train',  # dataset type
    >>>                                  use_filter=False,      # Whether to use filter or not
    >>>                                  use_indexer=True)      # indicates that the data needs to be mapped to index
    >>> partitioner = EdgeBasedGraphPartitioner(dataset_loader, k=2)
    >>> # create and compile a model as usual
    >>> partitioned_model = ScoringBasedEmbeddingModel(eta=2,
    >>>                                                k=50,
    >>>                                                scoring_type='DistMult')
    >>>
    >>> partitioned_model.compile(optimizer='adam', loss='multiclass_nll')
    >>>
    >>> partitioned_model.fit(partitioner,            # pass the partitioner object as input to the fit function this will generate data for the model during training
    >>>                       epochs=10)              # number of epochs

    """

    def __init__(self, data, k=2, random=False, index_by="", **kwargs):
        """Initialise the EdgeBasedGraphPartitioner.

        Parameters
        ----------
        data: GraphDataLoader
            Input data as a GraphDataLoader.
        k: int
            Number of buckets to split entities (i.e., vertices) into.
        random: bool
            Whether to draw edges/triples in random order.
        index_by: str
            Which index to use when returning triples (`"s"`, `"o"`, `"so"`, `"os"`).

        """

        self.partitions = []
        self._data = data
        self._k = k
        super().__init__(data, k=k, random=random, index_by=index_by, **kwargs)

    def get_data(self):
        """Get the underlying data handler."""
        return self._data

    @timing_and_memory
    def _split(
        self, seed=None, batch_size=1, random=False, index_by="", **kwargs
    ):
        """Split data into `k` equal size partitions by randomly drawing subset of edges from dataset.

        Returns
        -------
        partitions
             Parts of equal size containing triples
        """
        timestamp = datetime.now().strftime("%d-%m-%Y_%I-%M-%S_%p")
        self.size = self._data.backend.get_data_size()

        self.partition_size = int(np.ceil(self.size / self._k))
        logger.debug(self.partition_size)
        generator = self._data.backend._get_batch_generator(
            random=random,
            batch_size=self.partition_size,
            dataset_type=self._data.dataset_type,
            index_by=index_by,
        )

        for partition_nb, partition in enumerate(generator):
            fname = "partition_{}_{}.csv".format(partition_nb, timestamp)
            fname = os.path.join(self.root_dir, fname)
            self.files.append(fname)
            np.savetxt(
                fname, np.array(partition, dtype=int), delimiter="\t", fmt="%d"
            )
            # special case of GraphDataLoader to create partition datasets:
            # with remapped indexes (0, size_of_partition),
            # persisted, with partition number to look up remappings
            partition_loader = GraphDataLoader(
                fname,
                use_indexer=False,
                batch_size=batch_size,
                remap=True,
                parent=self._data,
                name="partition_{}".format(partition_nb),
            )
            self.partitions.append(partition_loader)


@register_partitioning_strategy("RandomEdges", "GeneralPartitionDataManager")
class RandomEdgesGraphPartitioner(EdgeBasedGraphPartitioner):
    """Partitioning strategy that splits edges into equal size
    partitions randomly drawing triples from the data.

    Example
    -------
    >>> from ampligraph.datasets imoprt load_fb15k_237
    >>> from ampligraph.datasets import GraphDataLoader
    >>> from ampligraph.datasets.sqlite_adapter import SQLiteAdapter
    >>> from ampligraph.datasets.graph_partitioner import RandomEdgesGraphPartitioner
    >>> from ampligraph.latent_features import ScoringBasedEmbeddingModel
    >>> dataset = load_fb15k_237()
    >>> dataset_loader = GraphDataLoader(dataset['train'],
    >>>                                  backend=SQLiteAdapter, # type of backend to use
    >>>                                  batch_size=2,          # batch size to use while iterating over this dataset
    >>>                                  dataset_type='train',  # dataset type
    >>>                                  use_filter=False,      # Whether to use filter or not
    >>>                                  use_indexer=True)      # indicates that the data needs to be mapped to index
    >>> partitioner = RandomEdgesGraphPartitioner(dataset_loader, k=2)
    >>> # create and compile a model as usual
    >>> partitioned_model = ScoringBasedEmbeddingModel(eta=2,
    >>>                                                k=50,
    >>>                                                scoring_type='DistMult')
    >>>
    >>> partitioned_model.compile(optimizer='adam', loss='multiclass_nll')
    >>>
    >>> partitioned_model.fit(partitioner,            # pass the partitioner object as input to the fit function this will generate data for the model during training
    >>>                       epochs=10)              # number of epochs

    """

    def __init__(self, data, k=2, **kwargs):
        """Initialise the RandomEdgesGraphPartitioner.

        Parameters
        ----------
        data: GraphDataLoader
            Input data as a GraphDataLoader.
        k: int
            Number of buckets to split entities (i.e., vertices) into.

        """
        self.partitions = []
        self._data = data
        self._k = k
        super().__init__(data, k, random=True, index_by="", **kwargs)


@register_partitioning_strategy("Naive", "GeneralPartitionDataManager")
class NaiveGraphPartitioner(EdgeBasedGraphPartitioner):
    """Partitioning strategy that splits edges into equal size
    partitions drawing triples from the data sequentially.

    Example
    -------
    >>> from ampligraph.datasets imoprt load_fb15k_237
    >>> from ampligraph.datasets import GraphDataLoader
    >>> from ampligraph.datasets.sqlite_adapter import SQLiteAdapter
    >>> from ampligraph.datasets.graph_partitioner import NaiveGraphPartitioner
    >>> from ampligraph.latent_features import ScoringBasedEmbeddingModel
    >>> dataset = load_fb15k_237()
    >>> dataset_loader = GraphDataLoader(dataset['train'],
    >>>                                  backend=SQLiteAdapter, # type of backend to use
    >>>                                  batch_size=2,          # batch size to use while iterating over this dataset
    >>>                                  dataset_type='train',  # dataset type
    >>>                                  use_filter=False,      # Whether to use filter or not
    >>>                                  use_indexer=True)      # indicates that the data needs to be mapped to index
    >>> partitioner = NaiveGraphPartitioner(dataset_loader, k=2)
    >>> # create and compile a model as usual
    >>> partitioned_model = ScoringBasedEmbeddingModel(eta=2,
    >>>                                                k=50,
    >>>                                                scoring_type='DistMult')
    >>>
    >>> partitioned_model.compile(optimizer='adam', loss='multiclass_nll')
    >>>
    >>> partitioned_model.fit(partitioner,            # pass the partitioner object as input to the fit function this will generate data for the model during training
    >>>                       epochs=10)              # number of epochs

    """

    def __init__(self, data, k=2, **kwargs):
        """Initialise the NaiveGraphPartitioner.

        Parameters
        ----------
        data: GraphDataLoader
            Input data as a GraphDataLoader.
        k: int
            Number of buckets to split entities (i.e., vertices) into.

        """
        self.partitions = []
        super().__init__(data, k, random=False, index_by="", **kwargs)


@register_partitioning_strategy("SortedEdges", "GeneralPartitionDataManager")
class SortedEdgesGraphPartitioner(EdgeBasedGraphPartitioner):
    """Partitioning strategy that splits edges into equal size
    partitions retrieving triples from the data ordered by subject.

    Example
    -------
    >>> from ampligraph.datasets imoprt load_fb15k_237
    >>> from ampligraph.datasets import GraphDataLoader
    >>> from ampligraph.datasets.sqlite_adapter import SQLiteAdapter
    >>> from ampligraph.datasets.graph_partitioner import SortedEdgesGraphPartitioner
    >>> from ampligraph.latent_features import ScoringBasedEmbeddingModel
    >>> dataset = load_fb15k_237()
    >>> dataset_loader = GraphDataLoader(dataset['train'],
    >>>                                  backend=SQLiteAdapter, # type of backend to use
    >>>                                  batch_size=2,          # batch size to use while iterating over this dataset
    >>>                                  dataset_type='train',  # dataset type
    >>>                                  use_filter=False,      # Whether to use filter or not
    >>>                                  use_indexer=True)      # indicates that the data needs to be mapped to index
    >>> partitioner = SortedEdgesGraphPartitioner(dataset_loader, k=2)
    >>> # create and compile a model as usual
    >>> partitioned_model = ScoringBasedEmbeddingModel(eta=2,
    >>>                                                k=50,
    >>>                                                scoring_type='DistMult')
    >>>
    >>> partitioned_model.compile(optimizer='adam', loss='multiclass_nll')
    >>>
    >>> partitioned_model.fit(partitioner,            # pass the partitioner object as input to the fit function this will generate data for the model during training
    >>>                       epochs=10)              # number of epochs

    """

    def __init__(self, data, k=2, **kwargs):
        """Initialise the SortedEdgesGraphPartitioner.

        Parameters
        ----------
        data: GraphDataLoader
            Input data as a GraphDataLoader.
        k: int
            Number of buckets to split entities (i.e., vertices) into.

        """

        self.partitions = []
        super().__init__(data, k, random=False, index_by="s", **kwargs)


@register_partitioning_strategy(
    "DoubleSortedEdges", "GeneralPartitionDataManager"
)
class DoubleSortedEdgesGraphPartitioner(EdgeBasedGraphPartitioner):
    """Partitioning strategy that splits edges into equal size
    partitions retrieving triples from the data ordered by subject and object.

    Example
    -------
    >>> from ampligraph.datasets imoprt load_fb15k_237
    >>> from ampligraph.datasets import GraphDataLoader
    >>> from ampligraph.datasets.sqlite_adapter import SQLiteAdapter
    >>> from ampligraph.datasets.graph_partitioner import DoubleSortedEdgesGraphPartitioner
    >>> from ampligraph.latent_features import ScoringBasedEmbeddingModel
    >>> dataset = load_fb15k_237()
    >>> dataset_loader = GraphDataLoader(dataset['train'],
    >>>                                  backend=SQLiteAdapter, # type of backend to use
    >>>                                  batch_size=2,          # batch size to use while iterating over this dataset
    >>>                                  dataset_type='train',  # dataset type
    >>>                                  use_filter=False,      # Whether to use filter or not
    >>>                                  use_indexer=True)      # indicates that the data needs to be mapped to index
    >>> partitioner = DoubleSortedEdgesGraphPartitioner(dataset_loader, k=2)
    >>> # create and compile a model as usual
    >>> partitioned_model = ScoringBasedEmbeddingModel(eta=2,
    >>>                                                k=50,
    >>>                                                scoring_type='DistMult')
    >>>
    >>> partitioned_model.compile(optimizer='adam', loss='multiclass_nll')
    >>>
    >>> partitioned_model.fit(partitioner,            # pass the partitioner object as input to the fit function this will generate data for the model during training
    >>>                       epochs=10)              # number of epochs

    """

    def __init__(self, data, k=2, **kwargs):
        """Initialise the DoubleSortedEdgesGraphPartitioner.

        Parameters
        ----------
        data: GraphDataLoader
            Input data as a GraphDataLoader.
        k: int
            Number of buckets to split entities (i.e., vertices) into.

        """
        self.partitions = []
        super().__init__(data, k, random=False, index_by="so", **kwargs)


def main():
    pass


if __name__ == "__main__":
    main()
