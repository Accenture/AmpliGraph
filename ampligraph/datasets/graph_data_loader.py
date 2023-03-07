# Copyright 2019-2023 The AmpliGraph Authors. All Rights Reserved.
#
# This file is Licensed under the Apache License, Version 2.0.
# A copy of the Licence is available in LICENCE, or at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
"""Data loader for graphs (big and small).

This module provides GraphDataLoader class that can be parametrized with an artificial backend that reads data in-memory
(:class:`~ampligraph.datasets.graph_data_loader.NoBackend`) or with a SQLite backend that stores and reads data
on-disk (:class:`~ampligraph.datasets.sqlite_adapter.SQLiteAdapter`).
"""
import logging
import tempfile
import uuid
from datetime import datetime

import numpy as np
import tensorflow as tf

from .data_indexer import DataIndexer
from .source_identifier import DataSourceIdentifier

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class NoBackend:
    """Class providing an artificial backend, that reads data into memory."""

    def __init__(
        self,
        identifier,
        use_indexer=True,
        remap=False,
        name="main_partition",
        parent=None,
        in_memory=True,
        root_directory=None,
        use_filter=False,
        verbose=False,
    ):
        """Initialise NoBackend.

        Parameters
        ----------
        identifier: initialize data source identifier, provides loader.
        use_indexer: bool or mapper object
             Flag or mapper object to tell whether data should be indexed (default: `False`).
        remap: bool
             Flag for partitioner to indicate whether to remap previously indexed data to (0, <size_of_partition>)
             (default: `False`).
        name: str
             Name identifying files for the indexer, partition name/id.
        parent:
             Parent data loader that persists data.
        verbose: bool
             Verbosity.
        """
        # in_memory = False
        self.verbose = verbose
        self.identifier = identifier
        self.use_indexer = use_indexer
        self.remap = remap
        self.name = name
        self.parent = parent
        self.in_memory = in_memory
        if root_directory is None:
            self.root_directory = tempfile.gettempdir()
        else:
            self.root_directory = root_directory
        self.use_filter = use_filter
        self.sources = {}

    def _add_dataset(self, data_source, dataset_type):
        msg = "Adding datasets to NoBackend not possible."
        raise NotImplementedError(msg)

    def __enter__(self):
        """Context manager enter function. Required by GraphDataLoader."""
        return self

    def __exit__(self, type, value, tb):
        """Context manager exit function. Required by GraphDataLoader."""
        pass

    def get_output_signature(self):
        """Get the output signature for the tf.data.Dataset object."""
        triple_tensor = tf.TensorSpec(shape=(None, 3), dtype=tf.int32)
        if self.data_shape > 3:
            weights_tensor = tf.TensorSpec(
                shape=(None, self.data_shape - 3), dtype=tf.float32
            )
            if self.use_filter:
                return (
                    triple_tensor,
                    tf.RaggedTensorSpec(shape=(2, None, None), dtype=tf.int32),
                    weights_tensor,
                )
            else:
                return (triple_tensor, weights_tensor)
        if self.use_filter:
            return (
                triple_tensor,
                tf.RaggedTensorSpec(shape=(2, None, None), dtype=tf.int32),
            )
        return triple_tensor

    def _load(self, data_source, dataset_type):
        """Loads data into self.data.

        Parameters
        ----------
        data_source: np.array or str
             Array or name of the file containing the data.
        dataset_type: str
             Kind of data to be loaded (`"train"` | `"test"` | `"validation"`).
        """
        logger.debug(
            "Simple in-memory data loading of {} dataset.".format(dataset_type)
        )
        self.data_source = data_source
        self.dataset_type = dataset_type
        if isinstance(self.data_source, np.ndarray):
            if self.use_indexer is True:
                self.mapper = DataIndexer(
                    self.data_source,
                    backend="in_memory" if self.in_memory else "sqlite",
                    root_directory=self.root_directory,
                )
                self.data = self.mapper.get_indexes(self.data_source)
            elif self.remap:
                # create a special mapping for partitions, persistent mapping from main indexes
                # to partition indexes
                self.mapper = DataIndexer(
                    self.data_source,
                    backend="sqlite",
                    name=self.name,
                    root_directory=self.root_directory,
                )
                self.data = self.mapper.get_indexes(self.data_source)
            else:
                self.mapper = self.use_indexer
                self.data = self.mapper.get_indexes(self.data_source)
        else:
            loader = self.identifier.fetch_loader()
            raw_data = loader(self.data_source)
            if self.use_indexer is True:
                self.mapper = DataIndexer(
                    raw_data,
                    backend="in_memory" if self.in_memory else "sqlite",
                    root_directory=self.root_directory,
                )
                self.data = self.mapper.get_indexes(raw_data)
            elif self.use_indexer is False:
                if self.remap:
                    # create a special mapping for partitions, persistent mapping from
                    # main indexes to partition indexes
                    self.mapper = DataIndexer(
                        raw_data,
                        backend="sqlite",
                        name=self.name,
                        root_directory=self.root_directory,
                    )
                    self.data = self.mapper.get_indexes(raw_data)
                else:
                    self.data = raw_data
                    logger.debug("Data won't be indexed")
            elif isinstance(self.use_indexer, DataIndexer):
                self.mapper = self.use_indexer
                self.data = self.mapper.get_indexes(raw_data)
        self.data_shape = self.mapper.backend.data_shape

    def _get_triples(self, subjects=None, objects=None, entities=None):
        """Get triples whose subjects belongs to ``subjects``, objects to ``objects``,
        or, if neither object nor subject is provided, triples whose subject or object belong to entities.
        """
        if subjects is None and objects is None:
            if entities is None:
                msg = "You have to provide either subjects and objects indexes or general entities indexes!"
                logger.error(msg)
                raise Exception(msg)

            subjects = entities
            objects = entities
        # check_subjects = np.vectorize(lambda t: t in subjects)
        if subjects is not None and objects is not None:
            check_triples = np.vectorize(
                lambda t, r: (t in objects and r in subjects)
                or (t in subjects and r in objects)
            )
            triples = self.data[
                check_triples(self.data[:, 2], self.data[:, 0])
            ]
        elif objects is None:
            triples = self.data[np.isin(self.data[:, 0], subjects)]
        elif subjects is None:
            triples = self.data[np.isin(self.data[:, 2], objects)]
        triples = np.append(
            triples,
            np.array(len(triples) * [self.dataset_type]).reshape(-1, 1),
            axis=1,
        )
        # triples_from_objects = self.data[check_objects(self.data[:,0])]
        # triples = np.vstack([triples_from_subjects, triples_from_objects])
        return triples

    def get_data_size(self):
        """Returns number of triples."""
        return np.shape(self.data)[0]

    def _get_complementary_entities(self, triples, use_filter=None):
        """Get subjects and objects complementary to a triple (?,p,?).

        Returns the participating entities in the relation ?-p-o and s-p-?.
        Function used during evaluation.

        WARNING: If the parent is set the triples returned are coming with parent indexing.

        Parameters
        ----------
        x_triple: nd-array, shape (N, 3)
            Triples `(s, p, o)` that we are querying.

        Returns
        -------
        entities: tuple
             Tuple containing two lists, one with the subjects and one of with the objects participating in the
             relations ?-p-o and s-p-?.
        """

        logger.debug("Getting complementary entities")

        if self.parent is not None:
            logger.debug(
                "Parent is set, WARNING: The triples returned are coming with parent indexing."
            )

            logger.debug("Recover original indexes.")
            triples_original_index = self.mapper.get_indexes(
                triples, order="ind2raw"
            )
            #            with shelve.open(self.mapper.entities_dict) as ents:
            #                with shelve.open(self.mapper.relations_dict) as rels:
            #                    triples_original_index = np.array([(ents[str(xx[0])], rels[str(xx[1])],
            # ents[str(xx[2])]) for xx in triples], dtype=np.int32)
            logger.debug("Query parent for data.")
            logger.debug("Original index: {}".format(triples_original_index))
            subjects = self.parent.get_complementary_subjects(
                triples_original_index, use_filter=use_filter
            )
            objects = self.parent.get_complementary_objects(
                triples_original_index, use_filter=use_filter
            )
            logger.debug(
                "What to do with this new indexes? Evaluation should happen in the original space, \
            shouldn't it? I'm assuming it does so returning in parent indexing."
            )
            return subjects, objects
        else:
            subjects = self._get_complementary_subjects(
                triples, use_filter=use_filter
            )
            objects = self._get_complementary_objects(
                triples, use_filter=use_filter
            )
        return subjects, objects

    def _get_complementary_subjects(self, triples, use_filter=False):
        """Get subjects complementary to triples (?,p,o).

        For a given triple retrieve all subjects coming from triples with same objects and predicates.

        Parameters
        ----------
        triples : list or array
             List or array of arrays with 3 elements (subject, predicate, object).

        Returns
        -------
        subjects : list
             Subjects present in the input triples.
        """

        logger.debug("Getting complementary subjects")

        if self.parent is not None:
            logger.debug(
                "Parent is set, WARNING: The triples returned are coming with parent indexing."
            )

            logger.debug("Recover original indexes.")
            triples_original_index = self.mapper.get_indexes(
                triples, order="ind2raw"
            )

            #            with shelve.open(self.mapper.reversed_entities_dict) as ents:
            #                with shelve.open(self.mapper.reversed_relations_dict) as rels:
            #                    triples_original_index = np.array([(ents[str(xx[0])],
            #                                                        rels[str(xx[1])],
            #                                                        ents[str(xx[2])]) for xx in triples],
            #                                                      dtype=np.int32)
            logger.debug("Query parent for data.")
            subjects = self.parent.get_complementary_subjects(
                triples_original_index
            )
            logger.debug(
                "What to do with this new indexes? Evaluation should happen in the \
            original space, shouldn't it? I'm assuming it does so returning in parent indexing."
            )
            return subjects
        elif self.use_filter is False or self.use_filter is None:
            self.use_filter = {"train-org": self.data}

        filtered = []
        for filter_name, filter_source in self.use_filter.items():
            source = self.get_source(filter_source, filter_name)

            tmp_filter = []
            for triple in triples:
                tmp = source[source[:, 2] == triple[2]]
                tmp_filter.append(list(set(tmp[tmp[:, 1] == triple[1]][:, 0])))
            filtered.append(tmp_filter)
        # Unpack data into one list per triple no matter what filter it comes
        # from
        unpacked = list(zip(*filtered))
        subjects = []
        for k in unpacked:
            lst = [j for i in k for j in i]
            subjects.append(np.array(lst, dtype=np.int32))

        return subjects

    def get_source(self, source, name):
        """Loads the data specified by ``name`` and keep it in the loaded dictionary.

        Used to load filter datasets.

        Parameters
        ----------
        source: ndarray or str
             Data source to load data from.
        name: str
             Name of the dataset to be loaded.

        Returns
        -------
        Loaded data : ndarray
             Numpy array containing loaded data indexed according to mapper.
        """
        if name not in self.sources:
            if isinstance(source, np.ndarray):
                raw_data = source
            else:
                identifier = DataSourceIdentifier(source)
                loader = identifier.fetch_loader()
                raw_data = loader(source)
            if name != "train-org":
                self.sources[name] = self.mapper.get_indexes(raw_data)
            else:
                self.sources[name] = raw_data
        return self.sources[name]

    def _get_complementary_objects(self, triples, use_filter=False):
        """Get objects complementary to triples (s,p,?).

        For a given triple retrieves all triples with same subjects and predicates.
        Function used during evaluation.

        Parameters
        ----------
        triples : list or array
             List or array of arrays with 3 elements (subject, predicate, object).

        Returns
        -------
        subjects : list
             Objects present in the input triples.
        """
        logger.debug("Getting complementary objects")

        if self.parent is not None:
            logger.debug(
                "Parent is set, WARNING: The triples returned are coming with parent indexing."
            )

            logger.debug("Recover original indexes.")
            triples_original_index = self.mapper.get_indexes(
                triples, order="ind2raw"
            )
            #            with shelve.open(self.mapper.reversed_entities_dict) as ents:
            #                with shelve.open(self.mapper.reversed_relations_dict) as rels:
            #                    triples_original_index = np.array([(ents[str(xx[0])], rels[str(xx[1])],
            # ents[str(xx[2])]) for xx in triples], dtype=np.int32)
            logger.debug("Query parent for data.")
            objects = self.parent.get_complementary_objects(
                triples_original_index
            )
            logger.debug(
                "What to do with this new indexes? Evaluation should happen in \
            the original space, shouldn't it? I'm assuming it does so returning in parent indexing."
            )
            return objects
        elif self.use_filter is False or self.use_filter is None:
            self.use_filter = {"train-org": self.data}
        filtered = []
        for filter_name, filter_source in self.use_filter.items():
            source = self.get_source(filter_source, filter_name)

            # load source if not loaded
            source = self.get_source(filter_source, filter_name)
            # filter

            tmp_filter = []
            for triple in triples:
                tmp = source[source[:, 0] == triple[0]]
                tmp_filter.append(list(set(tmp[tmp[:, 1] == triple[1]][:, 2])))
            filtered.append(tmp_filter)

        # Unpack data into one  list per triple no matter what filter it comes
        # from
        unpacked = list(zip(*filtered))
        objects = []
        for k in unpacked:
            lst = [j for i in k for j in i]
            objects.append(np.array(lst, dtype=np.int32))

        return objects

    def _intersect(self, dataloader):
        """Intersection between data and dataloader elements.

        Works only when dataloader is of type `NoBackend`.
        """
        if not isinstance(dataloader.backend, NoBackend):
            msg = "Intersection can only be calculated between same backends (NoBackend), \
            instead get {}".format(
                type(dataloader.backend)
            )
            logger.error(msg)
            raise Exception(msg)
        self.data = np.ascontiguousarray(self.data, dtype="int64")
        dataloader.backend.data = np.ascontiguousarray(
            dataloader.backend.data, dtype="int64"
        )
        av = self.data.view([("", self.data.dtype)] * self.data.shape[1])
        bv = dataloader.backend.data.view(
            [("", dataloader.backend.data.dtype)]
            * dataloader.backend.data.shape[1]
        )
        intersection = (
            np.intersect1d(av, bv)
            .view(self.data.dtype)
            .reshape(
                -1,
                self.data.shape[0 if self.data.flags["F_CONTIGUOUS"] else 1],
            )
        )
        return intersection

    def _get_batch_generator(
        self, batch_size, dataset_type="train", random=False, index_by=""
    ):
        """Data batch generator.

        Parameters
        ----------
        batch_size: int
             Size of a batch.
        dataset_type: str
             Kind of dataset that is needed (`"train"` | `"test"` | `"validation"`).
        random: not implemented.
        index_by: not implemented.

        Returns
        --------
        Batch : ndarray
             Batch of data of size `(batch_size, m)` where :math:`m≥3` and :math:`m>3` if numeric values
             associated to edges are available.
        """
        if not isinstance(batch_size, int):
            batch_size = int(batch_size)
        length = len(self.data)
        triples = range(0, length, batch_size)
        for start_index in triples:
            # if the last batch is smaller than the batch_size
            if start_index + batch_size >= length:
                batch_size = length - start_index
            out = self.data[start_index: start_index + batch_size,:3]
            if self.use_filter:
                # get the filter values
                participating_entities = self._get_complementary_entities(
                    out, self.use_filter
                )

            # focusE
            if self.data_shape > 3:
                weights = self.data[start_index: start_index + batch_size, 3:]
                # weights = preprocess_focusE_weights(data=out,
                #                                     weights=weights)
                if self.use_filter:
                    yield out, tf.ragged.constant(
                        participating_entities, dtype=tf.int32
                    ), weights
                else:
                    yield out, weights

            else:
                if self.use_filter:
                    yield out, tf.ragged.constant(
                        participating_entities, dtype=tf.int32
                    )
                else:
                    yield out

    def _clean(self):
        del self.data
        self.mapper.clean()


class GraphDataLoader:
    """Data loader for models to ingest graph data.

    This class is internally used by the model to store the data passed by the user and batch over it during
    training and evaluation, and to obtain filters during evaluation.

    It can be used by advanced users to load custom datasets which are large, for performing partitioned training.
    The complete dataset will not get loaded in memory. It will load the data in chunks based on which partition
    is being trained.

    Example
    -------
    >>> from ampligraph.datasets import GraphDataLoader, BucketGraphPartitioner
    >>> from ampligraph.datasets.sqlite_adapter import SQLiteAdapter
    >>> from ampligraph.latent_features import ScoringBasedEmbeddingModel
    >>> AMPLIGRAPH_DATA_HOME='/your/path/to/datasets/'
    >>> # Graph loader - loads the data from the file, numpy array, etc and generates batches for iterating
    >>> path_to_training = AMPLIGRAPH_DATA_HOME + 'fb15k-237/train.txt'
    >>> dataset_loader = GraphDataLoader(path_to_training,
    >>>                                  backend=SQLiteAdapter, # type of backend to use
    >>>                                  batch_size=1000,       # batch size to use while iterating over this dataset
    >>>                                  dataset_type='train',  # dataset type
    >>>                                  use_filter=False,      # Whether to use filter or not
    >>>                                  use_indexer=True)      # indicates that the data needs to be mapped to index
    >>>
    >>> # Choose the partitioner - in this case we choose RandomEdges partitioner
    >>> partitioner = BucketGraphPartitioner(dataset_loader, k=3)
    >>> partitioned_model = ScoringBasedEmbeddingModel(eta=2,
    >>>                                                k=50,
    >>>                                                scoring_type='DistMult')
    >>> partitioned_model.compile(optimizer='adam', loss='multiclass_nll')
    >>> partitioned_model.fit(partitioner,            # pass the partitioner object as input to the fit function this will generate data for the model during training
    >>>                       epochs=10)              # number of epochs
    >>> indexer = partitioned_model.data_handler.get_mapper()    # get the mapper from the trained model
    >>> path_to_test = AMPLIGRAPH_DATA_HOME + 'fb15k-237/test.txt'
    >>> dataset_loader_test = GraphDataLoader(path_to_test,
    >>>                                       backend=SQLiteAdapter,                         # type of backend to use
    >>>                                       batch_size=400,                                # batch size to use while iterating over this dataset
    >>>                                       dataset_type='test',                           # dataset type
    >>>                                       use_indexer=indexer                            # mapper to map test concepts to the same indices used during training
    >>>                                       )
    >>> ranks = partitioned_model.evaluate(dataset_loader_test, # pass the dataloader object to generate data for the model during training
    >>>                                    batch_size=400)
    >>> print(ranks)
    [[  85    7]
     [  95    9]
     [1074   22]
     ...
     [ 546   95]
     [9961 7485]
     [1494    2]]


    """

    def __init__(
        self,
        data_source,
        batch_size=1,
        dataset_type="train",
        backend=None,
        root_directory=None,
        use_indexer=True,
        verbose=False,
        remap=False,
        name="main_partition",
        parent=None,
        in_memory=False,
        use_filter=False,
    ):
        """Initialise persistent/in-memory data storage.

        Parameters
        ----------
        data_source: str or np.array or GraphDataLoader or AbstractGraphPartitioner
            File with data (e.g. CSV). Can be a path pointing to the file location, can be data loaded as numpy, a
            `GraphDataLoader` or an `AbstractGraphPartitioner` instance.
        batch_size: int
            Size of batch.
        dataset_type: str
            Kind of data provided (`"train"` | `"test"` | `"valid"`).
        backend: str
            Name of backend class (`NoBackend`, `SQLiteAdapter`) or already initialised backend.
            If `None`, `NoBackend` is used (in-memory processing).
        root_directory: str
             Path to a directory where the database will be created, and the data and mappings will be stored.
             If `None`, the root directory is obtained through the :meth:`tempfile.gettempdir()` method
             (default: `None`).
        use_indexer: bool or DataIndexer
            Flag to tell whether data should be indexed.
            If the DataIndexer object is passed, the mappings defined in the indexer will be reused
            to generate mappings for the current data.
        verbose: bool
            Verbosity.
        remap: bool
            Flag to be used by graph partitioner, indicates whether previously indexed data in partition has to
            be remapped to new indexes (0, <size_of_partition>). It has not to be used with ``use_indexer=True``.
            The new remappings will be persisted.
        name: str
            Name of the partition. This is internally used when the data is partitioned.
        parent: GraphDataLoader
            Parent dataloader. This is internally used when the data is partitioned.
        in_memory: bool
            Persist indexes or not.
        use_filter: bool or dict
            If `True`, current dataset will be used as filter.
            If `dict`, the datasets specified in the dict will be used for filtering.
            If `False`, the true positives will not be filtered from corruptions.
        """
        self.dataset_type = dataset_type
        self.data_source = data_source
        self.batch_size = batch_size
        if root_directory is None:
            self.root_directory = tempfile.gettempdir()
        else:
            self.root_directory = root_directory
        self.identifier = DataSourceIdentifier(self.data_source)
        self.use_indexer = use_indexer
        self.remap = remap
        self.in_memory = in_memory
        self.name = name
        self.parent = parent
        if use_filter is None or use_filter is True:
            self.use_filter = {"train": data_source}
        else:
            if isinstance(use_filter, dict) or use_filter is False:
                self.use_filter = use_filter
            else:
                msg = "use_filter should be a dictionary with keys as names of filters and \
                values as data sources, instead got {}".format(
                    use_filter
                )
                logger.error(msg)
                raise Exception(msg)
        if bool(use_indexer) != (not remap):
            msg = (
                "Either remap or Indexer should be specified at the same time."
            )
            logger.error(msg)
            raise Exception(msg)
        if isinstance(backend, type) and backend != NoBackend:
            self.backend = backend(
                "database_{}_{}.db".format(
                    datetime.now().strftime("%d-%m-%Y_%I-%M-%S_%f_%p"),
                    str(uuid.uuid4()),
                ),
                identifier=self.identifier,
                root_directory=self.root_directory,
                use_indexer=self.use_indexer,
                remap=self.remap,
                name=self.name,
                parent=self.parent,
                in_memory=self.in_memory,
                verbose=verbose,
                use_filter=self.use_filter,
            )
            logger.debug(
                "Initialized Backend with database at: {}".format(
                    self.backend.db_path
                )
            )

        elif backend is None or backend == NoBackend:
            self.backend = NoBackend(
                self.identifier,
                use_indexer=self.use_indexer,
                remap=self.remap,
                name=self.name,
                parent=self.parent,
                in_memory=self.in_memory,
                use_filter=self.use_filter,
            )
        else:
            self.backend = backend

        self.backend._load(self.data_source, dataset_type=self.dataset_type)
        self.data_shape = self.backend.data_shape
        self.batch_iterator = self.get_batch_generator(
            use_filter=self.use_filter, dataset_type=self.dataset_type
        )
        self.metadata = self.backend.mapper.metadata

    def __iter__(self):
        """Function needed to be used as an iterator."""
        return self

    @property
    def max_entities(self):
        """Maximum number of entities present in the dataset mapper."""
        return self.backend.mapper.get_entities_count()

    @property
    def max_relations(self):
        """Maximum number of relations present in the dataset mapper."""
        return self.backend.mapper.get_relations_count()

    def __next__(self):
        """Function needed to be used as an iterator."""
        return self.batch_iterator.__next__()

    def reload(self, use_filter=False, dataset_type="train"):
        """Reinstantiate batch iterator."""
        self.batch_iterator = self.get_batch_generator(
            use_filter=use_filter, dataset_type=dataset_type
        )

    def get_batch_generator(self, dataset_type="train", use_filter=False):
        """Get batch generator from the backend.

        Parameters
        ----------
        dataset_type: str
             Specifies whether data are generated for `"train"`, `"valid"` or `"test"` set.
        """
        return self.backend._get_batch_generator(
            self.batch_size, dataset_type=dataset_type
        )

    def get_tf_generator(self):
        """Generates a tensorflow.data.Dataset object."""
        return tf.data.Dataset.from_generator(
            self.backend._get_batch_generator,
            output_signature=self.backend.get_output_signature(),
            args=(self.batch_size, self.dataset_type, False, ""),
        ).prefetch(2)

    def add_dataset(self, data_source, dataset_type):
        """Adds the dataset to the backend (if possible)."""
        self.backend._add_dataset(data_source, dataset_type=dataset_type)

    def get_data_size(self):
        """Returns number of triples."""
        return self.backend.get_data_size()

    def intersect(self, dataloader):
        """Returns the intersection between the current data loader and another one specified in ``dataloader``.

        Parameters
        ----------
        dataloader: GraphDataLoader
             Dataloader for which to calculate the intersection for.

        Returns
        -------
        intersection: ndarray
             Array of intersecting elements.
        """

        return self.backend._intersect(dataloader)

    def get_participating_entities(
        self, triples, sides="s,o", use_filter=False
    ):
        """Get entities from triples with fixed subjects or fixed objects or both fixed.

        Parameters
        ----------
        triples: list or array
             List or array of arrays with 3 elements (subject, predicate, object).
        sides : str
             String specifying what entities to retrieve: `"s"` - subjects, `"o"` - objects,
             `"s,o"` - subjects and objects, `"o,s"` - objects and subjects.

        Returns
        -------
        entities : list
             List of subjects (if ``sides="s"``) or objects (if ``sides="o"``) or two lists with both
             (if ``sides="s,o"`` or ``sides="o,s"``).
        """
        if sides not in ["s", "o", "s,o", "o,s"]:
            msg = "Sides should be either 's' (subject), 'o' (object), or 's,o'/'o,s' (subject, object/object, subject), \
            instead got {}".format(
                sides
            )
            logger.error(msg)
            raise Exception(msg)
        if "s" in sides:
            subjects = self.get_complementary_subjects(
                triples, use_filter=use_filter
            )

        if "o" in sides:
            objects = self.get_complementary_objects(
                triples, use_filter=use_filter
            )

        if sides == "s,o":
            return subjects, objects
        if sides == "o,s":
            return objects, subjects
        if sides == "s":
            return subjects
        if sides == "o":
            return objects

    def get_complementary_subjects(self, triples, use_filter=False):
        """Get subjects complementary to triples (?,p,o).

        For a given triple retrieve all subjects coming from triples with same objects and predicates.

        Parameters
        ----------
        triples : list or array
             List or array of arrays with 3 elements (subject, predicate, object).

        Returns
        -------
        subjects : list
             Subjects present in the input triples.
        """
        return self.backend._get_complementary_subjects(
            triples, use_filter=use_filter
        )

    def get_complementary_objects(self, triples, use_filter=False):
        """Get objects complementary to  triples (s,p,?).

        For a given triple retrieve all triples with same subjects and predicates.
        Function used during evaluation.

        Parameters
        ----------
        triples : list or array
             List or array of arrays with 3 elements (subject, predicate, object).

        Returns
        -------
        subjects : list
             Objects present in the input triples.
        """
        return self.backend._get_complementary_objects(
            triples, use_filter=use_filter
        )

    def get_complementary_entities(self, triples, use_filter=False):
        """Get subjects and objects complementary to triples (?,p,?).

        Returns the participating entities in the relation ?-p-o and s-p-?.

        Parameters
        ----------
        x_triple: nd-array (N,3,)
           N triples (s-p-o) that we are querying.

        Returns
        -------
        entities: tuple
              Tuple containing two lists, one with the subjects and one of with the objects participating in the
             relations ?-p-o and s-p-?.
        """
        return self.backend._get_complementary_entities(
            triples, use_filter=use_filter
        )

    def get_triples(self, subjects=None, objects=None, entities=None):
        """Get triples that subject is in subjects and object is in objects, or
        triples that eiter subject or object is in entities.

        Parameters
        ----------
        subjects: list
             List of entities that triples subject should belong to.

        objects: list
             List of entities that triples object should belong to.

        entities: list
             List of entities that triples subject and object should belong to.

        Returns
        -------
        triples: list
             List of triples constrained by subjects and objects.

        """
        return self.backend._get_triples(subjects, objects, entities)

    def clean(self):
        """Cleans up the temporary files created for training/evaluation."""
        self.backend._clean()

    def on_epoch_end(self):
        pass

    def on_complete(self):
        pass
