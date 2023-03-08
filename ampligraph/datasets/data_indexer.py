# Copyright 2019-2023 The AmpliGraph Authors. All Rights Reserved.
#
# This file is Licensed under the Apache License, Version 2.0.
# A copy of the Licence is available in LICENCE, or at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
"""Data indexer.

This module provides a class that maps raw data to indexes and the other way around.
It can be persisted and contains supporting functions.

Example
-------
    >>>data = np.array([['/m/01', '/relation1', '/m/02'],
    >>>                 ['/m/01', '/relation2', '/m/07']])
    >>>mapper = DataIndexer(data)
    >>>mapper.get_indexes(data)

.. It extends functionality of to_idx(...) from  AmpliGraph 1:
   https://docs.ampligraph.org/en/1.3.1/generated/ampligraph.evaluation.to_idx.html?highlight=to_idx

"""
import logging
import os
import shelve
import shutil
import sqlite3
import tempfile
import uuid
from datetime import datetime

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
INDEXER_BACKEND_REGISTRY = {}


class DataIndexer:
    """Index graph unique entities and relations.

    Abstract class with unified API for different indexers implementations (`in-memory`, `shelves`, `sqlite`).

    It can support large datasets in two modelities:
    - using dictionaries for in-memory storage
    - using persistent dictionary storage (python shelves, sqlite), for dumping huge indexes.

    Methods:
     - create_mappings - core function that creates mappings.
     - get_indexes - given an array of triples, returns it in an indexed form,
       or given indexes, it returns the original triples (subject to parameters).
     - update_mappings [NotYetImplemented] - update mappings from new data.

    Properties:
     - data - data to be indexed, either a numpy array or a generator.

     Example
     -------
     >>> from ampligraph.datasets.data_indexer import DataIndexer
     >>> import numpy as np
     >>> # In-memory mapping
     >>> data = np.array([['a','b','c'],['c','b','d'],['d','e','f']])
     >>> mapper = DataIndexer(data, backend='in_memory')
     >>> print(mapper.get_indexes(data))
     [[0 0 1]
      [1 0 2]
      [2 1 3]]
     >>> # Persistent mapping
     >>> data = np.array([['a','b','c'],['c','b','d'],['d','e','f']])
     >>> mapper = DataIndexer(data, backend='sqlite')
     >>> print(mapper.get_indexes(data))
     [[0 0 1]
      [1 0 2]
      [2 1 3]]
    """

    def __init__(self, X, backend="in_memory", **kwargs):
        """Initialises the DataIndexer."""
        logger.debug("Initialisation of DataIndexer.")
        self.backend_type = backend
        self.data = X
        #        if len(kwargs) == 0:
        #            self.backend = INDEXER_BACKEND_REGISTRY.get(backend)(X)
        #        else:
        self.backend = INDEXER_BACKEND_REGISTRY.get(backend)(X, **kwargs)
        if not self.backend.mapped:
            self.backend.create_mappings()
        self.metadata = self.backend.metadata

    def update_mappings(self, X):
        """Update existing mappings with new data."""
        self.backend.update_mappings(X)

    def get_update_metadata(self, new_file_name=None):
        metadata = self.backend.get_update_metadata(new_file_name)
        metadata["backend"] = self.backend_type
        return metadata

    def get_indexes(self, X, type_of="t", order="raw2ind"):
        """Converts raw data to an indexed form or vice versa according to previously created mappings.

        Parameters
        ----------
        X: array
             Array with raw or indexed data.
        type_of: str
             Type of provided sample to be specified as one of the following values: `{"t", "e", "r"}`.
             It indicates whether the provided sample is an array of triples (`"t"`), a list of entities (`"e"`)
             or a list of relations (`"r"`).
        order: str
             It specifies whether it converts raw data to indexes (``order="raw2ind"``) or indexes to raw
             data (``order="ind2raw"``)

        Returns
        -------
        Y: array
             Array of the same size as `sample` but with indexes of elements instead of raw data or raw data instead
             of indexes.
        """
        return self.backend.get_indexes(X, type_of=type_of, order=order)

    def get_relations_count(self):
        """Get number of unique relations."""
        return self.backend.get_relations_count()

    def get_entities_count(self):
        """Get number of unique entities."""
        return self.backend.get_entities_count()

    def clean(self):
        """Remove persisted and in-memory objects."""
        return self.backend.clean()

    def get_entities_in_batches(self, batch_size=-1, random=False, seed=None):
        """Generator that retrieves entities and return them in batches.

        Parameters
        ----------
        batch_size: int
             Size of array that the batch should have, :math:`-1` when the whole dataset is required.
        random: bool
             Whether to return elements of batch in a random order (default: `False`).
        seed: int
             Used with ``random=True``, seed for repeatability of experiments.

        Yields
        ------
        Batch: numppy array
             Batch of data of size (batch_size, 3).

        """
        ents_len = self.get_entities_count()
        if batch_size == -1:
            batch_size = ents_len
        entities = list(range(0, ents_len, batch_size))
        for start_index in entities:
            if start_index + batch_size >= ents_len:
                batch_size = ents_len - start_index
            ents = list(range(start_index, start_index + batch_size))
            if random:
                np.random.seed(seed)
                np.random.shuffle(ents)
            yield np.array(ents)


def register_indexer_backend(name):
    """Decorator responsible for registering partition in the partition registry.

    Parameters
    ----------
    name: str
         Name of the new backend.

    Example
    -------
    >>>@register_indexer_backend("NewBackendName")
    >>>class NewBackend():
    >>>... pass
    """

    def insert_in_registry(class_handle):
        """Checks if backend already exists and if not registers it."""
        if name in INDEXER_BACKEND_REGISTRY.keys():
            msg = "Indexer backend with name {} already exists!".format(name)
            logger.error(msg)
            raise Exception(msg)

        INDEXER_BACKEND_REGISTRY[name] = class_handle
        class_handle.name = name

        return class_handle

    return insert_in_registry


@register_indexer_backend("in_memory")
class InMemory:
    def __init__(
        self,
        data,
        entities_dict=None,
        reversed_entities_dict=None,
        relations_dict=None,
        reversed_relations_dict=None,
        root_directory=tempfile.gettempdir(),
        name="main_partition",
        **kwargs
    ):
        """Initialise backend by creating mappings.

        Parameters
        ----------
        data: array
             Data to be indexed.
        entities_dict: dict or shelve path
             Dictionary or shelve path storing entities mappings; if not provided, it is created from data.
        reversed_entities_dict: dictionary or shelve path
             Dictionary or shelve path storing reversed entities mappings; if not provided, it is created from data.
        relations_dict: dictionary or shelve path
             Dictionary or shelve path storing relations mappings; if not provided, it is created from data.
        reversed_relations_dict: dictionary or shelve path
             Dictionary or shelve path storing reversed relations mappings; if not provided, it is created from data.
        root_directory: str
             Path of the directory where to store persistent mappings.
        """
        self.data = data
        self.mapped = False
        self.metadata = {}
        # ent to idx dict
        self.entities_dict = entities_dict
        self.reversed_entities_dict = reversed_entities_dict
        # rel to idx dict
        self.relations_dict = relations_dict
        self.reversed_relations_dict = reversed_relations_dict

        self.root_directory = root_directory
        self.name = name

        self.max_ents_index = -1
        self.max_rels_index = -1
        self.ents_length = 0
        self.rev_ents_length = 0
        self.rels_length = 0
        self.rev_rels_length = 0

    def get_all_entities(self):
        """Returns all the (raw) entities in the dataset"""
        return list(self.entities_dict.values())

    def get_all_relations(self):
        """Returns all the (raw) relations in the dataset"""
        return list(self.relations_dict.values())

    def create_mappings(self):
        """Create mappings of data into indexes.

        It creates four dictionaries with keys as unique entities/relations and values as indexes and reversed
        version of it. Dispatches to the adequate functions to create persistent or in-memory dictionaries.
        """

        if (
            isinstance(self.entities_dict, dict)
            and isinstance(self.reversed_entities_dict, dict)
            and isinstance(self.relations_dict, dict)
            and isinstance(self.reversed_relations_dict, dict)
        ):
            self._update_properties()
            logger.debug(
                "The mappings initialised from in-memory dictionaries."
            )
        elif (
            self.entities_dict is None
            and self.reversed_entities_dict is None
            and self.relations_dict is None
            and self.reversed_relations_dict is None
        ):
            logger.debug(
                "The mappings will be created for data in {}.".format(
                    self.name
                )
            )

            if isinstance(self.data, np.ndarray):
                self.update_dictionary_mappings()
            else:
                self.update_dictionary_mappings_in_chunks()
        else:
            logger.debug(
                "Provided initialization objects are not supported. Can't Initialise mappings."
            )
        self.mapped = True

    def _update_properties(self):
        """Initialise properties from the in-memory dictionary."""
        self.max_ents_index = self._get_max_ents_index()
        self.max_rels_index = self._get_max_rels_index()

        self.ents_length = len(self.entities_dict)
        self.rev_ents_length = len(self.reversed_entities_dict)
        self.rels_length = len(self.relations_dict)
        self.rev_rels_length = len(self.reversed_relations_dict)

    def _get_max_ents_index(self):
        """Get maximum index from entities dictionary."""
        return max(self.reversed_entities_dict.values())

    def _get_max_rels_index(self):
        """Get maximum index from relations dictionary."""
        return max(self.reversed_relations_dict.values())

    def _get_starting_index_ents(self):
        """Returns next index to continue adding elements to entities dictionary."""
        if not self.entities_dict:
            self.entities_dict = {}
            self.reversed_entities_dict = {}
            return 0
        else:
            return self.max_ents_index + 1

    def _get_starting_index_rels(self):
        """Returns next index to continue adding elements to relations dictionary."""
        if not self.relations_dict:
            self.relations_dict = {}
            self.reversed_relations_dict = {}
            return 0
        else:
            return self.max_rels_index + 1

    def update_mappings(self, new_data):
        """Update existing mappings with new data."""
        self.update_dictionary_mappings(new_data)

    def get_update_metadata(self, new_file_name=None):
        metadata = {
            "entities_dict": self.entities_dict,
            "reversed_entities_dict": self.reversed_entities_dict,
            "relations_dict": self.relations_dict,
            "reversed_relations_dict": self.reversed_relations_dict,
        }
        return metadata

    def update_dictionary_mappings(self, sample=None):
        """Index entities and relations.

        Creates shelves for mappings between entities and relations to indexes and reverse mapping.
        Remember to use mappings for entities with entities and relations with relations!
        """
        if sample is None:
            sample = self.data
        # logger.debug(sample)
        i = self._get_starting_index_ents()
        j = self._get_starting_index_rels()

        for d in sample:
            if d[0] not in self.reversed_entities_dict:
                self.reversed_entities_dict[d[0]] = i
                self.entities_dict[i] = d[0]
                i += 1
            if d[2] not in self.reversed_entities_dict:
                self.reversed_entities_dict[d[2]] = i
                self.entities_dict[i] = d[2]
                i += 1
            if d[1] not in self.reversed_relations_dict:
                self.reversed_relations_dict[d[1]] = j
                self.relations_dict[j] = d[1]
                j += 1

        self.max_ents_index = i - 1
        self.max_rels_index = j - 1

        self.ents_length = len(self.entities_dict)
        self.rev_ents_length = len(self.reversed_entities_dict)
        self.rels_length = len(self.relations_dict)
        self.rev_rels_length = len(self.reversed_relations_dict)

        if self.rev_ents_length != self.ents_length:
            msg = "Reversed entities index size not equal to index size ({} and {})".format(
                self.rev_ents_length, self.ents_length
            )
            logger.error(msg)
            raise Exception(msg)

        if self.rev_rels_length != self.rels_length:
            msg = "Reversed relations index size not equal to index size ({} and {})".format(
                self.rev_rels_length, self.rels_length
            )
            logger.error(msg)
            raise Exception(msg)

        logger.debug(
            "Mappings updated with: {} ents, {} rev_ents, {} rels and {} rev_rels".format(
                self.ents_length,
                self.rev_ents_length,
                self.rels_length,
                self.rev_rels_length,
            )
        )

    def update_dictionary_mappings_in_chunks(self):
        """Update dictionary mappings chunk by chunk."""
        for chunk in self.data:
            if isinstance(chunk, np.ndarray):
                self.update_dictionary_mappings(chunk)
            else:
                self.update_dictionary_mappings(chunk.values)

    def get_indexes(self, sample=None, type_of="t", order="raw2ind"):
        """Converts raw data to an indexed form or vice versa according to previously created mappings.

        Parameters
        ----------
        sample: array
             Array with raw or indexed data.
        type_of: str
             Type of provided sample to be specified as one of the following values: `{"t", "e", "r"}`.
             It indicates whether the provided sample is an array of triples (`"t"`), a list of entities (`"e"`)
             or a list of relations (`"r"`).
        order: str
             It specifies whether it converts raw data to indexes (``order="raw2ind"``) or indexes to raw
             data (``order="ind2raw"``)

        Returns
        -------
        Array: array
             Array of the same size as `sample` but with indexes of elements instead of raw data or raw data instead
             of indexes.
        """
        if type_of not in ["t", "e", "r"]:
            msg = "Type (type_of) should be one of the following: t, e, r, instead got {}".format(
                type_of
            )
            logger.error(msg)
            raise Exception(msg)

        if type_of == "t":
            if isinstance(sample, pd.DataFrame):
                sample = sample.values
            self.data_shape = sample.shape[1]
            indexed_data = self.get_indexes_from_a_dictionary(
                sample[:, :3], order=order
            )
            # focusE
            if sample.shape[1] > 3:
                # weights = preprocess_focusE_weights(data=sample[:, :3], weights=sample[:, 3:])
                weights = sample[:, 3:]
                return np.concatenate([indexed_data, weights], axis=1)
            else:
                return indexed_data
        else:
            return self.get_indexes_from_a_dictionary_single(
                sample, type_of=type_of, order=order
            )

    def get_indexes_from_a_dictionary(self, sample, order="raw2ind"):
        """Get indexed triples from an in-memory dictionary.

        Parameters
        ----------
        sample: array
             Array with raw or indexed triples.
        order: str
             It specifies whether it converts raw data to indexes (``order="raw2ind"``) or indexes to raw
             data (``order="ind2raw"``)

        Returns
        -------
        Array: array
             Array of the same size as `sample` but with indexes of elements instead of raw data or raw data instead
             of indexes.
        """
        if order == "raw2ind":
            entities = self.reversed_entities_dict
            relations = self.reversed_relations_dict
            dtype = np.int32
        elif order == "ind2raw":
            entities = self.entities_dict
            relations = self.relations_dict
            dtype = str
        else:
            msg = "No such order available options: ind2raw, raw2ind, instead got {}.".format(
                order
            )
            logger.error(msg)
            raise Exception(msg)
        if entities is None and relations is None:
            msg = "Requested entities and relation mappings are empty."
            logger.error(msg)
            raise Exception(msg)

        subjects = []
        objects = []
        predicates = []

        invalid_keys = 0
        for row in sample:
            try:
                s = entities[row[0]]
                p = relations[row[1]]
                o = entities[row[2]]
                subjects.append(s)
                predicates.append(p)
                objects.append(o)
            except KeyError:
                invalid_keys += 1

        if invalid_keys > 0:
            print(
                "\n{} triples containing invalid keys skipped!".format(
                    invalid_keys
                )
            )

        subjects = np.array(subjects, dtype=dtype)
        objects = np.array(objects, dtype=dtype)
        predicates = np.array(predicates, dtype=dtype)

        merged = np.stack([subjects, predicates, objects], axis=1)
        return merged

    def get_indexes_from_a_dictionary_single(
        self, sample, type_of="e", order="raw2ind"
    ):
        """Get indexed elements (entities, relations) or raw data from an in-memory dictionary.

        Parameters
        ----------
        sample: array
             Array with raw or indexed data.
        type_of: str
             Type of provided sample to be specified as one of the following values: `{"t", "e", "r"}`.
             It indicates whether the provided sample is an array of triples (`"t"`), a list of entities (`"e"`)
             or a list of relations (`"r"`).
        order: str
             It specifies whether it converts raw data to indexes (``order="raw2ind"``) or indexes to raw
             data (``order="ind2raw"``)

        Returns
        -------
        tmp: array
             Array of the same size as `sample` but with indexes of elements instead of raw data or raw data instead
             of indexes.
        """
        if order == "raw2ind":
            entities = self.reversed_entities_dict
            relations = self.reversed_relations_dict
            dtype = np.int32
        elif order == "ind2raw":
            entities = self.entities_dict
            relations = self.relations_dict
            dtype = str
        else:
            msg = "No such order available options: ind2raw, raw2ind, instead got {}.".format(
                order
            )
            logger.error(msg)
            raise Exception(msg)

        if entities is None and relations is None:
            msg = "Requested entities and relations mappings are empty."
            logger.error(msg)
            raise Exception(msg)

        if type_of == "e":
            elements = np.array([entities[x] for x in sample], dtype=dtype)
            return elements
        elif type_of == "r":
            elements = np.array([relations[x] for x in sample], dtype=dtype)
            return elements
        else:
            if type_of not in ["r", "e"]:
                msg = "No such option, should be r (relations) or e (entities), instead got {}".format(
                    type_of
                )
                logger.error(msg)
                raise Exception(msg)

    def get_relations_count(self):
        """Get number of unique relations."""
        return len(self.relations_dict)

    def get_entities_count(self):
        """Get number of unique entities."""
        return len(self.entities_dict)

    def clean(self):
        """Remove stored objects."""
        del self.entities_dict
        del self.reversed_entities_dict
        del self.relations_dict
        del self.reversed_relations_dict


@register_indexer_backend("shelves")
class Shelves:
    def __init__(
        self,
        data,
        entities_dict=None,
        reversed_entities_dict=None,
        relations_dict=None,
        reversed_relations_dict=None,
        root_directory=tempfile.gettempdir(),
        name="main_partition",
        **kwargs
    ):
        """Initialise backend by creating mappings.

        Parameters
        ----------
        data: array
             Data to be indexed.
        entities_dict: dict or shelve path
             Dictionary or shelve path storing entities mappings; if not provided, it is created from data.
        reversed_entities_dict: dictionary or shelve path
             Dictionary or shelve path storing reversed entities mappings; if not provided, it is created from data.
        relations_dict: dictionary or shelve path
             Dictionary or shelve path storing relations mappings; if not provided, it is created from data.
        reversed_relations_dict: dictionary or shelve path
             Dictionary or shelve path storing reversed relations mappings; if not provided, it is created from data.
        root_directory: str
             Path of the directory where to store persistent mappings.
        """
        self.data = data
        self.mapped = False
        self.metadata = {}
        self.entities_dict = entities_dict
        self.reversed_entities_dict = reversed_entities_dict
        self.relations_dict = relations_dict
        self.reversed_relations_dict = reversed_relations_dict

        self.root_directory = root_directory
        self.name = name

        self.max_ents_index = -1
        self.max_rels_index = -1
        self.ents_length = 0
        self.rev_ents_length = 0
        self.rels_length = 0
        self.rev_rels_length = 0

    def get_all_entities(self):
        """Returns all the (raw) entities in the dataset."""
        return list(self.entities_dict.values())

    def get_all_relations(self):
        """Returns all the (raw) relations in the dataset."""
        return list(self.relations_dict.values())

    def get_update_metadata(self, new_file_name=None):
        """Update dataset metadata."""
        metadata = {
            "entities_dict": self.entities_dict,
            "reversed_entities_dict": self.reversed_entities_dict,
            "relations_dict": self.relations_dict,
            "reversed_relations_dict": self.reversed_relations_dict,
        }
        return metadata

    def create_mappings(self):
        """Creates mappings of data into indexes.

        It creates four dictionaries: two having as keys the unique entities/relations and as values the indexes,
        while the other two are the reversed version of previous.
        This method also dispatches to the adequate functions to create persistent or in-memory dictionaries.
        """

        if (
            isinstance(self.entities_dict, str)
            and self.shelve_exists(self.entities_dict)
            and isinstance(self.reversed_entities_dict, str)
            and self.shelve_exists(self.reversed_entities_dict)
            and isinstance(self.relations_dict, str)
            and self.shelve_exists(self.relations_dict)
            and isinstance(self.reversed_relations_dict, str)
            and self.shelve_exists(self.reversed_relations_dict)
        ):
            self._update_properties()
            logger.debug(
                "The mappings initialised from persistent dictionaries (shelves)."
            )
        elif (
            self.entities_dict is None
            and self.reversed_entities_dict is None
            and self.relations_dict is None
            and self.reversed_relations_dict is None
        ):
            logger.debug(
                "The mappings will be created for data in {}.".format(
                    self.name
                )
            )

            if isinstance(self.data, np.ndarray):
                self.create_persistent_mappings_from_nparray()
            else:
                self.create_persistent_mappings_in_chunks()
        else:
            logger.debug(
                "Provided initialization objects are not supported. Can't Initialise mappings."
            )
        self.mapped = True

    def create_persistent_mappings_in_chunks(self):
        """Creates shelves for mappings from entities and relations to indexes and the reverse mappings.

        Four shelves are created in root_directory:
        - entities_<NAME>_<DATE>.shf - with map entities -> indexes
        - reversed_entities_<NAME>_<DATE>.shf - with map indexes -> entities
        - relations_<NAME>_<DATE>.shf - with map relations -> indexes
        - reversed_relations_<NAME>_<DATE>.shf - with map indexes -> relations

        Remember to use mappings for entities with entities and relations with relations!
        """
        date = datetime.now().strftime("%d-%m-%Y_%I-%M-%S_%f_%p")
        self.entities_dict = os.path.join(
            self.root_directory, "entities_{}_{}.shf".format(self.name, date)
        )
        self.reversed_entities_dict = os.path.join(
            self.root_directory,
            "reversed_entities_{}_{}.shf".format(self.name, date),
        )
        self.relations_dict = os.path.join(
            self.root_directory, "relations_{}_{}.shf".format(self.name, date)
        )
        self.reversed_relations_dict = os.path.join(
            self.root_directory,
            "reversed_relations_{}_{}.shf".format(self.name, date),
        )

        for chunk in self.data:
            if isinstance(chunk, pd.DataFrame):
                self.update_shelves(chunk.iloc[:, :3].values, rough=True)
            else:
                self.update_shelves(chunk[:, :3], rough=True)

        logger.debug(
            "We need to reindex all the data now so the indexes are continuous among chunks"
        )
        self.reindex()

        self.files_id = "_{}_{}.shf".format(self.name, date)
        files = [
            "entities",
            "reversed_entities",
            "relations",
            "reversed_relations",
        ]
        logger.debug(
            "Mappings are created in the following files:\n{}\n{}\n{}\n{}".format(
                *[x + self.files_id for x in files]
            )
        )
        self.metadata.update(
            {
                "entities_shelf": self.entities_dict,
                "reversed_entities_shelf": self.reversed_entities_dict,
                "relations": self.relations_dict,
                "reversed_relations_dict": self.reversed_relations_dict,
                "name": self.name,
            }
        )

    def reindex(self):
        """Reindex the data to continuous values from 0 to <MAX UNIQUE ENTITIES/RELATIONS>.

        This is needed where data is provided in chunks as we do not know the overlap
        between chunks upfront and indexes are not continuous.
        This guarantees that entities and relations have continuous indexes.
        """
        logger.debug("starting reindexing...")
        remapped_ents_file = "remapped_ents.shf"
        remapped_rev_ents_file = "remapped_rev_ents.shf"
        remapped_rels_file = "remapped_rels.shf"
        remapped_rev_rels_file = "remapped_rev_rels.shf"
        with shelve.open(self.reversed_entities_dict) as ents:
            with shelve.open(
                remapped_ents_file, writeback=True
            ) as remapped_ents:
                with shelve.open(
                    remapped_rev_ents_file, writeback=True
                ) as remapped_rev_ents:
                    for i, ent in enumerate(ents):
                        remapped_ents[str(i)] = str(ent)
                        remapped_rev_ents[str(ent)] = str(i)

        with shelve.open(self.reversed_relations_dict) as rels:
            with shelve.open(
                remapped_rels_file, writeback=True
            ) as remapped_rels:
                with shelve.open(
                    remapped_rev_rels_file, writeback=True
                ) as remapped_rev_rels:
                    for i, rel in enumerate(rels):
                        remapped_rels[str(i)] = str(rel)
                        remapped_rev_rels[str(rel)] = str(i)

        self.move_shelve(remapped_ents_file, self.entities_dict)
        self.move_shelve(remapped_rev_ents_file, self.reversed_entities_dict)
        self.move_shelve(remapped_rels_file, self.relations_dict)
        self.move_shelve(remapped_rev_rels_file, self.reversed_relations_dict)
        logger.debug("reindexing done!")
        self._update_properties()
        logger.debug("properties updated")

    def _update_properties(self, rough=False):
        """Initialise properties from the persistent dictionary (shelve)."""

        with shelve.open(self.entities_dict) as ents:
            self.max_ents_index = int(max(ents.keys(), key=lambda x: int(x)))
            self.ents_length = len(ents)
        with shelve.open(self.relations_dict) as rels:
            self.max_rels_index = int(max(rels.keys(), key=lambda x: int(x)))
            self.rels_length = len(rels)
        with shelve.open(self.reversed_entities_dict) as ents:
            self.rev_ents_length = len(ents)
        with shelve.open(self.reversed_relations_dict) as rels:
            self.rev_rels_length = len(rels)
        if not rough:
            if not self.rev_ents_length == self.ents_length:
                msg = "Reversed entities index size not equal to index size ({} and {})".format(
                    self.rev_ents_length, self.ents_length
                )
                logger.error(msg)
                raise Exception(msg)
            if not self.rev_rels_length == self.rels_length:
                msg = "Reversed relations index size not equal to index size ({} and {})".format(
                    self.rev_rels_length, self.rels_length
                )
                logger.error(msg)
                raise Exception(msg)
        else:
            logger.debug(
                "In a rough mode, the sizes may not be equal due to duplicates, \
            it will be fixed in reindexing at the later stage."
            )
            logger.debug(
                "Reversed entities index size and index size {} and {}".format(
                    self.rev_ents_length, self.ents_length
                )
            )
            logger.debug(
                "Reversed relations index size and index size: {} and {}".format(
                    self.rev_rels_length, self.rels_length
                )
            )

    def create_persistent_mappings_from_nparray(self):
        """Creates shelves for mappings from entities and relations to indexes and the reverse mappings.

        Four shelves are created in root_directory:
        - entities_<NAME>_<DATE>.shf - with map entities -> indexes
        - reversed_entities_<NAME>_<DATE>.shf - with map indexes -> entities
        - relations_<NAME>_<DATE>.shf - with map relations -> indexes
        - reversed_relations_<NAME>_<DATE>.shf - with map indexes -> relations

        Remember to use mappings for entities with entities and relations with relations!
        """

        date = datetime.now().strftime("%d-%m-%Y_%I-%M-%S_%f_%p")
        self.entities_dict = os.path.join(
            self.root_directory, "entities_{}_{}.shf".format(self.name, date)
        )
        self.reversed_entities_dict = os.path.join(
            self.root_directory,
            "reversed_entities_{}_{}.shf".format(self.name, date),
        )
        self.relations_dict = os.path.join(
            self.root_directory, "relations_{}_{}.shf".format(self.name, date)
        )
        self.reversed_relations_dict = os.path.join(
            self.root_directory,
            "reversed_relations_{}_{}.shf".format(self.name, date),
        )
        self.files_id = "_{}_{}.shf".format(self.name, date)
        files = [
            "entities",
            "reversed_entities",
            "relations",
            "reversed_relations",
        ]
        logger.debug(
            "Mappings are created in the following files:\n{}\n{}\n{}\n{}".format(
                *[x + self.files_id for x in files]
            )
        )
        self.metadata.update(
            {
                "entities_shelf": self.entities_dict,
                "reversed_entities_shelf": self.reversed_entities_dict,
                "relations": self.relations_dict,
                "reversed_relations_dict": self.reversed_relations_dict,
                "name": self.name,
            }
        )
        self.update_shelves()

    def update_shelves(self, sample=None, rough=False):
        """Update shelves with sample or full data when sample not provided."""
        if sample is None:
            sample = self.data
            if (
                self.shelve_exists(self.entities_dict)
                or self.shelve_exists(self.reversed_entities_dict)
                or self.shelve_exists(self.relations_dict)
                or self.shelve_exists(self.reversed_relations_dict)
            ):
                msg = "Shelves exists for some reason and are not empty!"
                logger.error(msg)
                raise Exception(msg)

        logger.debug("Sample: {}".format(sample))
        entities = set(sample[:, 0]).union(set(sample[:, 2]))
        predicates = set(sample[:, 1])

        start_ents = self._get_starting_index_ents()
        logger.debug("Start index entities: {}".format(start_ents))
        new_indexes_ents = range(start_ents, start_ents + len(entities))
        # maximum new index, usually less when multiple chunks provided due to
        # chunks
        if not len(new_indexes_ents) == len(entities):
            msg = "Etimated indexes length for entities not equal to entities length ({} and {})".format(
                len(new_indexes_ents), len(entities)
            )
            logger.error(msg)
            raise Exception(msg)

        start_rels = self._get_starting_index_rels()
        new_indexes_rels = range(start_rels, start_rels + len(predicates))
        logger.debug("Starts index relations: {}".format(start_rels))
        if not len(new_indexes_rels) == len(predicates):
            msg = "Estimated indexes length for relations not equal to relations length ({} and {})".format(
                len(new_indexes_rels), len(predicates)
            )
            logger.error(msg)
            raise Exception(msg)
        # print("new indexes rels: ", new_indexes_rels)
        logger.debug(
            "index rels size: {} and rels size: {}".format(
                len(new_indexes_rels), len(predicates)
            )
        )
        logger.debug(
            "index ents size: {} and entss size: {}".format(
                len(new_indexes_ents), len(entities)
            )
        )

        with shelve.open(self.entities_dict, writeback=True) as ents:
            with shelve.open(
                self.reversed_entities_dict, writeback=True
            ) as reverse_ents:
                with shelve.open(self.relations_dict, writeback=True) as rels:
                    with shelve.open(
                        self.reversed_relations_dict, writeback=True
                    ) as reverse_rels:
                        reverse_ents.update(
                            {
                                str(value): str(key)
                                for key, value in zip(
                                    new_indexes_ents, entities
                                )
                            }
                        )
                        ents.update(
                            {
                                str(key): str(value)
                                for key, value in zip(
                                    new_indexes_ents, entities
                                )
                            }
                        )
                        reverse_rels.update(
                            {
                                str(value): str(key)
                                for key, value in zip(
                                    new_indexes_rels, predicates
                                )
                            }
                        )
                        rels.update(
                            {
                                str(key): str(value)
                                for key, value in zip(
                                    new_indexes_rels, predicates
                                )
                            }
                        )
        self._update_properties(rough=rough)

    def shelve_exists(self, name):
        """Check if shelve with a given name exists."""
        if not os.path.isfile(name + ".bak"):
            return False
        if not os.path.isfile(name + ".dat"):
            return False
        if not os.path.isfile(name + ".dir"):
            return False
        return True

    def remove_shelve(self, name):
        """Remove shelve with a given name."""
        try:
            os.remove(name + ".bak")
            os.remove(name + ".dat")
            os.remove(name + ".dir")
        except Exception:
            if os.path.exists(name + ".db"):
                os.remove(name + ".db")

    def move_shelve(self, source, destination):
        """Move shelve to a different file."""
        try:
            os.rename(source + ".dir", destination + ".dir")
            os.rename(source + ".dat", destination + ".dat")
            os.rename(source + ".bak", destination + ".bak")
        except Exception:
            os.rename(source + ".db", destination + ".db")

    def _get_starting_index_ents(self):
        """Returns next index to continue adding elements to the entities dictionary."""
        if not self.entities_dict:
            return 0
        else:
            return self.max_ents_index + 1

    def _get_starting_index_rels(self):
        """Returns next index to continue adding elements to the relations dictionary."""
        if not self.relations_dict:
            return 0
        else:
            return self.max_rels_index + 1

    def _get_max_ents_index(self):
        """Get maximum index from entities dictionary."""
        with shelve.open(self.entities_dict) as ents:
            return int(max(ents.keys(), key=lambda x: int(x)))

    def _get_max_rels_index(self):
        """Get maximum index from relations dictionary."""
        with shelve.open(self.relations_dict) as rels:
            return int(max(rels.keys(), key=lambda x: int(x)))

    def update_mappings(self, new_data):
        self.update_shelves(new_data, rough=True)
        self.reindex()

    def get_indexes(self, sample=None, type_of="t", order="raw2ind"):
        """Converts raw data to an indexed form or vice versa according to previously created mappings.

        Parameters
        ----------
        sample: array
             Array with raw or indexed data.
        type_of: str
             Type of provided sample to be specified as one of the following values: `{"t", "e", "r"}`.
             It indicates whether the provided sample is an array of triples (`"t"`), a list of entities (`"e"`)
             or a list of relations (`"r"`).
        order: str
             It specifies whether it converts raw data to indexes (``order="raw2ind"``) or indexes to raw
             data (``order="ind2raw"``)

        Returns
        -------
        tmp: array
             Array of the same size as `sample` but with indexes of elements instead of raw data or raw data instead
             of indexes.
        """
        if type_of not in ["t", "e", "r"]:
            msg = "Type (type_of) should be one of the following: t, e, r, instead got {}".format(
                type_of
            )
            logger.error(msg)
            raise Exception(msg)

        if type_of == "t":
            self.data_shape = sample.shape[1]
            indexed_data = self.get_indexes_from_shelves(
                sample[:, :3], order=order
            )
            if sample.shape[1] > 3:
                weights = sample[:, 3:]
                # weights = preprocess_focusE_weights(data=sample[:, :3], weights=sample[:, 3:])
                return np.concatenate([indexed_data, weights], axis=1)
            return indexed_data
        else:
            return self.get_indexes_from_shelves_single(
                sample, type_of=type_of, order=order
            )

    def get_indexes_from_shelves(self, sample, order="raw2ind"):
        """Get indexed triples or raw data from shelves.

        Parameters
        ----------
        sample: array
             Array with a fragment of data of size (N,3), where each element is either (subject, predicate, object)
             or (indexes_subject, indexed_predicate, indexed_object).
        order: str
             Specify ``order="raw2ind"`` or ``order="ind2raw"`` whether to convert raw data to indexes or indexes
             to raw data.

        Returns
        -------
        tmp: array
             Array of size (N,3) where each element is either (indexes_subject, indexed_predicate, indexed_object)
             or (subject, predicate, object).
        """
        if isinstance(sample, pd.DataFrame):
            sample = sample.values
        # logger.debug(sample)
        if order == "raw2ind":
            entities = self.reversed_entities_dict
            relations = self.reversed_relations_dict
            dtype = int
        elif order == "ind2raw":
            entities = self.entities_dict
            relations = self.relations_dict
            dtype = str
        else:
            msg = "No such order available options: ind2raw, raw2ind, instead got {}.".format(
                order
            )
            logger.error(msg)
            raise Exception(msg)

        with shelve.open(entities) as ents:
            with shelve.open(relations) as rels:
                subjects = []
                objects = []
                predicates = []

                invalid_keys = 0
                for row in sample:
                    try:
                        s = ents[str(row[0])]
                        p = rels[str(row[1])]
                        o = ents[str(row[2])]
                        subjects.append(s)
                        predicates.append(p)
                        objects.append(o)
                    except KeyError:
                        invalid_keys += 1

                if invalid_keys > 0:
                    print(
                        "\n{} triples containing invalid keys skipped!".format(
                            invalid_keys
                        )
                    )

                out = np.array((subjects, predicates, objects), dtype=dtype).T
                return out

    def get_indexes_from_shelves_single(
        self, sample, type_of="e", order="raw2ind"
    ):
        """Get indexed elements or raw data from shelves for entities or relations.

        Parameters
        ----------
        sample: list
             List of entities or relations indexed or in raw format.
        type_of: str
             ``type_of="e"`` to get indexes/raw data for entities or ``type_of="r"`` to get indexes/raw data
             for relations.
        order: str
             ``order=raw2ind`` or ``order=ind2raw`` to specify whether to convert raw data to indexes or indexes
             to raw data.

        Returns
        -------
        tmp: array
             Array of the same size of sample with indexed or raw data.
        """
        if order == "raw2ind":
            entities = self.reversed_entities_dict
            relations = self.reversed_relations_dict
            dtype = int
        elif order == "ind2raw":
            entities = self.entities_dict
            relations = self.relations_dict
            dtype = str
        else:
            msg = "No such order available options: ind2raw, raw2ind, instead got {}.".format(
                order
            )
            logger.error(msg)
            raise Exception(msg)

        if type_of == "e":
            with shelve.open(entities) as ents:
                elements = [ents[str(elem)] for elem in sample]
            return np.array(elements, dtype=dtype)
        elif type_of == "r":
            with shelve.open(relations) as rels:
                elements = [rels[str(elem)] for elem in sample]
            return np.array(elements, dtype=dtype)
        else:
            if type_of not in ["r", "e"]:
                msg = "No such option, should be r (relations) or e (entities), instead got {}".format(
                    type_of
                )
                logger.error(msg)
                raise Exception(msg)

    def get_relations_count(self):
        """Get number of unique relations."""
        return self.rels_length

    def get_entities_count(self):
        """Get number of unique entities."""
        return self.ents_length

    def clean(self):
        """Remove persisted objects."""
        self.remove_shelve(self.entities_dict)
        self.remove_shelve(self.reversed_entities_dict)
        self.remove_shelve(self.relations_dict)
        self.remove_shelve(self.reversed_relations_dict)


@register_indexer_backend("sqlite")
class SQLite:
    def __init__(
        self,
        data,
        db_file=None,
        root_directory=None,
        name="main_partition",
        **kwargs
    ):
        """Initialise backend by creating mappings.

        Parameters
        ----------
        data: data to be indexed.
        root_directory: directory where to store persistent mappings.
        """
        logger.debug("Initialisation of SQLite indexer.")
        self.data = data
        self.metadata = {}
        if root_directory is None:
            self.root_directory = tempfile.gettempdir()
        else:
            self.root_directory = root_directory
        if db_file is not None:
            self.db_file = os.path.join(self.root_directory, db_file)
            self.mapped = True
        else:
            date = datetime.now().strftime("%d-%m-%Y_%I-%M-%S_%f_%p")
            self.db_file = os.path.join(
                self.root_directory, name + date + str(uuid.uuid4()) + ".db"
            )

            if os.path.exists(self.db_file):
                os.remove(self.db_file)
            self.mapped = False
        self.name = name

        self.max_ents_index = -1
        self.max_rels_index = -1
        self.ents_length = 0
        self.rels_length = 0

    def get_all_entities(self):
        """Returns all the (raw) entities in the dataset."""

        query = "select distinct name from entities"
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            output = None
            try:
                cursor.execute(query)
                output = cursor.fetchall()
                out_val = []
                for out in output:
                    out_val.append(out[0])
                conn.commit()
            except Exception as e:
                logger.debug("Query failed. The error '{}' occurred".format(e))
                logger.debug(query)
                logger.debug(output)
                return []

        return out_val

    def get_all_relations(self):
        """Returns all the (raw) relations in the dataset."""
        query = "select distinct name from relations"
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            output = None
            try:
                cursor.execute(query)
                output = cursor.fetchall()
                out_val = []
                for out in output:
                    out_val.append(out[0])
                conn.commit()
            except Exception as e:
                logger.debug("Query failed. The error '{}' occurred".format(e))
                logger.debug(query)
                logger.debug(output)
                return []

        return out_val

    def get_update_metadata(self, path):
        """Get the metadata update for the database."""
        self.root_directory = path
        self.root_directory = (
            "." if self.root_directory == "" else self.root_directory
        )
        new_file_name = os.path.join(
            self.root_directory, os.path.basename(self.db_file)
        )
        if not os.path.exists(new_file_name):
            shutil.copyfile(self.db_file, new_file_name)
        self.db_file = new_file_name
        metadata = {
            "root_directory": self.root_directory,
            "db_file": os.path.basename(self.db_file),
            "name": self.name,
        }
        return metadata

    def create_mappings(self):
        """Creates SQLite mappings."""
        logger.debug("Creating SQLite mappings.")
        if isinstance(self.data, np.ndarray):
            self.create_persistent_mappings_from_nparray()
        else:
            self.create_persistent_mappings_in_chunks()
        logger.debug("Database: {}.".format(self.db_file))
        self.metadata.update({"db": self.db_file, "name": self.name})
        self.mapped = True

    def update_db(self, sample=None):
        """Update database with sample or full data when sample not provided."""
        logger.debug("Update db with data.")
        if sample is None:
            sample = self.data
        logger.debug("sample = {}".format(sample))
        subjects = sample[:, 0]
        objects = sample[:, 2]
        relations = sample[:, 1]
        entities = np.concatenate((subjects, objects))

        data = {"entities": entities, "relations": relations}
        for table, elems in data.items():
            sql_create_table = """ CREATE TABLE IF NOT EXISTS tmp_{} (
                                                name text PRIMARY KEY
                                            );""".format(
                table
            )

            with sqlite3.connect(self.db_file) as conn:
                c = conn.cursor()
                c.execute(sql_create_table)
                conn.commit()

            tab = "tmp_{}"
            values_placeholder = "({})".format(", ".join(["?"] * 1))
            query = "INSERT OR IGNORE INTO {} VALUES {};".format(
                tab.format(table), values_placeholder
            )
            with sqlite3.connect(self.db_file) as conn:
                c = conn.cursor()
                tmp = [(str(v),) for v in elems]
                c.executemany(query, tmp)
                conn.commit()

    def _get_max(self, table):
        """Get the max value out of a table."""
        logger.debug("Get max.")
        query = "SELECT max(id) from {};".format(table)
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            maxi = None
            try:
                cursor.execute(query)
                maxi = cursor.fetchall()
                conn.commit()
            except Exception as e:
                logger.debug("Query failed. The error '{}' occurred".format(e))

        if maxi is None:
            logger.debug("Table is empty or not such table exists.")
            return maxi
        elif not isinstance(maxi, list) or not isinstance(maxi[0], tuple):
            raise ValueError(
                "Cannot get max for the table with provided condition."
            )
        logger.debug("Maximal value: {}.".format(maxi[0][0]))
        return maxi[0][0]

    def _get_max_ents_index(self):
        """Get the max index for entities."""
        return self._get_max("entities")

    def _get_max_rels_index(self):
        """Get the max index for relations."""
        return self._get_max("relations")

    def _update_properties(self):
        """Initialise properties from the database."""
        logger.debug("Update properties")
        self.max_ents_index = self._get_max_ents_index()
        self.max_rels_index = self._get_max_rels_index()

        self.ents_length = self.get_entities_count()
        self.rels_length = self.get_relations_count()

    def create_persistent_mappings_from_nparray(self):
        """Index entities and relations.

        Creates sqlite db for mappings between entities and relations to indexes.
        """
        self.update_db()
        self.index_data("entities")
        self.index_data("relations")

    def index_data(self, table):
        """Create new table with persisted id of elements."""
        logger.debug("Index data in SQLite.")
        query = [
            "CREATE TABLE IF NOT EXISTS {0}(id INTEGER PRIMARY KEY, name TEXT NOT NULL);".format(
                table
            ),
            "INSERT INTO {0}(id, name) SELECT rowid - 1, name FROM tmp_{0};".format(
                table
            ),
            "DROP TABLE tmp_{0};".format(table),
        ]
        with sqlite3.connect(self.db_file) as conn:
            c = conn.cursor()
            for q in query:
                c.execute(q)
                conn.commit()
        self._update_properties()

    def create_persistent_mappings_in_chunks(self):
        """Index entities and relations. Creates sqlite db for mappings between
        entities and relations to indexes in chunks.
        """
        for chunk in self.data:
            if isinstance(chunk, pd.DataFrame):
                self.update_db(sample=chunk.iloc[:, :3].values)
            else:
                self.update_db(chunk[:, :3])
        logger.debug(
            "We need to reindex all the data now so the indexes are continuous among chunks"
        )
        self.index_data("entities")
        self.index_data("relations")

    def update_mappings(self, new_data):
        raise NotImplementedError(
            "Updating existing mappings not supported, \
        try creating new mappings in chunks instead."
        )

    def get_indexes(self, sample=None, type_of="t", order="raw2ind"):
        """Converts raw data to an indexed form (or vice versa) according to previously created mappings.

        Parameters
        ----------
        sample: array
             Array with raw or indexed data.
        type_of: str
             Type of provided sample to be specified as one of the following values: `{"t", "e", "r"}`.
             It indicates whether the provided sample is an array of triples (`"t"`), a list of entities (`"e"`)
             or a list of relations (`"r"`).
        order: str
             It specifies whether it converts raw data to indexes (``order="raw2ind"``) or indexes to raw
             data (``order="ind2raw"``)

        Returns
        -------
        out: array
             Array of the same size as `sample` but with indexes of elements instead of raw data or raw data instead
             of indexes.
        """
        if type_of not in ["t", "e", "r"]:
            msg = "Type (type_of) should be one of the following: t, e, r, instead got {}".format(
                type_of
            )
            logger.error(msg)
            raise Exception(msg)

        if type_of == "t":
            self.data_shape = sample.shape[1]
            indexed_data = self.get_indexes_from_db(sample[:, :3], order=order)
            # focusE
            if sample.shape[1] > 3:
                weights = sample[:, 3:]
                return np.concatenate([indexed_data, weights], axis=1)
            return indexed_data
        else:
            out, _ = self.get_indexes_from_db_single(
                sample, type_of=type_of, order=order
            )
            return out

    def get_indexes_from_db(self, sample, order="raw2ind"):
        """Get indexed or raw triples from the database.

        Parameters
        ----------
        sample: ndarray
             Numpy array with a fragment of data of size (N,3), where each element is (subject, predicate, object)
             or (indexed_subject, indexed_predicate, indexed_object).
        order: str
              Specifies whether it should convert raw data to indexes (``order="raw2ind"``) or indexes
              to raw data (``order="ind2raw"``).

        Returns
        -------
        tmp: ndarray
             Numpy array of size (N,3) with indexed triples, where, depending on ``order``, each element is
             (indexed_subject, indexed_predicate, indexed_object) or (subject, predicate, object).
        """
        if isinstance(sample, pd.DataFrame):
            sample = sample.values

        subjects, subject_present = self.get_indexes_from_db_single(
            sample[:, 0], type_of="e", order=order
        )
        objects, objects_present = self.get_indexes_from_db_single(
            sample[:, 2], type_of="e", order=order
        )
        predicates, predicates_present = self.get_indexes_from_db_single(
            sample[:, 1], type_of="r", order=order
        )
        if order == "raw2ind":
            dtype = int
        elif order == "ind2raw":
            dtype = str
        else:
            msg = "No such order available options: ind2raw, raw2ind, instead got {}.".format(
                order
            )
            logger.error(msg)
            raise Exception(msg)

        present = (
            np.array(subject_present)
            & np.array(objects_present)
            & np.array(predicates_present)
        )
        out = np.array((subjects, predicates, objects), dtype=dtype).T
        before = out.shape[0]
        out = out[present]
        after = out.shape[0]

        if before - after > 0:
            print(
                "\n{} triples containing invalid keys skipped!".format(
                    before - after
                )
            )

        return out

    def get_indexes_from_db_single(self, sample, type_of="e", order="raw2ind"):
        """Get indexes or raw data from entities or relations.

        Parameters
        ----------
        sample: list
             List of entities or relations to get indexes for.
        type_of: str
             Specifies whether to get indexes/raw data for entities (``type_of="e"``) or relations (``type_of="r"``).
        order: str
             Specifies whether to convert raw data to indexes (``order="raw2ind"``) or indexes to raw
             data (``order="ind2raw"``).

        Returns
        -------
        tmp: list
             List of indexes/raw data.
        present: list
             List that specifies whether the mapping for the elements in `sample` were in the database (`True`) or
             not (:math:`-1`).

        """
        if type_of == "e":
            table = "entities"
        elif type_of == "r":
            table = "relations"
        else:
            msg = "No such option, should be r (relations) or e (entities), instead got {}".format(
                type_of
            )
            logger.error(msg)
            raise Exception(msg)

        if order == "raw2ind":
            query = "select name, ifnull(id, '-1') from {0} where name in ({1});".format(
                table, ",".join('"{}"'.format(v) for v in sample)
            )
            with sqlite3.connect(self.db_file) as conn:
                cursor = conn.cursor()
                output = None
                try:
                    cursor.execute(query)
                    output = dict(cursor.fetchall())
                    conn.commit()
                    out_values = []
                    present = []

                    for x in sample:
                        try:
                            out_values.append(output[str(x)])
                            present.append(True)
                        except KeyError:
                            out_values.append(str(-1))
                            present.append(False)

                    return out_values, present

                except Exception as e:
                    logger.debug(
                        "Query failed. The error '{}' occurred".format(e)
                    )
                    logger.debug(query)
                    logger.debug(output)
                    return []
        elif order == "ind2raw":
            query = "select * from {0} where id in ({1});".format(
                table, ",".join('"{}"'.format(v) for v in sample)
            )
            with sqlite3.connect(self.db_file) as conn:
                cursor = conn.cursor()
                output = None
                try:
                    cursor.execute(query)
                    output = dict(cursor.fetchall())
                    conn.commit()
                    out_values = []
                    present = []

                    for x in sample:
                        try:
                            out_values.append(output[x])
                            present.append(True)
                        except KeyError:
                            out_values.append(str(-1))
                            present.append(False)

                    return out_values, present

                except Exception as e:
                    logger.debug(
                        "Query failed. The error '{}' occurred".format(e)
                    )
                    return []
        else:
            msg = "No such order available options: ind2raw, raw2ind, instead got {}.".format(
                order
            )
            logger.error(msg)
            raise Exception(msg)

    def get_count(self, table, condition):
        """Return number of unique elements in a table according to condition."""
        logger.debug("Get count.")
        query = "SELECT count(*) from {} {};".format(table, condition)
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            count = None
            try:
                cursor.execute(query)
                count = cursor.fetchall()
                conn.commit()
            except Exception as e:
                logger.debug("Query failed. The error '{}' occurred".format(e))

        if count is None:
            logger.debug("Table is empty or not such table exists.")
            return count
        elif not isinstance(count, list) or not isinstance(count[0], tuple):
            raise ValueError(
                "Cannot get count for the table with provided condition."
            )
        logger.debug("Count is {}.".format(count[0][0]))
        return count[0][0]

    def get_relations_count(self, condition=""):
        """Return number of unique relations."""
        return self.get_count("relations", condition)

    def get_entities_count(self, condition=""):
        """Return number of unique entities."""
        return self.get_count("entities", condition)

    def _get_starting_index_ents(self):
        """Return next index to continue adding elements to entities dictionary."""
        if not self.db_file:
            return 0
        else:
            return self.max_ents_index + 1

    def _get_starting_index_rels(self):
        """Return next index to continue adding elements to relations dictionary."""
        if not self.db_file:
            return 0
        else:
            return self.max_rels_index + 1

    def clean(self):
        """Remove the database file."""
        os.remove(self.db_file)
        logger.debug("Indexer Database removed.")
