# Copyright 2019-2023 The AmpliGraph Authors. All Rights Reserved.
#
# This file is Licensed under the Apache License, Version 2.0.
# A copy of the Licence is available in LICENCE, or at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
import abc
import glob
import logging
import os
import shelve
import shutil
import tempfile
from datetime import datetime

import numpy as np
import tensorflow as tf

from .graph_partitioner import (
    PARTITION_ALGO_REGISTRY,
    AbstractGraphPartitioner,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

PARTITION_MANAGER_REGISTRY = {}


def register_partitioning_manager(name):
    """Decorator responsible for registering partition manager in the partition manager registry.

    Parameters
    ----------
    name: str
         Name of the new partition manager.

    Example
    -------
    >>>@register_partitioning_manager("NewManagerName")
    >>>class NewManagerName(PartitionDataManager):
    >>>... pass
    """

    def insert_in_registry(class_handle):
        """Checks if partition manager already exists and if not registers it."""
        if name in PARTITION_MANAGER_REGISTRY.keys():
            msg = "Partitioning Manager with name {} already exists!".format(
                name
            )
            logger.error(msg)
            raise Exception(msg)

        PARTITION_MANAGER_REGISTRY[name] = class_handle
        class_handle.name = name
        return class_handle

    return insert_in_registry


class PartitionDataManager(abc.ABC):
    def __init__(
        self,
        dataset_loader,
        model,
        strategy="Bucket",
        partitioner_k=3,
        root_directory=None,
        ent_map_fname=None,
        ent_meta_fname=None,
        rel_map_fname=None,
        rel_meta_fname=None,
    ):
        """Initializes the Partitioning Data Manager.

        Uses/Creates partitioner and generates and manages partition related parameters.
        This is the base class.

        Parameters
        ----------
        dataset_loader : AbstractGraphPartitioner or GraphDataLoader
            Either an instance of AbstractGraphPartitioner or GraphDataLoader.
        model : tf.keras.Model
            The model that is being trained.
        strategy : string
            Type of partitioning strategy to use.
        root_directory : str
            directory where the partition manager files will be stored.
        """
        self._model = model
        self.k = self._model.k
        self.internal_k = self._model.internal_k
        self.eta = self._model.eta
        self.partitioner_k = partitioner_k
        if (
            ent_map_fname is not None
            and ent_meta_fname is not None
            and rel_map_fname is not None
            and rel_meta_fname is not None
        ):
            self.root_directory = os.path.dirname(ent_map_fname)
            self.timestamp = os.path.basename(ent_map_fname).split("_")[-1][
                :-4
            ]
            self.ent_map_fname = ent_map_fname
            self.ent_meta_fname = ent_meta_fname
            self.rel_map_fname = rel_map_fname
            self.rel_meta_fname = rel_map_fname

        else:
            if root_directory is not None:
                self.root_directory = root_directory
            else:
                self.root_directory = tempfile.gettempdir()
            self.timestamp = datetime.now().strftime("%d-%m-%Y_%I-%M-%S_%f_%p")
            self.ent_map_fname = os.path.join(
                self.root_directory, "ent_partition_{}".format(self.timestamp)
            )
            self.ent_meta_fname = os.path.join(
                self.root_directory, "ent_metadata_{}".format(self.timestamp)
            )
            self.rel_map_fname = os.path.join(
                self.root_directory, "rel_partition_{}".format(self.timestamp)
            )
            self.rel_meta_fname = os.path.join(
                self.root_directory, "rel_metadata_{}".format(self.timestamp)
            )

        if isinstance(dataset_loader, AbstractGraphPartitioner):
            self.partitioner = dataset_loader
            self.partitioner_k = self.partitioner._k
        else:
            print("Partitioning may take a while...")
            self.partitioner = PARTITION_ALGO_REGISTRY.get(strategy)(
                dataset_loader, k=self.partitioner_k
            )

        self.num_ents = (
            self.partitioner._data.backend.mapper.backend.ents_length
        )
        self.num_rels = (
            self.partitioner._data.backend.mapper.backend.rels_length
        )
        self.max_ent_size = 0
        for i in range(len(self.partitioner.partitions)):
            self.max_ent_size = max(
                self.max_ent_size,
                self.partitioner.partitions[
                    i
                ].backend.mapper.backend.ents_length,
            )

        self._generate_partition_params()

    def _copy_files(self, base):
        for file in glob.glob(base + "*"):
            shutil.copy(file, self.root_directory)

    def get_update_metadata(self, filepath):
        self.root_directory = filepath
        self.root_directory = (
            "." if self.root_directory == "" else self.root_directory
        )

        # new_file_name = os.path.join(self.root_directory, '*{}.bak'.format(self.timestamp))
        try:
            self._copy_files(self.ent_map_fname)
            self._copy_files(self.ent_meta_fname)
            self._copy_files(self.rel_map_fname)
            self._copy_files(self.rel_meta_fname)
            self.ent_map_fname = os.path.join(
                self.root_directory, "ent_partition_{}".format(self.timestamp)
            )
            self.ent_meta_fname = os.path.join(
                self.root_directory, "ent_metadata_{}".format(self.timestamp)
            )
            self.rel_map_fname = os.path.join(
                self.root_directory, "rel_partition_{}".format(self.timestamp)
            )
            self.rel_meta_fname = os.path.join(
                self.root_directory, "rel_metadata_{}".format(self.timestamp)
            )
        except shutil.SameFileError:
            pass

        metadata = {
            "root_directory": self.root_directory,
            "partitioner_k": self.partitioner_k,
            "ent_map_fname": self.ent_map_fname,
            "ent_meta_fname": self.ent_meta_fname,
            "rel_map_fname": self.rel_map_fname,
            "rel_meta_fname": self.rel_meta_fname,
        }
        return metadata

    @property
    def max_entities(self):
        """Returns the maximum entity size that can occur in a partition."""
        return self.max_ent_size

    @property
    def max_relations(self):
        """Returns the maximum relation size that can occur in a partition."""
        return self.num_rels

    def _generate_partition_params(self):
        """Generates the metadata needed for persisting and loading partition embeddings and other parameters."""
        raise NotImplementedError("Abstract method not implemented")

    def _update_partion_embeddings(self, graph_data_loader, partition_number):
        """Persists the embeddings and other parameters after a partition is trained.

        Parameters
        ----------
        graph_data_loader : GraphDataLoader
            Data loader of the current partition that was trained.
        partition_number: int
            Partition number of the current partition that was trained.
        """
        raise NotImplementedError("Abstract method not implemented")

    def _change_partition(self, graph_data_loader, partition_number):
        """Gets a new partition to train and loads all the parameters of the partition.

        Parameters
        ----------
        graph_data_loader : GraphDataLoader
            Data loader of the next partition that will be trained.
        partition_number: int
            Partition number of the next partition that will be trained.
        """
        raise NotImplementedError("Abstract method not implemented")

    def data_generator(self):
        """Generates the data to be trained from the current partition.

        Once the partition data is exhausted, the current parameters are persisted; the partition is changed
        and the model is notified.

        Returns
        -------
        batch_data_from_current_partition: array of shape (n,3)
            A batch of triples from the current partition being trained.
        """
        for i, partition_data in enumerate(self.partitioner):
            # partition_data is an object of graph data loader
            # Perform tasks related to change of partition
            self._change_partition(partition_data, i)
            try:
                while True:
                    # generate data from the current partition
                    batch_data_from_current_partition = next(partition_data)
                    yield batch_data_from_current_partition

            except StopIteration:
                # No more data in current partition (parsed fully once), so the partition is trained
                # Hence persist the params related to the current partition.
                self._update_partion_embeddings(partition_data, i)

    def get_tf_generator(self):
        """Returns tensorflow data generator."""
        return tf.data.Dataset.from_generator(
            self.data_generator,
            output_types=tf.dtypes.int32,
            output_shapes=(None, 3),
        ).prefetch(0)

    def __iter__(self):
        """Function needed to be used as an iterator."""
        return self

    def __next__(self):
        """Function needed to be used as an iterator."""
        return next(self.batch_iterator)

    def reload(self):
        """Reload the data for the next epoch."""
        self.partitioner.reload()
        self.batch_iterator = iter(self.data_generator())

    def on_epoch_end(self):
        """Activities to be performed at the end of an epoch."""
        pass

    def on_complete(self):
        """Activities to be performed at the end of training.

        The manager persists the data (splits the entity partitions into individual embeddings).
        """
        pass


@register_partitioning_manager("GeneralPartitionDataManager")
class GeneralPartitionDataManager(PartitionDataManager):
    """Manages the partitioning related controls.

    Handles data generation and informs the model about changes in partition.
    """

    def __init__(
        self,
        dataset_loader,
        model,
        strategy="RandomEdges",
        partitioner_k=3,
        root_directory=None,
    ):
        """Initialize the Partitioning Data Manager.

        Uses/Creates partitioner and generates partition related parameters.

        Parameters
        ----------
        dataset_loader : AbstractGraphPartitioner or GraphDataLoader
            Either an instance of AbstractGraphPartitioner or GraphDataLoader.
        model: tf.keras.Model
            The model that is being trained.
        strategy: str
            Type of partitioning strategy to use.
        root_directory: str
            Directory where the partition manager files will be stored.
        """
        super(GeneralPartitionDataManager, self).__init__(
            dataset_loader, model, strategy, partitioner_k, root_directory
        )

    def _generate_partition_params(self):
        """Generates the metadata needed for persisting and loading partition embeddings and other parameters."""

        # create entity embeddings and optimizer hyperparams for all entities

        # compute each partition size
        update_part_size = int(np.ceil(self.num_ents / self.partitioner_k))
        num_optimizer_hyperparams = (
            self._model.optimizer.get_hyperparam_count()
        )
        # for each partition
        for part_num in range(self.partitioner_k):
            with shelve.open(
                self.ent_map_fname, writeback=True
            ) as ent_partition:
                # create the key (entity index) and value (optim params and
                # embs)
                for i in range(
                    update_part_size * part_num,
                    min(update_part_size * (part_num + 1), self.num_ents),
                ):
                    out_dict_key = str(i)
                    opt_param = np.zeros(
                        shape=(1, num_optimizer_hyperparams, self.internal_k),
                        dtype=np.float32,
                    )
                    # ent_emb = xavier(self.num_ents, self.internal_k, num_ents_bucket)
                    ent_emb = self._model.encoding_layer.ent_init(
                        shape=(1, self.internal_k), dtype=tf.float32
                    ).numpy()
                    ent_partition.update({out_dict_key: [opt_param, ent_emb]})

        # create relation embeddings and optimizer hyperparams for all relations
        # relations are not partitioned
        with shelve.open(self.rel_map_fname, writeback=True) as rel_partition:
            for i in range(self.num_rels):
                out_dict_key = str(i)
                # TODO change the hardcoding from 3 to actual hyperparam of
                # optim
                opt_param = np.zeros(
                    shape=(1, num_optimizer_hyperparams, self.internal_k),
                    dtype=np.float32,
                )
                # rel_emb = xavier(self.num_rels, self.internal_k, self.num_rels)
                rel_emb = self._model.encoding_layer.rel_init(
                    shape=(1, self.internal_k), dtype=tf.float32
                ).numpy()
                rel_partition.update({out_dict_key: [opt_param, rel_emb]})

    def _update_partion_embeddings(self, graph_data_loader, partition_number):
        """Persists the embeddings and other parameters after a partition is trained.

        Parameters
        ----------
        graph_data_loader : GraphDataLoader
            Data loader of the current partition that was trained.
        partition_number: int
            Partition number of the current partition that was trained.
        """
        # set the trained params back for persisting (exclude paddings)
        self.all_ent_embs = self._model.encoding_layer.ent_emb.numpy()[
            : len(self.ent_original_ids), :
        ]
        self.all_rel_embs = self._model.encoding_layer.rel_emb.numpy()[
            : len(self.rel_original_ids), :
        ]

        # get the optimizer params related to the embeddings
        (
            ent_opt_hyperparams,
            rel_opt_hyperparams,
        ) = self._model.optimizer.get_entity_relation_hyperparams()

        # get the number of params that are created by the optimizer
        num_opt_hyperparams = self._model.optimizer.get_hyperparam_count()

        # depending on optimizer, you can have 0 or more params
        if num_opt_hyperparams > 0:
            # store the params
            original_ent_hyperparams = []
            original_rel_hyperparams = []

            # get all the different params related to entities and relations
            # eg: beta1, beta2 related to embeddings (when using adam)
            for i in range(num_opt_hyperparams):
                original_ent_hyperparams.append(
                    ent_opt_hyperparams[i][: len(self.ent_original_ids)]
                )
                original_rel_hyperparams.append(
                    rel_opt_hyperparams[i][: len(self.rel_original_ids)]
                )

            # store for persistance
            self.all_rel_opt_params = np.stack(original_rel_hyperparams, 1)
            self.all_ent_opt_params = np.stack(original_ent_hyperparams, 1)

        # Open the buckets related to the partition and concat

        try:
            # persist entity related embs and optim params
            ent_partition = shelve.open(self.ent_map_fname, writeback=True)
            for i, key in enumerate(self.ent_original_ids):
                ent_partition[str(key)] = [
                    self.all_ent_opt_params[i: i + 1],
                    self.all_ent_embs[i: i + 1],
                ]

        finally:
            ent_partition.close()

        try:
            # persist relation related embs and optim params
            rel_partition = shelve.open(self.rel_map_fname, writeback=True)
            for i, key in enumerate(self.rel_original_ids):
                rel_partition[str(key)] = [
                    self.all_rel_opt_params[i: i + 1],
                    self.all_rel_embs[i: i + 1],
                ]

        finally:
            rel_partition.close()

    def _change_partition(self, graph_data_loader, partition_number):
        """Gets a new partition to train and loads all the parameters of the partition.

        Parameters
        ----------
        graph_data_loader : GraphDataLoader
            Data loader of the next partition that will be trained.
        partition_number: int
            Partition number of the next partition will be trained.
        """
        # from the graph data loader of the current partition get the original
        # entity ids
        ent_count_in_partition = (
            graph_data_loader.backend.mapper.get_entities_count()
        )
        self.ent_original_ids = graph_data_loader.backend.mapper.get_indexes(
            np.arange(ent_count_in_partition), type_of="e", order="ind2raw"
        )
        """
        with shelve.open(graph_data_loader.backend.mapper.entities_dict) as partition:
            # get the partition keys(remapped 0 - partition size)
            partition_keys = sorted([int(key) for key in partition.keys()])
            # get the original key's i.e. original entity ids (between 0 and total entities in dataset)
            self.ent_original_ids = [partition[str(key)] for key in partition_keys]
        """
        with shelve.open(self.ent_map_fname) as partition:
            self.all_ent_embs = []
            self.all_ent_opt_params = []
            for key in self.ent_original_ids:
                self.all_ent_opt_params.append(partition[key][0])
                self.all_ent_embs.append(partition[key][1])
            self.all_ent_embs = np.concatenate(self.all_ent_embs, 0)
            self.all_ent_opt_params = np.concatenate(
                self.all_ent_opt_params, 0
            )

        rel_count_in_partition = (
            graph_data_loader.backend.mapper.get_relations_count()
        )
        self.rel_original_ids = graph_data_loader.backend.mapper.get_indexes(
            np.arange(rel_count_in_partition), type_of="r", order="ind2raw"
        )
        """
        with shelve.open(graph_data_loader.backend.mapper.relations_dict) as partition:
            partition_keys = sorted([int(key) for key in partition.keys()])
            self.rel_original_ids = [partition[str(key)] for key in partition_keys]
        """

        with shelve.open(self.rel_map_fname) as partition:
            self.all_rel_embs = []
            self.all_rel_opt_params = []
            for key in self.rel_original_ids:
                self.all_rel_opt_params.append(partition[key][0])
                self.all_rel_embs.append(partition[key][1])
            self.all_rel_embs = np.concatenate(self.all_rel_embs, 0)
            self.all_rel_opt_params = np.concatenate(
                self.all_rel_opt_params, 0
            )

        # notify the model about the partition change
        self._model.partition_change_updates(
            len(self.ent_original_ids), self.all_ent_embs, self.all_rel_embs
        )

        # Optimizer params will exist only after it has been persisted once
        if self._model.current_epoch > 1:
            # TODO: needs to be better handled
            # get the optimizer params of the embs that will be trained
            rel_optim_hyperparams = []
            ent_optim_hyperparams = []

            num_opt_hyperparams = self._model.optimizer.get_hyperparam_count()
            for i in range(num_opt_hyperparams):
                rel_hyperparam_i = self.all_rel_opt_params[:, i, :]
                rel_hyperparam_i = np.pad(
                    rel_hyperparam_i,
                    ((0, self.num_rels - rel_hyperparam_i.shape[0]), (0, 0)),
                    "constant",
                    constant_values=(0),
                )
                rel_optim_hyperparams.append(rel_hyperparam_i)

                ent_hyperparam_i = self.all_ent_opt_params[:, i, :]
                ent_hyperparam_i = np.pad(
                    ent_hyperparam_i,
                    (
                        (0, self.max_ent_size - ent_hyperparam_i.shape[0]),
                        (0, 0),
                    ),
                    "constant",
                    constant_values=(0),
                )
                ent_optim_hyperparams.append(ent_hyperparam_i)

            # notify the optimizer and update the optimizer hyperparams
            self._model.optimizer.set_entity_relation_hyperparams(
                ent_optim_hyperparams, rel_optim_hyperparams
            )

    def on_complete(self):
        """Activities to be performed at the end of training.

        The manager persists the data (splits the entity partitions into individual embeddings).
        """
        update_part_size = int(np.ceil(self.num_ents / self.partitioner_k))
        for part_num in range(self.partitioner_k):
            with shelve.open(
                self.ent_map_fname, writeback=True
            ) as ent_partition:
                for i in range(
                    update_part_size * part_num,
                    min(update_part_size * (part_num + 1), self.num_ents),
                ):
                    ent_partition[str(i)] = ent_partition[str(i)][1][0]

        # create relation embeddings and optimizer hyperparams for all relations
        # relations are not partitioned
        with shelve.open(self.rel_map_fname, writeback=True) as rel_partition:
            for i in range(self.num_rels):
                rel_partition[str(i)] = rel_partition[str(i)][1][0]


@register_partitioning_manager("BucketPartitionDataManager")
class BucketPartitionDataManager(PartitionDataManager):
    """Manages the partitioning related controls.

    Handles data generation and informs model about changes in partition.
    """

    def __init__(
        self,
        dataset_loader,
        model,
        strategy="Bucket",
        partitioner_k=3,
        root_directory=None,
    ):
        """Initialize the Partitioning Data Manager.
        Uses/Creates partitioner and generates partition related parameters.

        Parameters
        ----------
        dataset_loader : AbstractGraphPartitioner or GraphDataLoader
            Either an instance of AbstractGraphPartitioner or GraphDataLoader.
        model: tf.keras.Model
            The model that is being trained.
        strategy: str
            Type of partitioning strategy to use.
        root_directory: str
            Directory where the partition manager files will be stored.
        """
        super(BucketPartitionDataManager, self).__init__(
            dataset_loader, model, strategy, partitioner_k, root_directory
        )

    def _generate_partition_params(self):
        """Generates the metadata needed for persisting and loading partition embeddings and other parameters."""

        num_optimizer_hyperparams = (
            self._model.optimizer.get_hyperparam_count()
        )

        # create entity embeddings and optimizer hyperparams for all entities
        for i in range(self.partitioner_k):
            with shelve.open(
                self.ent_map_fname, writeback=True
            ) as ent_partition:
                with shelve.open(self.partitioner.files[i]) as bucket:
                    out_dict_key = str(i)
                    num_ents_bucket = bucket["indexes"].shape[0]
                    # print(num_ents_bucket)
                    # TODO change the hardcoding from 3 to actual hyperparam of
                    # optim
                    opt_param = np.zeros(
                        shape=(
                            num_ents_bucket,
                            num_optimizer_hyperparams,
                            self.internal_k,
                        ),
                        dtype=np.float32,
                    )
                    ent_emb = self._model.encoding_layer.ent_init(
                        shape=(num_ents_bucket, self.internal_k),
                        dtype=tf.float32,
                    ).numpy()
                    ent_partition.update({out_dict_key: [opt_param, ent_emb]})

        # create relation embeddings and optimizer hyperparams for all relations
        # relations are not partitioned
        with shelve.open(self.rel_map_fname, writeback=True) as rel_partition:
            out_dict_key = str(0)
            # TODO change the hardcoding from 3 to actual hyperparam of optim
            opt_param = np.zeros(
                shape=(
                    self.num_rels,
                    num_optimizer_hyperparams,
                    self.internal_k,
                ),
                dtype=np.float32,
            )
            rel_emb = self._model.encoding_layer.rel_init(
                shape=(self.num_rels, self.internal_k), dtype=tf.float32
            ).numpy()
            rel_partition.update({out_dict_key: [opt_param, rel_emb]})

        # for every partition
        for i in range(len(self.partitioner.partitions)):
            # get the source and dest bucket
            # print(self.partitioner.partitions[i].backend.mapper.metadata['name'])
            splits = (
                self.partitioner.partitions[i]
                .backend.mapper.metadata["name"]
                .split("-")
            )
            source_bucket = splits[0][-1]
            dest_bucket = splits[1]
            all_keys_merged_buckets = []
            # get all the unique entities present in the buckets
            with shelve.open(
                self.partitioner.files[int(source_bucket)]
            ) as bucket:
                all_keys_merged_buckets.extend(bucket["indexes"])
            if source_bucket != dest_bucket:
                with shelve.open(
                    self.partitioner.files[int(dest_bucket)]
                ) as bucket:
                    all_keys_merged_buckets.extend(bucket["indexes"])

            # since we would be concatenating the bucket embeddings, let's find what 0, 1, 2 etc indices of
            # embedding matrix means.
            # bucket entity value to ent_emb matrix index mappings eg: 2001 ->
            # 0, 2002->1, 2003->2, ...
            merged_bucket_to_ent_mat_mappings = {}
            for key, val in zip(
                all_keys_merged_buckets,
                np.arange(0, len(all_keys_merged_buckets)),
            ):
                merged_bucket_to_ent_mat_mappings[key] = val
            emb_mat_order = []

            # partitions do not contain all entities of the bucket they belong to.
            # they will produce data from 0->n idx. So we need to remap the get position of the
            # entities of the partition in the concatenated emb matrix
            # data_index -> original_ent_index -> ent_emb_matrix mappings (a->b->c) 0->2002->1, 1->2003->2
            # (because 2001 may not exist in this partition)
            # a->b mapping
            num_ents_bucket = self.partitioner.partitions[
                i
            ].backend.mapper.get_entities_count()
            sorted_partition_keys = np.arange(num_ents_bucket)
            sorted_partition_values = self.partitioner.partitions[
                i
            ].backend.mapper.get_indexes(
                sorted_partition_keys, type_of="e", order="ind2raw"
            )
            for val in sorted_partition_values:
                # a->b->c mapping
                emb_mat_order.append(
                    merged_bucket_to_ent_mat_mappings[int(val)]
                )

            # store it
            with shelve.open(self.ent_meta_fname, writeback=True) as metadata:
                metadata[str(i)] = emb_mat_order

            rel_mat_order = []
            # with
            # shelve.open(self.partitioner.partitions[i].backend.mapper.metadata['relations'])
            # as rel_sh:
            num_rels_bucket = self.partitioner.partitions[
                i
            ].backend.mapper.get_relations_count()
            sorted_partition_keys = np.arange(num_rels_bucket)
            sorted_partition_values = self.partitioner.partitions[
                i
            ].backend.mapper.get_indexes(
                sorted_partition_keys, type_of="r", order="ind2raw"
            )
            # a : 0 to n
            for val in sorted_partition_values:
                # a->b mapping
                rel_mat_order.append(int(val))

            with shelve.open(self.rel_meta_fname, writeback=True) as metadata:
                metadata[str(i)] = rel_mat_order

    def _update_partion_embeddings(self, graph_data_loader, partition_number):
        """Persists the embeddings and other parameters after a partition is trained.

        Parameters
        ----------
        graph_data_loader : GraphDataLoader
            Data loader of the current partition that was trained.
        partition_number: int
            Partition number of the current partition that was trained.
        """
        # set the trained params back for persisting (exclude paddings)
        self.all_ent_embs[
            self.ent_original_ids
        ] = self._model.encoding_layer.ent_emb.numpy()[
            : len(self.ent_original_ids), :
        ]
        self.all_rel_embs[
            self.rel_original_ids
        ] = self._model.encoding_layer.rel_emb.numpy()[
            : len(self.rel_original_ids), :
        ]

        # get the optimizer params related to the embeddings
        (
            ent_opt_hyperparams,
            rel_opt_hyperparams,
        ) = self._model.optimizer.get_entity_relation_hyperparams()

        # get the number of params that are created by the optimizer
        num_opt_hyperparams = self._model.optimizer.get_hyperparam_count()

        # depending on optimizer, you can have 0 or more params
        if num_opt_hyperparams > 0:
            # store the params
            original_ent_hyperparams = []
            original_rel_hyperparams = []

            # get all the different params related to entities and relations
            # eg: beta1, beta2 related to embeddings (when using adam)
            for i in range(num_opt_hyperparams):
                original_ent_hyperparams.append(
                    ent_opt_hyperparams[i][: len(self.ent_original_ids)]
                )
                original_rel_hyperparams.append(
                    rel_opt_hyperparams[i][: len(self.rel_original_ids)]
                )

            # store for persistance
            self.all_rel_opt_params[self.rel_original_ids, :, :] = np.stack(
                original_rel_hyperparams, 1
            )
            self.all_ent_opt_params[self.ent_original_ids, :, :] = np.stack(
                original_ent_hyperparams, 1
            )

        # Open the buckets related to the partition and concat
        splits = graph_data_loader.backend.mapper.metadata["name"].split("-")
        source_bucket = splits[0][-1]
        dest_bucket = splits[1]

        try:
            # persist entity related embs and optim params
            s = shelve.open(self.ent_map_fname, writeback=True)

            # split and save self.all_ent_opt_params and self.all_ent_embs into
            # respective buckets
            opt_params = [
                self.all_ent_opt_params[: self.split_opt_idx],
                self.all_ent_opt_params[self.split_opt_idx:],
            ]
            emb_params = [
                self.all_ent_embs[: self.split_emb_idx],
                self.all_ent_embs[self.split_emb_idx:],
            ]

            s[source_bucket] = [opt_params[0], emb_params[0]]
            s[dest_bucket] = [opt_params[1], emb_params[1]]

        finally:
            s.close()

        try:
            # persist relation related embs and optim params
            s = shelve.open(self.rel_map_fname, writeback=True)
            s["0"] = [self.all_rel_opt_params, self.all_rel_embs]

        finally:
            s.close()

    def _change_partition(self, graph_data_loader, partition_number):
        """Gets a new partition to train and loads all the parameters of the partition.

        Parameters
        ----------
        graph_data_loader : GraphDataLoader
            Data loader of the next partition that will be trained.
        partition_number: int
            Partition number of the next partition will be trained.
        """
        try:
            # open the meta data related to the partition
            s = shelve.open(self.ent_meta_fname)
            # entities mapping ids
            self.ent_original_ids = s[str(partition_number)]
        finally:
            s.close()

        try:
            s = shelve.open(self.rel_meta_fname)
            # entities mapping ids
            self.rel_original_ids = s[str(partition_number)]

        finally:
            s.close()

        # Open the buckets related to the partition and concat
        splits = graph_data_loader.backend.mapper.metadata["name"].split("-")
        source_bucket = splits[0][-1]
        dest_bucket = splits[1]

        try:
            s = shelve.open(self.ent_map_fname)
            source_bucket_params = s[source_bucket]
            dest_source_bucket_params = s[dest_bucket]
            # full ent embs
            self.all_ent_embs = np.concatenate(
                [source_bucket_params[1], dest_source_bucket_params[1]]
            )
            self.split_emb_idx = source_bucket_params[1].shape[0]

            self.all_ent_opt_params = np.concatenate(
                [source_bucket_params[0], dest_source_bucket_params[0]]
            )
            self.split_opt_idx = source_bucket_params[0].shape[0]

            # now select only partition embeddings
            ent_embs = self.all_ent_embs[self.ent_original_ids]
            ent_opt_params = self.all_ent_opt_params[self.ent_original_ids]
        finally:
            s.close()

        try:
            s = shelve.open(self.rel_map_fname)
            # full rel embs
            self.all_rel_embs = s["0"][1]
            self.all_rel_opt_params = s["0"][0]
            # now select only partition embeddings
            rel_embs = self.all_rel_embs[self.rel_original_ids]
            rel_opt_params = self.all_rel_opt_params[self.rel_original_ids]
        finally:
            s.close()

        # notify the model about the partition change
        self._model.partition_change_updates(
            len(self.ent_original_ids), ent_embs, rel_embs
        )

        # Optimizer params will exist only after it has been persisted once
        if self._model.current_epoch > 1 or (
            self._model.current_epoch == 1
            and partition_number > self.partitioner_k
        ):
            # TODO: needs to be better handled
            # get the optimizer params of the embs that will be trained
            rel_optim_hyperparams = []
            ent_optim_hyperparams = []

            num_opt_hyperparams = self._model.optimizer.get_hyperparam_count()
            for i in range(num_opt_hyperparams):
                rel_hyperparam_i = rel_opt_params[:, i, :]
                rel_hyperparam_i = np.pad(
                    rel_hyperparam_i,
                    ((0, self.num_rels - rel_hyperparam_i.shape[0]), (0, 0)),
                    "constant",
                    constant_values=(0),
                )
                rel_optim_hyperparams.append(rel_hyperparam_i)

                ent_hyperparam_i = ent_opt_params[:, i, :]
                ent_hyperparam_i = np.pad(
                    ent_hyperparam_i,
                    (
                        (0, self.max_ent_size - ent_hyperparam_i.shape[0]),
                        (0, 0),
                    ),
                    "constant",
                    constant_values=(0),
                )
                ent_optim_hyperparams.append(ent_hyperparam_i)

            # notify the optimizer and update the optimizer hyperparams
            self._model.optimizer.set_entity_relation_hyperparams(
                ent_optim_hyperparams, rel_optim_hyperparams
            )

    def on_complete(self):
        """Activities to be performed on end of training.

        The manager persists the data (splits the entity partitions into individual embeddings).
        """
        for i in range(self.partitioner_k - 1, -1, -1):
            with shelve.open(self.partitioner.files[i]) as bucket:
                with shelve.open(
                    self.ent_map_fname, writeback=True
                ) as ent_partition:
                    # get the bucket embeddings
                    # split and store separately
                    for key, val in zip(
                        bucket["indexes"], ent_partition[str(i)][1]
                    ):
                        ent_partition[str(key)] = val
                    if i != 0:
                        del ent_partition[str(i)]
        with shelve.open(self.rel_map_fname, writeback=True) as rel_partition:
            # get the bucket embeddings
            # split and store separately
            for key in range(rel_partition["0"][1].shape[0] - 1, -1, -1):
                rel_partition[str(key)] = rel_partition["0"][1][key]


def get_partition_adapter(
    dataset_loader,
    model,
    strategy="Bucket",
    partitioning_k=3,
    root_directory=None,
):
    """Returns partition manager depending on the one registered by the partitioning strategy.

    Parameters
    ----------
    dataset_loader: AbstractGraphPartitioner or GraphDataLoader
        Parent dataset loader that will be used for partitioning.
    model: tf.keras.Model
        KGE model that will be managed while using partitioning.
    strategy: str
        Graph partitioning strategy.
    """
    if isinstance(dataset_loader, AbstractGraphPartitioner):
        partitioner_manager = PARTITION_MANAGER_REGISTRY.get(
            dataset_loader.manager
        )(
            dataset_loader,
            model,
            dataset_loader.name,
            dataset_loader._k,
            root_directory,
        )

    else:
        partitioner = PARTITION_ALGO_REGISTRY.get(strategy)(
            dataset_loader, k=partitioning_k
        )
        partitioner_manager = PARTITION_MANAGER_REGISTRY.get(
            partitioner.manager
        )(partitioner, model, strategy, partitioning_k, root_directory)

    return partitioner_manager
