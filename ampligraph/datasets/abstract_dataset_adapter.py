# Copyright 2019-2021 The AmpliGraph Authors. All Rights Reserved.
#
# This file is Licensed under the Apache License, Version 2.0.
# A copy of the Licence is available in LICENCE, or at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
import abc


class AmpligraphDatasetAdapter(abc.ABC):
    """Abstract class for dataset adapters
       Developers can design in similar format to adapt data from different sources to feed to ampligraph.
    """
    def __init__(self):
        """Initialize the class variables
        """
        self.dataset = {}

        # relation to idx mappings
        self.rel_to_idx = {}
        # entities to idx mappings
        self.ent_to_idx = {}
        # Mapped status of each dataset
        self.mapped_status = {}
        # link weights for focusE
        self.focusE_numeric_edge_values = {}

    def use_mappings(self, rel_to_idx, ent_to_idx):
        """Use an existing mapping with the datasource.
        """
        self.rel_to_idx = rel_to_idx
        self.ent_to_idx = ent_to_idx
        # set the mapped status to false, since we are changing the dictionary
        for key in self.dataset.keys():
            self.mapped_status[key] = False

    def generate_mappings(self, use_all=False):
        """Generate mappings from either train set or use all dataset to generate mappings
        Parameters
        ----------
        use_all : boolean
            If True, it generates mapping from all the data. If False, it only uses training set to generate mappings

        Returns
        -------
        rel_to_idx : dictionary
            Relation to idx mapping dictionary
        ent_to_idx : dictionary
            entity to idx mapping dictionary
        """
        raise NotImplementedError('Abstract Method not implemented!')

    def get_size(self, dataset_type="train"):
        """Returns the size of the specified dataset
        Parameters
        ----------
        dataset_type : string
            type of the dataset

        Returns
        -------
        size : int
            size of the specified dataset
        """

        raise NotImplementedError('Abstract Method not implemented!')

    def data_exists(self, dataset_type="train"):
        """Checks if a dataset_type exists in the adapter.
        Parameters
        ----------
        dataset_type : string
            type of the dataset

        Returns
        -------
        exists : bool
            Boolean indicating if dataset_type exists in the adapter.
        """

        raise NotImplementedError('Abstract Method not implemented!')

    def set_data(self, dataset, dataset_type=None, mapped_status=False):
        """set the dataset based on the type
        Parameters
        ----------
        dataset : nd-array or dictionary
            dataset of triples
        dataset_type : string
            if the dataset parameter is an nd- array then this indicates the type of the data being based
        mapped_status : bool
            indicates whether the data has already been mapped to the indices

        """
        raise NotImplementedError('Abstract Method not implemented!')

    def map_data(self, remap=False):
        """map the data to the mappings of ent_to_idx and rel_to_idx
        Parameters
        ----------
        remap : boolean
            remap the data, if already mapped. One would do this if the dictionary is updated.
        """
        raise NotImplementedError('Abstract Method not implemented!')

    def set_filter(self, filter_triples):
        """set's the filter that need to be used while generating evaluation batch
        Parameters
        ----------
        filter_triples : nd-array
            triples that would be used as filter
        """
        raise NotImplementedError('Abstract Method not implemented!')

    def get_next_batch(self, batches_count=-1, dataset_type="train", use_filter=False):
        """Generator that returns the next batch of data.

        Parameters
        ----------
        dataset_type: string
            indicates which dataset to use
        batches_count: int
            number of batches per epoch (default: -1, i.e. uses batch_size of 1)
        use_filter : bool
            Flag to indicate whether to return the concepts that need to be filtered

        Returns
        -------
        batch_output : nd-array
            yields a batch of triples from the dataset type specified
        participating_objects : nd-array [n,1]
            all objects that were involved in the s-p-? relation. This is returned only if use_filter is set to true.
        participating_subjects : nd-array [n,1]
            all subjects that were involved in the ?-p-o relation. This is returned only if use_filter is set to true.
        """
        raise NotImplementedError('Abstract Method not implemented!')

    def cleanup(self):
        """Cleans up the internal state
        """
        raise NotImplementedError('Abstract Method not implemented!')
