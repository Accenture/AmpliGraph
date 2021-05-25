# Copyright 2019-2021 The AmpliGraph Authors. All Rights Reserved.
#
# This file is Licensed under the Apache License, Version 2.0.
# A copy of the Licence is available in LICENCE, or at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
import numpy as np
from ..datasets import AmpligraphDatasetAdapter, SQLiteAdapter


class NumpyDatasetAdapter(AmpligraphDatasetAdapter):

    def __init__(self):
        """Initialize the class variables
        """
        super(NumpyDatasetAdapter, self).__init__()
        # NumpyDatasetAdapter uses SQLAdapter to filter (if filters are set)
        self.filter_adapter = None

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
        from ..evaluation import create_mappings
        if use_all:
            complete_dataset = []
            for key in self.dataset.keys():
                complete_dataset.append(self.dataset[key])
            self.rel_to_idx, self.ent_to_idx = create_mappings(np.concatenate(complete_dataset, axis=0))

        else:
            self.rel_to_idx, self.ent_to_idx = create_mappings(self.dataset["train"])

        return self.rel_to_idx, self.ent_to_idx

    def use_mappings(self, rel_to_idx, ent_to_idx):
        """Use an existing mapping with the datasource.
        """
        super().use_mappings(rel_to_idx, ent_to_idx)

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
        return self.dataset[dataset_type].shape[0]

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

        return dataset_type in self.dataset.keys()

    def get_next_batch(self, batches_count=-1, dataset_type="train", use_filter=False):
        """Generator that returns the next batch of data.

        Parameters
        ----------
        batches_count: int
            number of batches per epoch (default: -1, i.e. uses batch_size of 1)
        dataset_type: string
            indicates which dataset to use
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
        # if data is not already mapped, then map before returning the batch
        if not self.mapped_status[dataset_type]:
            self.map_data()

        if batches_count == -1:
            batch_size = 1
            batches_count = self.get_size(dataset_type)
        else:
            batch_size = int(np.ceil(self.get_size(dataset_type) / batches_count))

        for i in range(batches_count):
            output = []
            out = np.int32(self.dataset[dataset_type][(i * batch_size):((i + 1) * batch_size), :])
            output.append(out)
            
            try:
                focusE_numeric_edge_values_batch = self.focusE_numeric_edge_values[
                    dataset_type][(i * batch_size):((i + 1) * batch_size), :]
                output.append(focusE_numeric_edge_values_batch)
            except KeyError:
                pass
            
            if use_filter:
                # get the filter values by querying the database
                participating_objects, participating_subjects = self.filter_adapter.get_participating_entities(out)
                output.append(participating_objects)
                output.append(participating_subjects)
                
            yield output

    def map_data(self, remap=False):
        """map the data to the mappings of ent_to_idx and rel_to_idx
        Parameters
        ----------
        remap : boolean
            remap the data, if already mapped. One would do this if the dictionary is updated.
        """
        from ..evaluation import to_idx
        if len(self.rel_to_idx) == 0 or len(self.ent_to_idx) == 0:
            self.generate_mappings()

        for key in self.dataset.keys():
            if (not self.mapped_status[key]) or (remap is True):
                self.dataset[key] = to_idx(self.dataset[key],
                                           ent_to_idx=self.ent_to_idx,
                                           rel_to_idx=self.rel_to_idx)
                self.mapped_status[key] = True

    def _validate_data(self, data):
        """ Validates the data
        """
        if type(data) != np.ndarray:
            msg = 'Invalid type for input data. Expected ndarray, got {}'.format(type(data))
            raise ValueError(msg)

        if (np.shape(data)[1]) != 3:
            msg = 'Invalid size for input data. Expected number of column 3, got {}'.format(np.shape(data)[1])
            raise ValueError(msg)

    def set_data(self, dataset, dataset_type=None, mapped_status=False, focusE_numeric_edge_values=None):
        """set the dataset based on the type.
            Note: If you pass the same dataset type (which exists) it will be overwritten

        Parameters
        ----------
        dataset : nd-array or dictionary
            dataset of triples
        dataset_type : string
            if the dataset parameter is an nd- array then this indicates the type of the data being based
        mapped_status : bool
            indicates whether the data has already been mapped to the indices
        focusE_numeric_edge_values: nd-array
            list of all the numeric values associated the link
        """
        if isinstance(dataset, dict):
            for key in dataset.keys():
                self._validate_data(dataset[key])
                self.dataset[key] = dataset[key]
                self.mapped_status[key] = mapped_status
                if focusE_numeric_edge_values is not None:
                    self.focusE_numeric_edge_values[key] = focusE_numeric_edge_values[key]
        elif dataset_type is not None:
            self._validate_data(dataset)
            self.dataset[dataset_type] = dataset
            self.mapped_status[dataset_type] = mapped_status
            if focusE_numeric_edge_values is not None:
                self.focusE_numeric_edge_values[dataset_type] = focusE_numeric_edge_values
        else:
            raise Exception("Incorrect usage. Expected a dictionary or a combination of dataset and it's type.")

        # If the concept-idx mappings are present, then map the passed dataset
        if not (len(self.rel_to_idx) == 0 or len(self.ent_to_idx) == 0):
            self.map_data()

    def set_filter(self, filter_triples, mapped_status=False):
        """set's the filter that need to be used while generating evaluation batch
           Note: This adapter uses SQL backend to do filtering
        Parameters
        ----------
        filter_triples : nd-array
            triples that would be used as filter
        """
        self.filter_adapter = SQLiteAdapter()
        self.filter_adapter.use_mappings(self.rel_to_idx, self.ent_to_idx)
        self.filter_adapter.set_data(filter_triples, "filter", mapped_status)

    def cleanup(self):
        """Cleans up the internal state.
        """
        if self.filter_adapter is not None:
            self.filter_adapter.cleanup()
            self.filter_adapter = None
