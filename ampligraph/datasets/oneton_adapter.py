import numpy as np
from ..datasets import NumpyDatasetAdapter
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class OneToNDatasetAdapter(NumpyDatasetAdapter):
    r"""1-to-N Dataset Adapter.

        Given a triples dataset X comprised of n triples in the form (s, p, o), this dataset adapter will
        generate one-hot outputs for each (s, p) tuple to all entities o that are found in X.

        E.g: X = [[a, p, b],
                  [a, p, d],
                  [c, p, d],
                  [c, p, e],
                  [c, p, f]]

        Gives a one-hot vector mapping of entities to indices:

            Entities: [a, b, c, d, e, f]
            Indices: [0, 1, 2, 3, 4, 5]

        One-hot outputs are produced for each (s, p) tuple to all valid object indices in the dataset:

                  #  [a, b, c, d, e, f]
            (a, p) : [0, 1, 0, 1, 0, 0]

        The ```get_next_batch``` function yields the (s, p, o) triple and one-hot vector corresponding to the (s, p)
        tuple.

        If batches are generated with ```unique_pairs=True``` then only one instance of each unique (s, p) tuple
        is returned:

            (a, p) : [0, 1, 0, 1, 0, 0]
            (c, p) : [0, 0, 0, 1, 1, 1]

        Otherwise batch outputs are generated in dataset order (required for evaluating test set, but gives a higher
        weight to more frequent (s, p) pairs if used during model training):

            (a, p) : [0, 1, 0, 1, 0, 0]
            (a, p) : [0, 1, 0, 1, 0, 0]
            (c, p) : [0, 0, 0, 1, 1, 1]
            (c, p) : [0, 0, 0, 1, 1, 1]
            (c, p) : [0, 0, 0, 1, 1, 1]

    """

    def __init__(self, low_memory=False):
        """Initialize the class variables

        Parameters
        ----------
        low_memory : bool
            If low_memory flag set to True the output vectors indices are generated on-the-fly in the batch yield
            function, which lowers memory usage but increases training time.

        """
        super(OneToNDatasetAdapter, self).__init__()

        self.filter_mapping = None
        self.filtered_status = {}
        self.paired_status = {}
        self.output_mapping = None
        self.output_onehot = {}
        self.low_memory = low_memory

    def set_filter(self, filter_triples, mapped_status=False):
        """ Set the filter to be used while generating batch outputs.

        Parameters
        ----------
        filter_triples : nd-array
            Triples to be used as a filter.
        mapped_status : bool
            Bool indicating if filter has already been mapped to internal indices.

        """

        self.set_data(filter_triples, 'filter', mapped_status)
        self.filter_mapping = self.generate_output_mapping('filter')

    def generate_outputs(self, dataset_type='train', use_filter=False, unique_pairs=True):
        """Generate one-hot outputs for a dataset.

        Parameters
        ----------
        dataset_type : string
            Indicates which dataset to generate outputs for.
        use_filter : bool
            Bool indicating whether to generate outputs using the filter set by `set_filter()`. Default: False
        unique_pairs : bool
            Bool indicating whether to generate outputs according to unique pairs of (subject, predicate), otherwise
            will generate outputs in same row-order as the triples in the specified dataset. Default: True.

        """

        if dataset_type not in self.dataset.keys():
            msg = 'Unable to generate outputs: dataset `{}` not found. ' \
                  'Use `set_data` to set dataset in adapter first.'.format(dataset_type)
            raise KeyError(msg)

        if dataset_type in ['valid', 'test']:
            if unique_pairs:
                # This is just a friendly warning - in most cases the test and valid sets should NOT be unique_pairs.
                msg = 'Generating outputs for dataset `{}` with unique_pairs=True. ' \
                      'Are you sure this is desired behaviour?'.format(dataset_type)
                logger.warning(msg)

        if use_filter:
            if self.filter_mapping is None:
                msg = 'Filter not found: cannot generate one-hot outputs with `use_filter=True` ' \
                      'if a filter has not been set.'
                raise ValueError(msg)
            else:
                output_dict = self.filter_mapping
        else:
            if self.output_mapping is None:
                msg = 'Output mapping was not created before generating one-hot vectors. '
                raise ValueError(msg)
            else:
                output_dict = self.output_mapping

        if self.low_memory:
            # With low_memory=True the output indices are generated on the fly in the batch yield function
            pass
        else:
            if unique_pairs:
                X = np.unique(self.dataset[dataset_type][:, [0, 1]], axis=0).astype(np.int32)
            else:
                X = self.dataset[dataset_type]

            # Initialize np.array of shape [len(X), num_entities]
            self.output_onehot[dataset_type] = np.zeros((len(X), len(self.ent_to_idx)), dtype=np.int8)

            # Set one-hot indices using output_dict
            for i, x in enumerate(X):
                indices = output_dict.get((x[0], x[1]), [])
                self.output_onehot[dataset_type][i, indices] = 1

            # Set flags indicating filter and unique pair status of outputs for given dataset.
            self.filtered_status[dataset_type] = use_filter
            self.paired_status[dataset_type] = unique_pairs

    def generate_output_mapping(self, dataset_type='train'):
        """ Creates dictionary keyed on (subject, predicate) to list of objects

        Parameters
        ----------
        dataset_type : string
            Indicates which dataset to generate output mapping from.

        Returns
        -------
            dict
        """

        # if data is not already mapped, then map before creating output map
        if not self.mapped_status[dataset_type]:
            self.map_data()

        output_mapping = dict()

        for s, p, o in self.dataset[dataset_type]:
            output_mapping.setdefault((s, p), []).append(o)

        return output_mapping

    def set_output_mapping(self, output_dict, clear_outputs=True):
        """ Set the mapping used to generate one-hot outputs vectors.

        Setting a new output mapping will clear_outputs any previously generated outputs, as otherwise
        can lead to a situation where old outputs are returned from batch function.

        Parameters
        ----------
        output_dict : dict
            (subject, predicate) to object indices
        clear_outputs: bool
            Clears any one hot outputs held by the adapter, as otherwise can lead to a situation where onehot
            outputs generated by a different mapping are returned from the batch function. Default: True.

        """

        self.output_mapping = output_dict

        # Clear any onehot outputs previously generated
        if clear_outputs:
            self.clear_outputs()

    def clear_outputs(self, dataset_type=None):
        """ Clears generated one-hot outputs currently held by the adapter.

        Parameters
        ----------
        dataset_type: string
            indicates which dataset to clear_outputs. Default: None (clears all).

        """

        if dataset_type is None:
            self.output_onehot = {}
            self.filtered_status = {}
            self.paired_status = {}
        else:
            del self.output_onehot[dataset_type]
            del self.filtered_status[dataset_type]
            del self.paired_status[dataset_type]

    def verify_outputs(self, dataset_type, use_filter, unique_pairs):
        """Verifies if one-hot outputs currently held in adapter correspond to the use_filter and unique_pairs
        options.

        Parameters
        ----------
        dataset_type: string
            indicates which dataset to use
        use_filter : bool
            Flag to indicate whether the one-hot outputs are generated from filtered or unfiltered datasets
        unique_pairs : bool
            Flag to indicate whether the one-hot outputs are generated by unique (s, p) pairs or in dataset order.

        Returns
        -------
        bool
            If False then outputs must be re-generated for the specified dataset and parameters.

        """

        if dataset_type not in self.output_onehot.keys():
            # One-hot outputs have not been generated for this dataset_type
            return False

        if dataset_type not in self.filtered_status.keys():
            # This shouldn't happen.
            logger.debug('Dataset {} is in adapter, but filtered_status is not set.'.format(dataset_type))
            return False

        if dataset_type not in self.paired_status.keys():
            logger.debug('Dataset {} is in adapter, but paired_status is not set.'.format(dataset_type))
            return False

        if use_filter != self.filtered_status[dataset_type]:
            return False

        if unique_pairs != self.paired_status[dataset_type]:
            return False

        return True

    def get_next_batch(self, batches_count=-1, dataset_type='train', use_filter=False, unique_pairs=True):
        """Generator that returns the next batch of data.

        Parameters
        ----------
        batches_count: int
            number of batches per epoch (default: -1, i.e. uses batch_size of 1)
        dataset_type: string
            indicates which dataset to use
        use_filter : bool
            Flag to indicate whether the one-hot outputs are generated from filtered or unfiltered datasets
        unique_pairs : bool
            Flag to indicate whether the one-hot outputs are generated by unique (s, p) pairs or in dataset order.

        Returns
        -------
        batch_output : nd-array, shape=[batch_size, 3]
            A batch of triples from the dataset type specified. If unique_pairs=True, then the object column
            will be set to zeros.
        batch_onehot : nd-array
            A batch of onehot arrays corresponding to `batch_output` triples
        """

        # if data is not already mapped, then map before returning the batch
        if not self.mapped_status[dataset_type]:
            self.map_data()

        if unique_pairs:
            X = np.unique(self.dataset[dataset_type][:, [0, 1]], axis=0).astype(np.int32)
            X = np.c_[X, np.zeros(len(X))]  # Append dummy object columns
        else:
            X = self.dataset[dataset_type]
        dataset_size = len(X)

        if batches_count == -1:
            batch_size = 1
            batches_count = dataset_size
        else:
            batch_size = int(np.ceil(dataset_size / batches_count))

        if use_filter and self.filter_mapping is None:
            msg = 'Cannot set `use_filter=True` if a filter has not been set in the adapter. '
            logger.error(msg)
            raise ValueError(msg)

        if not self.low_memory:

            if not self.verify_outputs(dataset_type, use_filter=use_filter, unique_pairs=unique_pairs):
                # Verifies that onehot outputs are as expected given filter and unique_pair settings
                msg = 'Generating one-hot outputs for {} [filtered: {}, unique_pairs: {}]'\
                    .format(dataset_type, use_filter, unique_pairs)
                logger.info(msg)
                self.generate_outputs(dataset_type, use_filter=use_filter, unique_pairs=unique_pairs)

            # Yield batches
            for i in range(batches_count):

                out = np.int32(X[(i * batch_size):((i + 1) * batch_size), :])
                out_onehot = self.output_onehot[dataset_type][(i * batch_size):((i + 1) * batch_size), :]

                yield out, out_onehot

        else:
            # Low-memory, generate one-hot outputs per batch on the fly
            if use_filter:
                output_dict = self.filter_mapping
            else:
                output_dict = self.output_mapping

            # Yield batches
            for i in range(batches_count):

                out = np.int32(X[(i * batch_size):((i + 1) * batch_size), :])
                out_onehot = np.zeros(shape=[out.shape[0], len(self.ent_to_idx)], dtype=np.int32)

                for j, x in enumerate(out):
                    indices = output_dict.get((x[0], x[1]), [])
                    out_onehot[j, indices] = 1

                yield out, out_onehot

    def get_next_batch_subject_corruptions(self, batch_size=-1, dataset_type='train', use_filter=True):
        """Batch generator for subject corruptions.

        To avoid multiple redundant forward-passes through the network, subject corruptions are performed once for
        each relation, and results accumulated for valid test triples.

        If there are no test triples for a relation, then that relation is ignored.

        Use batch_size to control memory usage (as a batch_size*N tensor will be allocated, where N is number
        of unique entities.)

        Parameters
        ----------
        batch_size: int
            Maximum batch size returned.
        dataset_type: string
            indicates which dataset to use
        use_filter : bool
            Flag to indicate whether to return the one-hot outputs are generated from filtered or unfiltered datasets

        Returns
        -------

        test_triples : nd-array of shape (?, 3)
            The set of all triples from the dataset type specified that include the predicate currently returned
            in batch_triples.
        batch_triples : nd-array of shape (M, 3), where M is the subject corruption batch size.
            A batch of triples corresponding to subject corruptions of just one predicate.
        batch_onehot : nd-array of shape (M, N), where N is number of unique entities.
            A batch of onehot arrays corresponding to the batch_triples output.

        """

        if use_filter:
            output_dict = self.filter_mapping
        else:
            output_dict = self.output_mapping

        if batch_size == -1:
            batch_size = len(self.ent_to_idx)

        ent_list = np.array(list(self.ent_to_idx.values()))
        rel_list = np.array(list(self.rel_to_idx.values()))

        for rel in rel_list:

            # Select test triples that have this relation
            rel_idx = self.dataset[dataset_type][:, 1] == rel
            test_triples = self.dataset[dataset_type][rel_idx]

            ent_idx = 0

            while ent_idx < len(ent_list):

                ents = ent_list[ent_idx:ent_idx + batch_size]
                ent_idx += batch_size

                # Note: the object column is just a dummy value so set to 0
                out = np.stack([ents, np.repeat(rel, len(ents)), np.repeat(0, len(ents))], axis=1)

                # Set one-hot filter
                out_filter = np.zeros((out.shape[0], len(ent_list)), dtype=np.int8)
                for j, x in enumerate(out):
                    indices = output_dict.get((x[0], x[1]), [])
                    out_filter[j, indices] = 1

                yield test_triples, out, out_filter

    def _validate_data(self, data):
        """Validates the data
        """
        if type(data) != np.ndarray:
            msg = 'Invalid type for input data. Expected ndarray, got {}'.format(type(data))
            raise ValueError(msg)

        if (np.shape(data)[1]) != 3:
            msg = 'Invalid size for input data. Expected number of column 3, got {}'.format(np.shape(data)[1])
            raise ValueError(msg)

    def set_data(self, dataset, dataset_type=None, mapped_status=False):
        """Set the dataset based on the type.

        Note: If you pass the same dataset type (which exists) it will be overwritten

        Parameters
        ----------
        dataset : nd-array or dictionary
            dataset of triples
        dataset_type : string
            if the dataset parameter is an nd- array then this indicates the type of the data being based
        mapped_status : bool
            indicates whether the data has already been mapped to the indices

        """
        if isinstance(dataset, dict):
            for key in dataset.keys():
                self._validate_data(dataset[key])
                self.dataset[key] = dataset[key]
                self.mapped_status[key] = mapped_status
        elif dataset_type is not None:
            self._validate_data(dataset)
            self.dataset[dataset_type] = dataset
            self.mapped_status[dataset_type] = mapped_status
        else:
            raise Exception("Incorrect usage. Expected a dictionary or a combination of dataset and it's type.")

        # If the concept-idx mappings are present, then map the passed dataset
        if not (len(self.rel_to_idx) == 0 or len(self.ent_to_idx) == 0):
            print('Mapping set data: {}'.format(dataset_type))
            self.map_data()
