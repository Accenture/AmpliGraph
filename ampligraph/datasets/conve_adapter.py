import numpy as np
from ..datasets import NumpyDatasetAdapter
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class ConvEDatasetAdapter(NumpyDatasetAdapter):

    def __init__(self, low_memory=False):
        """Initialize the class variables
        """
        super(ConvEDatasetAdapter, self).__init__()

        self.filter_mapping = None
        self.filtered_status = None
        self.output_mapping = None
        self.output_onehot = {}
        self.low_memory = low_memory

    def set_filter(self, filter_triples, mapped_status=False):
        """ Set the filter to be used while generating an evaluation batch.

        Parameters
        ----------
        filter_triples : nd-array
            triples that would be used as filter
        """

        self.set_data(filter_triples, 'filter', mapped_status)
        self.filter_mapping = self.generate_output_mapping('filter')

    def generate_onehot_outputs(self, dataset_type='train', use_filter=False):
        """ Create one-hot outputs for a dataset using an output mapping.

        Parameters
        ----------
        dataset_type : string indicating which dataset to create onehot outputs for
        use_filter : bool indicating whether to use a filter when generating onehot outputs.

        Returns
        -------

        """

        if dataset_type not in self.dataset.keys():
            msg = 'Dataset `{}` not found: cannot generate one-hot outputs. ' \
                  'Please use `set_data` to set the dataset first.'.format(dataset_type)
            raise ValueError(msg)

        if use_filter:
            # Generate one-hot outputs using the filter
            if self.filter_mapping is None:
                msg = 'Filter not found: cannot generate one-hot outputs with `use_filter=True` ' \
                      'if a filter has not been set.'
                raise ValueError(msg)
            else:
                output_dict = self.filter_mapping
        else:
            # Generate one-hot outputs using the dataset set with create_output_mapping() and set_output_mapping()
            if self.output_mapping is None:
                msg = 'Output mapping was not created before generating one-hot vectors. '
                raise ValueError(msg)
            else:
                output_dict = self.output_mapping

        if not self.low_memory:

            # Initialize np.array of shape [dataset_size, num_entities]
            self.output_onehot[dataset_type] = np.zeros((self.dataset[dataset_type].shape[0], len(self.ent_to_idx)),
                                                        dtype=np.int8)

            # Set one-hot indices using output_dict
            for i, x in enumerate(self.dataset[dataset_type]):
                indices = output_dict.get((x[0], x[1]), [])
                self.output_onehot[dataset_type][i, indices] = 1

            self.filtered_status[dataset_type] = use_filter

        else:
            # NB: With low_memory=True the output indices are generated on the fly in the batch generators
            pass

    def generate_output_mapping(self, dataset_type='train'):
        """ Creates dictionary keyed on (subject, predicate) to list of objects

        Parameters
        ----------
        dataset_type

        Returns
        -------

        """

        # if data is not already mapped, then map before creating output map
        if not self.mapped_status[dataset_type]:
            self.map_data()

        output_mapping = dict()

        for s, p, o in self.dataset[dataset_type]:
            output_mapping.setdefault((s, p), []).append(o)

        return output_mapping

    def set_output_mapping(self, output_dict):
        """ Set the output mapping used to generate onehot vectors. Required for loading saved model parameters.

        Parameters
        ----------
        output_dict : dictionary of subject, predicate to object indices

        Returns
        -------

        """

        self.output_mapping = output_dict

    def get_next_batch(self, batches_count=-1, dataset_type='train', use_filter=False):
        """Generator that returns the next batch of data.
        
        Parameters
        ----------
        batches_count: int
            number of batches per epoch (default: -1, i.e. uses batch_size of 1)
        dataset_type: string
            indicates which dataset to use
        use_filter : bool
            Flag to indicate whether to return the one-hot outputs are generated from filtered or unfiltered datasets
        Returns
        -------
        batch_output : nd-array
            A batch of triples from the dataset type specified
        batch_onehot : nd-array
            A batch of onehot arrays corresponding to `batch_output` triples
        """

        # if data is not already mapped, then map before returning the batch
        if not self.mapped_status[dataset_type]:
            self.map_data()

        if batches_count == -1:
            batch_size = 1
            batches_count = self.get_size(dataset_type)
        else:
            batch_size = int(np.ceil(self.get_size(dataset_type) / batches_count))

        if use_filter and self.filter_mapping is None:
            msg = 'Cannot set `use_filter=True` if a filter has not been set in the adapter. '
            raise ValueError(msg)

        if not self.low_memory:

            # If onehot outputs for dataset_type aren't initialized then create them, or
            # If using a filter, and the onehot outputs for dataset_type were previously generated without the filter
            if dataset_type not in self.output_onehot.keys() or (use_filter and not self.filtered_status[dataset_type]):
                self.generate_onehot_outputs(dataset_type, use_filter=use_filter)


            # Yield batches
            for i in range(batches_count):

                out = np.int32(self.dataset[dataset_type][(i * batch_size):((i + 1) * batch_size), :])
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

                out = np.int32(self.dataset[dataset_type][(i * batch_size):((i + 1) * batch_size), :])

                out_onehot = np.zeros(shape=[out.shape[0], len(self.ent_to_idx)], dtype=np.int32)
                for j, x in enumerate(out):
                    indices = output_dict.get((x[0], x[1]), [])
                    out_onehot[j, indices] = 1

                yield out, out_onehot

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
