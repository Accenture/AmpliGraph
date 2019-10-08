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
        self.filter_adapter = None
        self.filter_mapping = None
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

        filter_adapter = ConvEDatasetAdapter()
        filter_adapter.use_mappings(self.rel_to_idx, self.ent_to_idx)
        filter_adapter.set_data(filter_triples, 'filter', mapped_status)
        self.filter_mapping = filter_adapter.generate_output_mapping('filter')

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
            # This should be the training set in most instances.
            if self.output_mapping is None:
                msg = 'Output mapping was not created before generating one-hot vectors. '
                raise ValueError(msg)
            else:
                output_dict = self.output_mapping

        if not self.low_memory:

            self.output_onehot[dataset_type] = np.zeros((self.dataset[dataset_type].shape[0], len(self.ent_to_idx)),
                                                        dtype=np.int8)

            for idx, triple in enumerate(self.dataset[dataset_type]):
                key = (triple[0], triple[1])
                if key in output_dict.keys():
                    indices = output_dict[key]
                else:
                    indices = []

                self.output_onehot[dataset_type][idx, indices] = 1.0
        else:
            # NB: With low_memory=True the output indices are generated on the fly in the batch generators
            logger.debug('Low memory=True')
            pass

    def generate_output_mapping(self, dataset_type='train'):
        """ Creates dictionary of subject, predicate to object(s)

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
            key = (s, p)
            if key in output_mapping:
                output_mapping[key].append(o)
            else:
                output_mapping[key] = [o]

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

    def get_next_train_batch(self, batch_size=1, dataset_type="train"):
        """Generator that returns the next batch of data.
        
        Parameters
        ----------
        batch_size : int
            data size that needs to be returned
        dataset_type: string
            indicates which dataset to use
        Returns
        -------
        batch_output : nd-array
            yields a batch of triples from the dataset type specified
        """

        # if data is not already mapped, then map before returning the batch
        if not self.mapped_status[dataset_type]:
            self.map_data()

        if not self.low_memory:
            # If onehot outputs aren't initialized ..
            if dataset_type not in self.output_onehot.keys():
                self.generate_onehot_outputs(dataset_type, use_filter=False)

            batches_count = int(np.ceil(self.get_size(dataset_type) / batch_size))
            for i in range(batches_count):
                out = np.int32(self.dataset[dataset_type][(i * batch_size):((i + 1) * batch_size), :])
                out_onehot = self.output_onehot[dataset_type][(i * batch_size):((i + 1) * batch_size), :]
                yield out, out_onehot

        else:

            batches_count = int(np.ceil(self.get_size(dataset_type) / batch_size))
            for i in range(batches_count):
                out = np.int32(self.dataset[dataset_type][(i * batch_size):((i + 1) * batch_size), :])
                out_onehot = np.zeros(shape=[out.shape[0], len(self.ent_to_idx)], dtype=np.int32)

                for j, triple in enumerate(out):
                    key = (triple[0], triple[1])
                    if key in self.output_mapping.keys():
                        indices = self.output_mapping[(triple[0], triple[1])]
                        out_onehot[j, indices] = 1

                # logger.info('ada - batch {} out {} onehot {}'.format(i, out.shape, out_onehot.shape))

                yield out, out_onehot

    def get_next_eval_batch(self, batch_size=1, dataset_type='test'):
        """Generator that returns the next batch of data.
        
        Parameters
        ----------
        batch_size : int
            data size that needs to be returned
        dataset_type: string
            indicates which dataset to use
        Returns
        -------
        batch_output : nd-array
            yields a batch of triples from the dataset type specified
        """
        # if data is not already mapped, then map before returning the batch
        if not self.mapped_status[dataset_type]:
            self.map_data()

        if not self.low_memory:
            # If onehot outputs aren't initialized ..
            if dataset_type not in self.output_onehot.keys():
                self.generate_onehot_outputs(dataset_type, use_filter=False)

            batches_count = int(np.ceil(self.get_size(dataset_type) / batch_size))
            for i in range(batches_count):
                out = np.int32(self.dataset[dataset_type][(i * batch_size):((i + 1) * batch_size), :])
                out_onehot = self.output_onehot[dataset_type][(i * batch_size):((i + 1) * batch_size), :]
                yield out, out_onehot
        else:

            batches_count = int(np.ceil(self.get_size(dataset_type) / batch_size))
            for i in range(batches_count):
                out = np.int32(self.dataset[dataset_type][(i * batch_size):((i + 1) * batch_size), :])
                out_onehot = np.zeros(shape=[out.shape[0], len(self.ent_to_idx)], dtype=np.int8)

                for j, triple in enumerate(out):
                    indices = self.output_mapping[(triple[0], triple[1])]
                    out_onehot[j, indices] = 1

                yield out, out_onehot

    def get_next_batch_with_filter(self, batch_size=1, dataset_type='test'):
        """Generator that returns the next batch of filtered data.

        Parameters
        ----------
        batch_size : int
            data size that needs to be returned
        dataset_type: string
            indicates which dataset to use
        Returns
        -------
        batch_output : nd-array [n,3]
            yields a batch of triples from the dataset type specified
        participating_objects : nd-array [n,1]
            all objects that were involved in the s-p-? relation
        participating_subjects : nd-array [n,1]
            all subjects that were involved in the ?-p-o relation
        """

        if not self.mapped_status[dataset_type]:
            self.map_data()

        if self.filter_mapping is None:
            msg = 'Cannot use `get_next_batch_with_filter` if a filter has not been set. '
            raise ValueError(msg)

        if not self.low_memory:

            # If onehot outputs haven't been created for this dataset_type
            if dataset_type not in self.output_onehot.keys():
                # TODO: This could lead to situation where reusing dataset handler could lead to nonfiltered data
                # being returned from generator. The fix is to include a 'filter_mapped_status' class var
                self.generate_onehot_outputs(dataset_type, use_filter=True)

            batches_count = int(np.ceil(self.get_size(dataset_type) / batch_size))
            for i in range(batches_count):
                # generate the batch
                out = np.int32(self.dataset[dataset_type][(i * batch_size):((i + 1) * batch_size), :])
                out_filter = np.copy(self.output_onehot[dataset_type][(i * batch_size):((i + 1) * batch_size), :])
                # out_filter[:, out[:, 2]] = 0.0  # set positive index to 0 for filtering
                yield out, out_filter
        else:

            batches_count = int(np.ceil(self.get_size(dataset_type) / batch_size))
            for i in range(batches_count):
                out = np.int32(self.dataset[dataset_type][(i * batch_size):((i + 1) * batch_size), :])
                out_filter = np.zeros(shape=[out.shape[0], len(self.ent_to_idx)], dtype=np.int8)

                for j, triple in enumerate(out):
                    indices = self.output_mapping[(triple[0], triple[1])]
                    out_filter[j, indices] = 1

                # out_filter[:, out[:, 2]] = 0.0
                yield out, out_filter

    def _validate_data(self, data):
        """valiates the data
        """
        if type(data) != np.ndarray:
            msg = 'Invalid type for input data. Expected ndarray, got {}'.format(type(data))
            raise ValueError(msg)

        if (np.shape(data)[1]) != 3:
            msg = 'Invalid size for input data. Expected number of column 3, got {}'.format(np.shape(data)[1])
            raise ValueError(msg)

    def set_data(self, dataset, dataset_type=None, mapped_status=False):
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
