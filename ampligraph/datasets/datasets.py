import pandas as pd
import os
import numpy as np
import logging
import urllib
import zipfile
from pathlib import Path

AMPLIGRAPH_ENV_NAME = 'AMPLIGRAPH_DATA_HOME'
REMOTE_DATASET_SERVER = 'https://s3-eu-west-1.amazonaws.com/ampligraph/datasets/'
DATASET_FILE_NAME = {'WN18': 'wn18.zip',
                     'WN18RR': 'wn18RR.zip',
                     'FB15K': 'fb15k.zip',
                     'FB15K_237': 'fb15k-237.zip',
                     'YAGO3_10': 'YAGO3-10.zip',
                     }

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def _get_data_home(data_home=None):
    if data_home is None:
        data_home = os.environ.get(AMPLIGRAPH_ENV_NAME, os.path.join('~', 'ampligraph_datasets'))
    data_home = os.path.expanduser(data_home)
    if not os.path.exists(data_home):
        os.makedirs(data_home)
    logger.debug('data_home is set to {}'.format(data_home))
    return data_home


def _unzip_dataset(data_home, file_path):
    # TODO - add error checking
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        logger.debug('Unzipping {} to {}'.format(file_path, data_home))
        zip_ref.extractall(data_home)
    os.remove(file_path)


def _fetch_remote_data(url, dataset_dir, data_home):
    file_path = '{}.zip'.format(dataset_dir)
    if not Path(file_path).exists():
        urllib.request.urlretrieve(url, file_path)
        # TODO - add error checking
    _unzip_dataset(data_home, file_path)


def _fetch_dataset(dataset_name, data_home=None, url=None):
    data_home = _get_data_home(data_home)
    dataset_dir = os.path.join(data_home, dataset_name)
    if not os.path.exists(dataset_dir):
        if url is None:
            msg = 'No dataset at {} and no url provided.'.format(dataset_dir)
            logger.error(msg)
            raise Exception(msg)
        _fetch_remote_data(url, dataset_dir, data_home)
    return dataset_dir


def load_from_csv(directory_path, file_name, sep='\t', header=None):
    """Load a csv file
    
    Loads a knowledge graph serialized in a csv file as:
    .. code-block:: text

       subj1    relationX   obj1
       subj1    relationY   obj2
       subj3    relationZ   obj2
       subj4    relationY   obj2
       ...

        .. note::
            Duplicates are filtered.
    
    Parameters
    ----------
    
    folder_name: str
        base folder within AMPLIGRAPH_DATA_HOME where the file is stored.
    file_name : str
        file name
    sep : str
        The subject-predicate-object separator (default \t).
    header : int, None
        The row of the header of the csv file. Same as pandas.read_csv header param.
    
    Returns
    -------
    
    triples : ndarray , shape [n, 3]
        the actual triples of the file.
    
    Examples
    --------
    
    >>> from ampligraph.datasets import load_from_csv
    >>> X = load_from_csv('folder', 'dataset.csv', sep=',')
    >>> X[:3]
    array([['a', 'y', 'b'],
           ['b', 'y', 'a'],
           ['a', 'y', 'c']],
          dtype='<U1')

    """

    logger.debug('Loading data from {}.'.format(file_name))
    df = pd.read_csv(os.path.join(directory_path, file_name),
                     sep=sep,
                     header=header,
                     names=None,
                     dtype=str)
    logger.debug('Dropping duplicates.')
    df = df.drop_duplicates()
    return df.values


def load_dataset(dataset_name=None, url=None, data_home=None, train_name='train.txt', valid_name='valid.txt', test_name='test.txt'):
    if dataset_name is None:
        if url is None:
            raise ValueError('The dataset name or url must be provided to load a dataset.')
        dataset_name = url[url.rfind('/') + 1:url.rfind('.')]
    dataset_path = _fetch_dataset(dataset_name, data_home, url)
    train = load_from_csv(dataset_path, train_name)
    valid = load_from_csv(dataset_path, valid_name)
    test = load_from_csv(dataset_path, test_name)
    return {'train': train, 'valid': valid, 'test': test}

def _load_core_dataset(dataset_key,data_home=None):
    return load_dataset(url='{}{}'.format(REMOTE_DATASET_SERVER, DATASET_FILE_NAME[dataset_key]), data_home=data_home)

def load_wn18(data_home=None):
    """Load the WN18 dataset

        WN18 is a subset of Wordnet. It was first presented by :cite:`bordes2013translating`.
        The dataset is divided in three splits:

        - ``train``
        - ``valid``
        - ``test``

    Returns
    -------

    splits : dict
        The dataset splits {'train': train, 'valid': valid, 'test': test}. Each split is an ndarray of shape [n, 3].
    
    Examples
    --------
    >>> from ampligraph.datasets import load_wn18
    >>> X = load_wn18()
    >>> X['test'][:3]
    array([['06845599', '_member_of_domain_usage', '03754979'],
           ['00789448', '_verb_group', '01062739'],
           ['10217831', '_hyponym', '10682169']], dtype=object)

    """
    
    return  _load_core_dataset('WN18',data_home)


def load_wn18rr(data_home=None):
    """ Load the WN18RR dataset
    
    The dataset is described in :cite:`DettmersMS018`. It is divided in three splits:
        - ``train``
        - ``valid``        
        - ``test``
    
    Returns
    -------
    
    splits : dict
        The dataset splits: {'train': train, 'valid': valid, 'test': test}. Each split is an ndarray of shape [n, 3].
    
    Examples
    -------
    
    >>> from ampligraph.datasets import load_wn18rr
    >>> X = load_wn18rr()
    >>> X["valid"][0]
    array(['02174461', '_hypernym', '02176268'], dtype=object)
    
    """

    return  _load_core_dataset('WN18RR',data_home)


def load_fb15k(data_home=None):
    """Load the FB15k dataset

    FB15k is a split of Freebase, first proposed by :cite:`bordes2013translating`.

    The dataset is divided in three splits:
    
    - ``train``
    - ``valid``
    - ``test``
    
    Returns
    -------
    
    splits : dict
        The dataset splits: {'train': train, 'valid': valid, 'test': test}. Each split is an ndarray of shape [n, 3].
    
    Examples
    --------
    
    >>> from ampligraph.datasets import load_fb15k
    >>> X = load_fb15k()
    >>> X['test'][:3]
    array([['/m/01qscs',
            '/award/award_nominee/award_nominations./award/award_nomination/award',
            '/m/02x8n1n'],
           ['/m/040db', '/base/activism/activist/area_of_activism', '/m/0148d'],
           ['/m/08966',
            '/travel/travel_destination/climate./travel/travel_destination_monthly_climate/month',
            '/m/05lf_']], dtype=object)

    """

    return  _load_core_dataset('FB15K',data_home)


def load_fb15k_237(data_home=None):
    """Load the FB15k-237 dataset
    
    FB15k-237 is a reduced version of FB15k. It was first proposed by :cite:`toutanova2015representing`.
        The dataset is divided in three splits:
        - ``train``
        - ``valid``
        - ``test``
    
    Returns
    -------
    
    splits : dict
        The dataset splits: {'train': train, 'valid': valid, 'test': test}. Each split is an ndarray of shape [n, 3].
    
    Examples
    --------
    
    >>> from ampligraph.datasets import load_fb15k_237
    >>> X = load_fb15k_237()
    >>> X["train"][2]
    array(['/m/07s9rl0', '/media_common/netflix_genre/titles', '/m/0170z3'],
      dtype=object)
    """

    return  _load_core_dataset('FB15K_237',data_home)


def load_yago3_10(data_home=None):
    """ Load the YAGO3-10 dataset
    
    The dataset is presented in :cite:`mahdisoltani2013yago3`. It is divided in three splits:
        - ``train``
        - ``valid``        
        - ``test``
    
    Returns
    -------
    
    splits : dict
        The dataset splits: {'train': train, 'valid': valid, 'test': test}. Each split is an ndarray of shape [n, 3].
    
    Examples
    -------
    
    >>> from ampligraph.datasets import load_yago3_10
    >>> X = load_yago3_10()
    >>> X["valid"][0]
    array(['Mikheil_Khutsishvili', 'playsFor', 'FC_Merani_Tbilisi'], dtype=object)    
    
    """

    return  _load_core_dataset('YAGO3_10',data_home)

def load_all_datasets(data_home=None):
    load_wn18(data_home)
    load_wn18rr(data_home)
    load_fb15k(data_home)
    load_fb15k_237(data_home)
    load_yago3_10(data_home)

def load_from_rdf(folder_name, file_name, format='nt', data_home=None):
    """Load an RDF file

        Loads an RDF knowledge graph using rdflib APIs.
        The entire graph will be loaded in memory, and converted into an rdflib `Graph` object.


    Parameters
    ----------
    folder_name: str
        base folder within AMPLIGRAPH_DATA_HOME where the file is stored.
    file_name : str
        file name
    format : str
        The RDF serialization format (nt, ttl, rdf/xml - see rdflib documentation)

    Returns
    -------
        triples : ndarray , shape [n, 3]
            the actual triples of the file.
    """

    logger.debug('Loading rdf data from {}.'.format(file_name))
    data_home = _get_data_home(data_home)
    from rdflib import Graph
    g = Graph()
    g.parse(os.path.join(data_home, folder_name, file_name), format=format, publicID='http://test#')
    return np.array(g)


def load_from_ntriples(folder_name, file_name, data_home=None):
    """Load RDF ntriples as csv statements

        Loads an RDF knowledge graph serialized as ntriples, without building an RDF graph in mmeory.
        This function is faster than ``load_from_rdf()``.


    Parameters
    ----------
    folder_name: str
        base folder within AMPLIGRAPH_DATA_HOME where the file is stored.
    file_name : str
        file name

    Returns
    -------
        triples : ndarray , shape [n, 3]
            the actual triples of the file.
    """

    logger.debug('Loading rdf ntriples from {}.'.format(file_name))
    data_home = _get_data_home(data_home)
    df = pd.read_csv(os.path.join(data_home, folder_name, file_name),
                     sep=' ',
                     header=None,
                     names=None,
                     dtype=str,
                     usecols=[0, 1, 2])
    return df.as_matrix()
