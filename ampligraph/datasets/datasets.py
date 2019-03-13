import pandas as pd
import os
import numpy as np
import logging
import urllib
import zipfile
from pathlib import Path

AMPLIGRAPH_DATA_HOME = os.environ['AMPLIGRAPH_DATA_HOME']
AMPLIGRAPH_ENV_NAME = 'AMPLIGRAPH_DATA_HOME'

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def _get_data_home(data_home=None):
    if data_home is None:
        data_home = os.get(AMPLIGRAPH_ENV_NAME,os.path.join('~','ampligraph_dataset'))
    data_home = os.path.expanduser(data_home)
    if not os.path.exists(data_home):
        os.makedirs(data_home)
    return data_home 

def _unzip_dataset(dataset_dir, file_name):
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    #TODO - add error checking
    zip_ref = zipFile.ZipFile('{}.zip'.format(file_name),'r')
    zip_ref.extractall(dataset_dir)
    zip_ref.close()

def _fetch_remote_data(url,dataset_dir,data_home):
    file_name = '{}.zip'.format(dataset_dir)
    if not Path(file_name).exists():
        urllib.request.urlretrieve(url,file_name)
        #TODO - add error checking
    _unzip_dataset(dataset_dir,file_name)
    
def _fetch_dataset(dataset_name,data_home=None,url=None):
    #This checks if dataset already exits
    data_home = _get_data_home(data_home)
    dataset_dir = os.path.join(data_home,dataset_name)
    if not os.path.exists(dataset_dir):
        if url is None:
            raise Exception('No dataset at {} and no url provided.'.format(local_path))
        _fetch_remote_data(url,dataset_dir,data_home)
    return dataset_dir

def load_dataset(dataset_name, url, data_home=None, train_name='train.txt', valid_name='valid.txt', test_name='test.txt'):
    local_path = _fetch_dataset(dataset_name, data_home, url)
    train = load_from_csv(local_path, train_name)
    valid = load_from_csv(local_path, valid_name)
    test = load_from_csv(local_path, test_name)
    return {'train':train,'valid':valid,'test':test}

def load_wn18(data_home=None):
    return load_dataset('wn18','https://drive.google.com/drive/folders/16GBu89NCVyyYetry91tMntzpV_mSQ-gK/wn18.zip',data_home)


def load_from_csv(folder_name, file_name, sep='\t', header=None):
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
    df = pd.read_csv(os.path.join(AMPLIGRAPH_DATA_HOME, folder_name, file_name),
                     sep=sep,
                     header=header,
                     names=None,
                     dtype=str)
    logger.debug('Dropping duplicates.')
    df = df.drop_duplicates()
    return df.as_matrix()


def load_wn11():
    """Load the WN11 Dataset
        
        The dataset is divided in three splits:

        - ``train``
        - ``valid``
        - ``test``

    .. note:: The function filters duplicates.
        :cite:`Hamaguchi2017` reports unfiltered numbers.

    Returns
    -------

    splits : dict
        The dataset splits: {'train': train, 'valid': valid, 'test': test}. Each split is an ndarray of shape [n, 3].

    Examples
    --------
    >>> from ampligraph.datasets import load_wn11
    >>> X = load_wn11()
    >>> print("X[valid'][0]: ", X['valid'][0])
    X[valid'][0]:  ['__genus_xylomelum_1' '_type_of' '__dicot_genus_1' '1']


    """

    logger.debug('Loading wn11 data.')
    train = load_from_csv('wordnet11', 'train.txt')
    valid = load_from_csv('wordnet11', 'dev.txt')
    test = load_from_csv('wordnet11', 'test.txt')
    return {'train': train, 'valid': valid, 'test': test}


def load_fb13():
    """Load the FB13 Dataset

        The dataset is divided in three splits:

        - ``train``
        - ``valid``
        - ``test``

    .. note:: The function filters duplicates.
        :cite:`Hamaguchi2017` reports unfiltered numbers.

    Returns
    -------

    splits : dict
        The dataset splits: {'train': train, 'valid': valid, 'test': test}. Each split is an ndarray of shape [n, 3].

    Examples
    --------
    >>> from ampligraph.datasets import load_fb13
    >>> X = load_fb13()
    >>> print("X['valid'][0]: ", X['valid'][0])
    X['valid'][0]:  ['cornelie_van_zanten' 'gender' 'female' '1']


    """

    logger.debug('Loading freebase13 data.')
    train = load_from_csv('freebase13', 'train.txt')
    valid = load_from_csv('freebase13', 'dev.txt')
    test = load_from_csv('freebase13', 'test.txt')
    return {'train': train, 'valid': valid, 'test': test}


def load_from_rdf(folder_name, file_name, format='nt'):
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
    from rdflib import Graph
    g = Graph()
    g.parse(os.path.join(AMPLIGRAPH_DATA_HOME, folder_name, file_name), format=format, publicID='http://test#')
    return np.array(g)


def load_from_ntriples(folder_name, file_name):
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
    df = pd.read_csv(os.path.join(AMPLIGRAPH_DATA_HOME, folder_name, file_name),
                     sep=' ',
                     header=None,
                     names=None,
                     dtype=str,
                     usecols=[0, 1, 2])
    # df = df.drop_duplicates()
    return df.as_matrix()

#def load_wn18():
#    """Load the WN18 dataset
#
#        The dataset is divided in three splits:
#
#        - ``train``
#        - ``valid``
#        - ``test``
#
#    Returns
#    -------
#
#    splits : dict
#        The dataset splits {'train': train, 'valid': valid, 'test': test}. Each split is an ndarray of shape [n, 3].
#
#    Examples
#    --------
#    >>> from ampligraph.datasets import load_wn18
#    >>> X = load_wn18()
#    >>> X['test'][:3]
#    array([['06845599', '_member_of_domain_usage', '03754979'],
#           ['00789448', '_verb_group', '01062739'],
#           ['10217831', '_hyponym', '10682169']], dtype=object)
#
#    """
#
#    logger.debug('Loading wordnet WN18')
#    train = load_from_csv('wn18', 'train.txt')
#    valid = load_from_csv('wn18', 'valid.txt')
#    test = load_from_csv('wn18', 'test.txt')
#
#    return {'train': train, 'valid': valid, 'test': test}
#

def load_fb15k():
    """Load the FB15k dataset

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

    logger.debug('Loading Freebase FB15k')
    train = load_from_csv('fb15k', 'train.txt')
    valid = load_from_csv('fb15k', 'valid.txt')
    test = load_from_csv('fb15k', 'test.txt')

    return {'train': train, 'valid': valid, 'test': test}


def load_fb15k_237():
    """Load the FB15k-237 dataset
    
    FB15k-237 is a reduced version of FB15k.

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

    """

    logger.debug('Loading sample of Freebase FB15k-237')
    train = load_from_csv('fb15k-237', 'train.txt')
    valid = load_from_csv('fb15k-237', 'valid.txt')
    test = load_from_csv('fb15k-237', 'test.txt')
    return {'train': train, 'valid': valid, 'test': test}


def load_yago3_10():
    """ Load the YAGO3-10 dataset
    
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
    >>> from ampligraph.datasets import load_yago3_10
    >>> X = load_yago3_10()
    >>> X["valid"][0]
    array(['Mikheil_Khutsishvili', 'playsFor', 'FC_Merani_Tbilisi'], dtype=object)    



    """

    logger.debug('Loading YAGO3-10.')
    train=load_from_csv("YAGO3-10", "train.txt", sep="\t")
    test=load_from_csv("YAGO3-10", "test.txt", sep="\t")
    valid=load_from_csv("YAGO3-10", "valid.txt", sep="\t")
    
    return {"train": train,  "test": test, "valid": valid}

def load_wn18rr():
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

    logger.debug('Loading WN18RR.')
    train=load_from_csv("wn18rr", "train.txt", sep="\t")
    test=load_from_csv("wn18rr", "test.txt", sep="\t")
    valid=load_from_csv("wn18rr", "valid.txt", sep="\t")
    
    return {"train": train,  "test": test, "valid": valid}    

