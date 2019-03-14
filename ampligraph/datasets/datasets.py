import pandas as pd
import os
import numpy as np
import logging
import urllib
import zipfile
from pathlib import Path

AMPLIGRAPH_ENV_NAME = 'AMPLIGRAPH_DATA_HOME'
REMOTE_DATASET_SERVER = 'https://s3-eu-west-1.amazonaws.com/ampligraph/datasets/'
DATASET_FILE_NAME = {'WN18':'wn18.zip',
'WN18RR':'wn18RR.zip',
'FB15K':'fb15k.zip',
'FB15K_237':'fb15k-237.zip',
'YAGO3_10':'YAGO3-10.zip',
'FB13':'freebase13.zip',
'WN11':'wordnet11.zip'
}


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def _get_data_home(data_home=None):
    if data_home is None:
        data_home = os.environ.get(AMPLIGRAPH_ENV_NAME,os.path.join('~','ampligraph_dataset'))
    data_home = os.path.expanduser(data_home)
    if not os.path.exists(data_home):
        os.makedirs(data_home)
    logger.debug('data_home is set to {}'.format(data_home))
    return data_home 

def _unzip_dataset(data_home, file_path):
    #TODO - add error checking
    with zipfile.ZipFile(file_path,'r') as zip_ref:
        logger.debug('Unzipping {} to {}'.format(file_path, data_home))
        zip_ref.extractall(data_home)
    os.remove(file_path)

def _fetch_remote_data(url,dataset_dir,data_home):
    file_path = '{}.zip'.format(dataset_dir)
    if not Path(file_path).exists():
        urllib.request.urlretrieve(url,file_path)
        #TODO - add error checking
    _unzip_dataset(data_home,file_path)
    
def _fetch_dataset(dataset_name,data_home=None,url=None):
    data_home = _get_data_home(data_home)
    dataset_dir = os.path.join(data_home,dataset_name)
    if not os.path.exists(dataset_dir):
        if url is None:
            msg = 'No dataset at {} and no url provided.'.format(local_path)
            logger.error(msg)
            raise Exception(msg)
        _fetch_remote_data(url,dataset_dir,data_home)
    return dataset_dir

def load_from_csv(directory_path,file_name, sep='\t', header=None):
    logger.debug('Loading data from {}.'.format(file_name))
    df = pd.read_csv(os.path.join(directory_path, file_name),
                     sep=sep,
                     header=header,
                     names=None,
                     dtype=str)
    logger.debug('Dropping duplicates.')
    df = df.drop_duplicates()
    return df.values

def load_dataset(url, data_home=None, train_name='train.txt', valid_name='valid.txt', test_name='test.txt'):
    dataset_name = url[url.rfind('/')+1:url.rfind('.')]
    dataset_path = _fetch_dataset(dataset_name, data_home, url)
    train = load_from_csv(dataset_path, train_name)
    valid = load_from_csv(dataset_path, valid_name)
    test = load_from_csv(dataset_path, test_name)
    return {'train':train,'valid':valid,'test':test}

def load_wn18(data_home=None):
    return load_dataset( '{}{}'.format(REMOTE_DATASET_SERVER,DATASET_FILE_NAME['WN18']), data_home)

def load_wn18rr(data_home=None):
    return load_dataset( '{}{}'.format(REMOTE_DATASET_SERVER,DATASET_FILE_NAME['WN18RR']), data_home)

def load_fb15k(data_home=None):
    return load_dataset( '{}{}'.format(REMOTE_DATASET_SERVER,DATASET_FILE_NAME['FB15K']), data_home)

def load_fb15k_237(data_home=None):
    return load_dataset( '{}{}'.format(REMOTE_DATASET_SERVER,DATASET_FILE_NAME['FB15K_237']), data_home)

def load_yago3_10(data_home=None):
    return load_dataset( '{}{}'.format(REMOTE_DATASET_SERVER,DATASET_FILE_NAME['YAGO3_10']), data_home)

def load_fb13(data_home=None):
    #return load_dataset( '{}{}'.format(REMOTE_DATASET_SERVER,DATASET_FILE_NAME['FB13']), data_home)
    msg = 'Currently not supported due to filename name error. Blocked by issue #50'
    logger.error(msg)
    raise NotImplementedError(msg)

def load_wn11(data_home=None):
    #return load_dataset( '{}{}'.format(REMOTE_DATASET_SERVER,DATASET_FILE_NAME['WN11']), data_home)
    msg = 'Currently not supported due to filename name error. Blocked by issue #50'
    logger.error(msg)
    raise NotImplementedError(msg)

def load_all_datasets(data_home=None):
    load_wn18(data_home)
    load_wn18rr(data_home)
    load_fb15k(data_home)
    load_fb15k_237(data_home)
    load_yago3_10(data_home)
    #load_fb13(data_home)
    #load_wn11(data_home)


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
