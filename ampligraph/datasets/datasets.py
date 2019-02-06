import pandas as pd
import os
import numpy as np
from hdt import HDTDocument

AMPLIGRAPH_DATA_HOME = os.environ['AMPLIGRAPH_DATA_HOME']


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
    df = pd.read_csv(os.path.join(AMPLIGRAPH_DATA_HOME, folder_name, file_name),
                     sep=sep,
                     header=header,
                     names=None,
                     dtype=str)

    df = df.drop_duplicates()
    return df.as_matrix()

def load_fb15k_sample():
    """Load sample of Freebase dataset (FB15k-237).
    This is a reduced version of FB15k.

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

    """
    train = load_from_csv('fb15K-sample', 'train.txt')
    valid = load_from_csv('fb15K-sample', 'valid.txt')
    test = load_from_csv('fb15K-sample', 'test.txt')
    return {'train': train, 'valid': valid, 'test': test}


def load_wn11():
    """Load the Wordnet11 Dataset.

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

    """
    train = load_from_csv('wordnet11', 'train.txt')
    valid = load_from_csv('wordnet11', 'dev.txt')
    test = load_from_csv('wordnet11', 'test.txt')
    return {'train': train, 'valid': valid, 'test': test}


def load_fb13():
    """Load the Freebase13 Dataset.

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

    """
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

    from rdflib import Graph
    g = Graph()
    g.parse(os.path.join(AMPLIGRAPH_DATA_HOME, folder_name, file_name), format=format, publicID='http://test#')
    return np.array(g)


def load_from_ntriples(folder_name, file_name):
    """Load RDF ntriples

        Loads an RDF knowledge graph serialized as ntriples
        This function does not use rdflib APIs.


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
    df = pd.read_csv(os.path.join(AMPLIGRAPH_DATA_HOME, folder_name, file_name),
                     sep=' ',
                     header=None,
                     names=None,
                     dtype=str,
                     usecols=[0, 1, 2])
    # df = df.drop_duplicates()
    return df.as_matrix()


def load_dbpedia_1k_s():
    """Load a sample of DBpedia (DBpedia-1k-S).

        The dataset is split to evalaute a relational model with unseen entities.
        Splits:

        - ``train``
        - ``valid``
        - ``aux``
        - ``test``

    Returns
    -------

    splits : dict
        The dataset splits {'train': train, 'valid': valid, 'aux': aux,'test': test}.
        Each split is an ndarray of shape [n, 3].

    Examples
    --------
    >>> from ampligraph.datasets import load_dbpedia_1k_s
    >>> X = load_dbpedia_1k_s()
    >>> X['test'][:3]
    array([["<http://dbpedia.org/resource/Take_a_Chance_(Stockton's_Wing_album)>",
            '<http://dbpedia.org/ontology/recordLabel>',
            '<http://dbpedia.org/resource/Tara_Music_label>'],
           ['<http://dbpedia.org/resource/Novel_Nature>',
            '<http://dbpedia.org/ontology/hometown>',
            '<http://dbpedia.org/resource/Washington_(state)>'],
           ['<http://dbpedia.org/resource/Back_Home_Tour>',
            '<http://dbpedia.org/ontology/recordLabel>',
            '<http://dbpedia.org/resource/Sony_BMG>']], dtype=object)


    """
    train = load_from_ntriples('dbpedia_1k_s', 'training_1000.nt')
    valid = load_from_ntriples('dbpedia_1k_s', 'validation_1000.nt')
    aux = load_from_ntriples('dbpedia_1k_s', 'auxiliary_1000.nt')
    test = load_from_ntriples('dbpedia_1k_s', 'test_1000.nt')

    return {'train': train, 'valid': valid, 'aux': aux, 'test': test}


def load_wn18():
    """Load sample of Wordnet dataset (WN18).


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
    train = load_from_csv('wn18', 'train.txt')
    valid = load_from_csv('wn18', 'valid.txt')
    test = load_from_csv('wn18', 'test.txt')

    return {'train': train, 'valid': valid, 'test': test}


def load_fb15k():
    """Load sample of Freebase dataset (FB15k).

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
    train = load_from_csv('fb15k', 'train.txt')
    valid = load_from_csv('fb15k', 'valid.txt')
    test = load_from_csv('fb15k', 'test.txt')

    return {'train': train, 'valid': valid, 'test': test}


def load_fb15k_237():
    """Load sample of Freebase dataset (FB15k-237).
    This is a reduced version of FB15k.

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
    train = load_from_csv('fb15k-237', 'train.txt')
    valid = load_from_csv('fb15k-237', 'valid.txt')
    test = load_from_csv('fb15k-237', 'test.txt')
    return {'train': train, 'valid': valid, 'test': test}

def load_from_hdt(folder_name, file_name):
    """ Loads a knowledge graph serialized in the HDT format

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

    Examples
    --------
    >>> from ampligraph.datasets import load_from_hdt
    >>> X = load_from_hdt('wdf', 'swdf-2012-11-28.hdt')
    >>> X[:3]
    array([['_:b1', 'http://www.w3.org/1999/02/22-rdf-syntax-ns#_1',
        'http://data.semanticweb.org/person/barry-norton'],
       ['_:b1', 'http://www.w3.org/1999/02/22-rdf-syntax-ns#_2',
        'http://data.semanticweb.org/person/reto-krummenacher'],
       ['_:b1', 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type',
        'http://www.w3.org/1999/02/22-rdf-syntax-ns#Seq']], dtype=object)



    """

    f = os.path.join(AMPLIGRAPH_DATA_HOME,folder_name, file_name)

    document = HDTDocument(f)
 
    (triples, cardinivity) = document.search_triples("", "", "")

    arr = np.empty((cardinivity, 3), dtype='object')

    tmp = []

    for triple in triples:
        tmp.append(list(triple))
    arr[::] = tmp[::]
    
    return arr

def load_wdf():
    arr = load_from_hdt('wdf', 'swdf-2012-11-28.hdt')
    return {'shape': arr.shape}


def aux_load_ICEWS(foldername, filename):
    df = pd.read_csv(os.path.join(AMPLIGRAPH_DATA_HOME, foldername, filename),
                     sep='\t',
                     header=None,
                     dtype=float)

    df = df.drop_duplicates()
    df= df.drop(4, axis= 1)
    df=delet_self_quote(df)
    return df.as_matrix()

def delet_self_quote(df):
    n=df.shape[0]
    m=df.as_matrix()
    to_drop=[]
    for i in range(n) : 
        if m[i,0]==m[i,2]:
            to_drop.append(i)
    df=df.drop(to_drop, axis=0)
    return df


def load_ICEWS( ):
    """ Loads the ICEWS dataset

    Loads the ICEWS dataset described in :cite:`trivedi2017know`.

    The dataset is divided in two splits:

        - ``train``
        - ``test``

    Returns
    -------

    splits : dict
        The dataset splits: {'train': train, 'test': test}. Each split is an ndarray of shape [n, 4].
    """

    train=aux_load_ICEWS("ICEWS", "train.txt")
    test=aux_load_ICEWS("ICEWS", "test.txt")
    return {'train': train,  'test': test}

def load_ICEWS_reduced( ):
    """ Loads the ICEWS-500 dataset

    Loads the ICEWS-500 dataset described in :cite:`trivedi2017know`.

    The dataset is divided in two splits:

        - ``train``
        - ``test``

    Returns
    -------

    splits : dict
        The dataset splits: {'train': train, 'test': test}. Each split is an ndarray of shape [n, 4].
    """

    train=aux_load_ICEWS("ICEWS_500", "train_500.txt")
    test=aux_load_ICEWS("ICEWS_500", "test_500.txt")
    return {'train': train,  'test': test}


