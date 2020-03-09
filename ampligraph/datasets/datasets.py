# Copyright 2019 The AmpliGraph Authors. All Rights Reserved.
#
# This file is Licensed under the Apache License, Version 2.0.
# A copy of the Licence is available in LICENCE, or at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
import pandas as pd
import os
import numpy as np
import logging
import urllib
import zipfile
from pathlib import Path
import hashlib
from collections import namedtuple

AMPLIGRAPH_ENV_NAME = 'AMPLIGRAPH_DATA_HOME'

DatasetMetadata = namedtuple('DatasetMetadata', ['dataset_name', 'filename', 'url', 'train_name', 'valid_name',
                                                 'test_name', 'train_checksum', 'valid_checksum', 'test_checksum'])

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def _clean_data(X, return_idx=False):
    """
    Clean dataset X by removing unseen entities and relations from valid and test sets.

    Parameters
    ----------
    X: dict
        Dicionary containing the following keys: train, valid, test.
        Each key should contain an ndarray of shape [n, 3].

    return_idx: bool
        Whether to return the indices of the remaining rows in valid and test respectively.

    Returns
    -------
    filtered_X: dict
        Dicionary containing the following keys: train, valid, test.
        Each key contains an ndarray of shape [n, 3].
        Valid and test do not contain entities or relations that are not present in train.

    valid_idx: ndarray
        Indices of the remaining rows of the valid dataset (with respect to the original valid ndarray).

    test_idx: ndarray
        Indices of the remaining rows of the test dataset (with respect to the original test ndarray).

    """
    train = pd.DataFrame(X["train"], columns=['s', 'p', 'o'])
    valid = pd.DataFrame(X["valid"], columns=['s', 'p', 'o'])
    test = pd.DataFrame(X["test"], columns=['s', 'p', 'o'])

    train_ent = np.unique(np.concatenate((train.s, train.o)))
    train_rel = train.p.unique()

    valid_idx = valid.s.isin(train_ent) & valid.o.isin(train_ent) & valid.p.isin(train_rel)
    test_idx = test.s.isin(train_ent) & test.o.isin(train_ent) & test.p.isin(train_rel)

    filtered_valid = valid[valid_idx].values
    filtered_test = test[test_idx].values

    filtered_X = {'train': train.values, 'valid': filtered_valid, 'test': filtered_test}

    if return_idx:
        return filtered_X, valid_idx, test_idx
    else:
        return filtered_X


def _get_data_home(data_home=None):
    """Get to location of the dataset folder to use.

    Automatically determine the dataset folder to use.
    If data_home is provided this location a check is
    performed to see if the path exists and creates one if it does not.
    If data_home is None the AMPLIGRAPH_ENV_NAME dataset is used.
    If AMPLIGRAPH_ENV_NAME is not set the a default environment ``~/ampligraph_datasets`` is used.

    Parameters
    ----------

    data_home : str
       The path to the folder that contains the datasets.

    Returns
    -------

    str
        The path to the dataset directory

    """

    if data_home is None:
        data_home = os.environ.get(AMPLIGRAPH_ENV_NAME, os.path.join('~', 'ampligraph_datasets'))
    data_home = os.path.expanduser(data_home)
    if not os.path.exists(data_home):
        os.makedirs(data_home)
    logger.debug('data_home is set to {}'.format(data_home))
    return data_home


def _md5(file_path):
    md5hash = hashlib.md5()
    chunk_size = 4096
    with open(file_path, 'rb') as f:
        content_buffer = f.read(chunk_size)
        while content_buffer:
            md5hash.update(content_buffer)
            content_buffer = f.read(chunk_size)
    return md5hash.hexdigest()


def _unzip_dataset(remote, source, destination, check_md5hash=False):
    """Unzip a file from a source location to a destination.

    Parameters
    ----------

    source : str
        The path to the zipped file
    destination : str
        The destination directory to unzip the files to.

    """

    # TODO - add error checking
    with zipfile.ZipFile(source, 'r') as zip_ref:
        logger.debug('Unzipping {} to {}'.format(source, destination))
        zip_ref.extractall(destination)
    if check_md5hash:
        for file_name, remote_checksum in [[remote.train_name, remote.train_checksum],
                                           [remote.valid_name, remote.valid_checksum],
                                           [remote.test_name, remote.test_checksum]]:
            file_path = os.path.join(destination, remote.dataset_name, file_name)
            checksum = _md5(file_path)
            if checksum != remote_checksum:
                os.remove(source)
                msg = '{} has an md5 checksum of ({}) which is different from the expected ({}), ' \
                      'the file may be corrupted.'.format(file_path, checksum, remote_checksum)
                logger.error(msg)
                raise IOError(msg)
    os.remove(source)


def _fetch_remote_data(remote, download_dir, data_home, check_md5hash=False):
    """Download a remote datasets.

    Parameters
    ----------

    remote : DatasetMetadata
        Named tuple containing remote datasets meta information: dataset name, dataset filename,
        url, train filename, validation filename, test filename, train checksum, valid checksum, test checksum.
    download_dir : str
        The location to download the file to.
    data_home : str
        The location to save the dataset.
    check_md5hash : bool
        Whether to check the MD5 hash of the dataset file.

    """

    file_path = '{}.zip'.format(download_dir)
    if not Path(file_path).exists():
        urllib.request.urlretrieve(remote.url, file_path)
        # TODO - add error checking
    _unzip_dataset(remote, file_path, data_home, check_md5hash)


def _fetch_dataset(remote, data_home=None, check_md5hash=False):
    """Get a dataset.

    Gets the directory of a dataset. If the dataset is not found it is downloaded automatically.

    Parameters
    ----------

    remote : DatasetMetadata
        Named tuple containing remote datasets meta information: dataset name, dataset filename,
        url, train filename, validation filename, test filename, train checksum, valid checksum, test checksum.
    data_home : str
        The location to save the dataset to.
    check_md5hash : bool
        Whether to check the MD5 hash of the dataset file.

    Returns
    ------

    str
        The location of the dataset.
    """
    data_home = _get_data_home(data_home)
    dataset_dir = os.path.join(data_home, remote.dataset_name)
    if not os.path.exists(dataset_dir):
        if remote.url is None:
            msg = 'No dataset at {} and no url provided.'.format(dataset_dir)
            logger.error(msg)
            raise Exception(msg)

        _fetch_remote_data(remote, dataset_dir, data_home, check_md5hash)
    return dataset_dir


def _add_reciprocal_relations(triples_df):
    """Add reciprocal relations to the triples

    Parameters
    ----------

    triples_df : Dataframe
        Dataframe of triples

    Returns
    -------
    triples_df : Dataframe
        Dataframe of triples and their reciprocals
    """
    # create a copy of the original triples to add reciprocal relations
    df_reciprocal = triples_df.copy()

    # swap subjects and objects
    cols = list(df_reciprocal.columns)
    cols[0], cols[2] = cols[2], cols[0]
    df_reciprocal.columns = cols

    # add reciprocal relations
    df_reciprocal.iloc[:, 1] = df_reciprocal.iloc[:, 1] + "_reciprocal"

    # append to original triples
    triples_df = triples_df.append(df_reciprocal)
    return triples_df


def load_from_csv(directory_path, file_name, sep='\t', header=None, add_reciprocal_rels=False):
    """Load a knowledge graph from a csv file

    Loads a knowledge graph serialized in a csv file as:

    .. code-block:: text

       subj1    relationX   obj1
       subj1    relationY   obj2
       subj3    relationZ   obj2
       subj4    relationY   obj2
       ...

    .. note::
        The function filters duplicated statements.

    .. note::
        It is recommended to use :meth:`ampligraph.evaluation.train_test_split_no_unseen` to split custom
        knowledge graphs into train, validation, and test sets. Using this function will lead to validation, test sets
        that do not include triples with entities that do not occur in the training set.


    Parameters
    ----------

    directory_path: str
        Folder where the input file is stored.
    file_name : str
        File name.
    sep : str
        The subject-predicate-object separator (default \t).
    header : int, None
        The row of the header of the csv file. Same as pandas.read_csv header param.
    add_reciprocal_rels : bool
        Flag which specifies whether to add reciprocal relations. For every <s, p, o> in the dataset
        this creates a corresponding triple with reciprocal relation <o, p_reciprocal, s>. (default: False)


    Returns
    -------

    triples : ndarray , shape [n, 3]
        The actual triples of the file.

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
    if add_reciprocal_rels:
        df = _add_reciprocal_relations(df)

    return df.values


def _load_dataset(dataset_metadata, data_home=None, check_md5hash=False, add_reciprocal_rels=False):
    """Load a dataset from the details provided.

    DatasetMetadata = namedtuple('DatasetMetadata', ['dataset_name', 'filename', 'url', 'train_name', 'valid_name',
                                                     'test_name', 'train_checksum', 'valid_checksum', 'test_checksum'])

    Parameters
    ----------
    dataset_metadata : DatasetMetadata
        Named tuple containing remote datasets meta information: dataset name, dataset filename,
        url, train filename, validation filename, test filename, train checksum, valid checksum, test checksum.

    data_home : str
        The location to save the dataset to (default: None).

    check_md5hash : boolean
        If True, check the md5hash of the files after they are downloaded (default: False).

    add_reciprocal_rels : bool
        Flag which specifies whether to add reciprocal relations. For every <s, p, o> in the dataset
        this creates a corresponding triple with reciprocal relation <o, p_reciprocal, s>. (default: False).
    """

    if dataset_metadata.dataset_name is None:
        if dataset_metadata.url is None:
            raise ValueError('The dataset name or url must be provided to load a dataset.')
        dataset_metadata.dataset_name = dataset_metadata.url[dataset_metadata.url.rfind('/') + 1:dataset_metadata
                                                             .url.rfind('.')]
    dataset_path = _fetch_dataset(dataset_metadata, data_home, check_md5hash)

    train = load_from_csv(dataset_path,
                          dataset_metadata.train_name,
                          add_reciprocal_rels=add_reciprocal_rels)
    valid = load_from_csv(dataset_path,
                          dataset_metadata.valid_name,
                          add_reciprocal_rels=add_reciprocal_rels)
    test = load_from_csv(dataset_path,
                         dataset_metadata.test_name,
                         add_reciprocal_rels=add_reciprocal_rels)

    return {'train': train, 'valid': valid, 'test': test}


def load_wn18(check_md5hash=False, add_reciprocal_rels=False):
    """Load the WN18 dataset

    .. warning::
        The dataset includes a large number of inverse relations that spilled to the test set, and its use in
        experiments has been deprecated. Use WN18RR instead.

    WN18 is a subset of Wordnet. It was first presented by :cite:`bordes2013translating`.

    The WN18 dataset is loaded from file if it exists at the ``AMPLIGRAPH_DATA_HOME`` location.
    If ``AMPLIGRAPH_DATA_HOME`` is not set the the default  ``~/ampligraph_datasets`` is checked.

    If the dataset is not found at either location it is downloaded and placed in ``AMPLIGRAPH_DATA_HOME``
    or ``~/ampligraph_datasets``.

    The dataset is divided in three splits:

    - ``train``: 141,442 triples
    - ``valid`` 5,000 triples
    - ``test`` 5,000 triples

    ========= ========= ======= ======= ============ ===========
     Dataset  Train     Valid   Test    Entities     Relations
    ========= ========= ======= ======= ============ ===========
    WN18      141,442   5,000   5,000   40,943        18
    ========= ========= ======= ======= ============ ===========

    Parameters
    ----------
    check_md5hash : bool
        If ``True`` check the md5hash of the files. Defaults to ``False``.

    add_reciprocal_rels : bool
        Flag which specifies whether to add reciprocal relations. For every <s, p, o> in the dataset
        this creates a corresponding triple with reciprocal relation <o, p_reciprocal, s>. (default: False).

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

    wn18 = DatasetMetadata(
        dataset_name='wn18',
        filename='wn18.zip',
        url='https://s3-eu-west-1.amazonaws.com/ampligraph/datasets/wn18.zip',
        train_name='train.txt',
        valid_name='valid.txt',
        test_name='test.txt',
        train_checksum='7d68324d293837ac165c3441a6c8b0eb',
        valid_checksum='f4f66fec0ca83b5ebe7ad7003404e61d',
        test_checksum='b035247a8916c7ec3443fa949e1ff02c'
    )

    return _load_dataset(wn18,
                         data_home=None,
                         check_md5hash=check_md5hash,
                         add_reciprocal_rels=add_reciprocal_rels)


def load_wn18rr(check_md5hash=False, clean_unseen=True, add_reciprocal_rels=False):
    """Load the WN18RR dataset

    The dataset is described in :cite:`DettmersMS018`.

    The WN18RR dataset is loaded from file if it exists at the ``AMPLIGRAPH_DATA_HOME`` location.
    If ``AMPLIGRAPH_DATA_HOME`` is not set the the default  ``~/ampligraph_datasets`` is checked.

    If the dataset is not found at either location it is downloaded and placed in ``AMPLIGRAPH_DATA_HOME``
    or ``~/ampligraph_datasets``.


    It is divided in three splits:

    - ``train``
    - ``valid``
    - ``test``

    ========= ========= ======= ======= ============ ===========
     Dataset  Train     Valid   Test    Entities     Relations
    ========= ========= ======= ======= ============ ===========
    WN18RR    86,835    3,034   3,134   40,943        11
    ========= ========= ======= ======= ============ ===========

    .. warning:: WN18RR's validation set contains 198 unseen entities over 210 triples.
        The test set has 209 unseen entities, distributed over 210 triples.

    Parameters
    ----------
    clean_unseen : bool
        If ``True``, filters triples in validation and test sets that include entities not present in the training set.

    check_md5hash : bool
        If ``True`` check the md5hash of the datset files. Defaults to ``False``.

    add_reciprocal_rels : bool
        Flag which specifies whether to add reciprocal relations. For every <s, p, o> in the dataset
        this creates a corresponding triple with reciprocal relation <o, p_reciprocal, s>. (default: False).

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

    wn18rr = DatasetMetadata(
        dataset_name='wn18RR',
        filename='wn18RR.zip',
        url='https://s3-eu-west-1.amazonaws.com/ampligraph/datasets/wn18RR.zip',
        train_name='train.txt',
        valid_name='valid.txt',
        test_name='test.txt',
        train_checksum='35e81af3ae233327c52a87f23b30ad3c',
        valid_checksum='74a2ee9eca9a8d31f1a7d4d95b5e0887',
        test_checksum='2b45ba1ba436b9d4ff27f1d3511224c9'
    )

    if clean_unseen:
        return _clean_data(_load_dataset(wn18rr,
                                         data_home=None,
                                         check_md5hash=check_md5hash,
                                         add_reciprocal_rels=add_reciprocal_rels))
    else:
        return _load_dataset(wn18rr,
                             data_home=None,
                             check_md5hash=check_md5hash,
                             add_reciprocal_rels=add_reciprocal_rels)


def load_fb15k(check_md5hash=False, add_reciprocal_rels=False):
    """Load the FB15k dataset

    .. warning::
        The dataset includes a large number of inverse relations that spilled to the test set, and its use in
        experiments has been deprecated. Use FB15k-237 instead.

    FB15k is a split of Freebase, first proposed by :cite:`bordes2013translating`.

    The FB15k dataset is loaded from file if it exists at the ``AMPLIGRAPH_DATA_HOME`` location.
    If ``AMPLIGRAPH_DATA_HOME`` is not set the the default  ``~/ampligraph_datasets`` is checked.

    If the dataset is not found at either location it is downloaded and placed in ``AMPLIGRAPH_DATA_HOME``
    or ``~/ampligraph_datasets``.

    The dataset is divided in three splits:

    - ``train``
    - ``valid``
    - ``test``

    ========= ========= ======= ======= ============ ===========
     Dataset  Train     Valid   Test    Entities     Relations
    ========= ========= ======= ======= ============ ===========
    FB15K     483,142   50,000  59,071  14,951        1,345
    ========= ========= ======= ======= ============ ===========

    Parameters
    ----------
    check_md5hash : boolean
        If ``True`` check the md5hash of the files. Defaults to ``False``.

    add_reciprocal_rels : bool
        Flag which specifies whether to add reciprocal relations. For every <s, p, o> in the dataset
        this creates a corresponding triple with reciprocal relation <o, p_reciprocal, s>. (default: False).


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

    FB15K = DatasetMetadata(
        dataset_name='fb15k',
        filename='fb15k.zip',
        url='https://s3-eu-west-1.amazonaws.com/ampligraph/datasets/fb15k.zip',
        train_name='train.txt',
        valid_name='valid.txt',
        test_name='test.txt',
        train_checksum='5a87195e68d7797af00e137a7f6929f2',
        valid_checksum='275835062bb86a86477a3c402d20b814',
        test_checksum='71098693b0efcfb8ac6cd61cf3a3b505'
    )

    return _load_dataset(FB15K,
                         data_home=None,
                         check_md5hash=check_md5hash,
                         add_reciprocal_rels=add_reciprocal_rels)


def load_fb15k_237(check_md5hash=False, clean_unseen=True, add_reciprocal_rels=False):
    """Load the FB15k-237 dataset

    FB15k-237 is a reduced version of FB15K. It was first proposed by :cite:`toutanova2015representing`.

    The FB15k-237 dataset is loaded from file if it exists at the ``AMPLIGRAPH_DATA_HOME`` location.
    If ``AMPLIGRAPH_DATA_HOME`` is not set the the default  ``~/ampligraph_datasets`` is checked.

    If the dataset is not found at either location it is downloaded and placed in ``AMPLIGRAPH_DATA_HOME``
    or ``~/ampligraph_datasets``.

    The dataset is divided in three splits:

    - ``train``
    - ``valid``
    - ``test``

    ========= ========= ======= ======= ============ ===========
     Dataset  Train     Valid   Test    Entities     Relations
    ========= ========= ======= ======= ============ ===========
    FB15K-237 272,115   17,535  20,466  14,541        237
    ========= ========= ======= ======= ============ ===========


    .. warning::
        FB15K-237's validation set contains 8 unseen entities over 9 triples.
        The test set has 29 unseen entities, distributed over 28 triples.

    Parameters
    ----------
    check_md5hash : boolean
        If ``True`` check the md5hash of the files. Defaults to ``False``.

    clean_unseen : bool
        If ``True``, filters triples in validation and test sets that include entities not present in the training set.

    add_reciprocal_rels : bool
        Flag which specifies whether to add reciprocal relations. For every <s, p, o> in the dataset
        this creates a corresponding triple with reciprocal relation <o, p_reciprocal, s>. (default: False).

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

    fb15k_237 = DatasetMetadata(
        dataset_name='fb15k-237',
        filename='fb15k-237.zip',
        url='https://s3-eu-west-1.amazonaws.com/ampligraph/datasets/fb15k-237.zip',
        train_name='train.txt',
        valid_name='valid.txt',
        test_name='test.txt',
        train_checksum='c05b87b9ac00f41901e016a2092d7837',
        valid_checksum='6a94efd530e5f43fcf84f50bc6d37b69',
        test_checksum='f5bdf63db39f455dec0ed259bb6f8628'
    )

    if clean_unseen:
        return _clean_data(_load_dataset(fb15k_237,
                                         data_home=None,
                                         check_md5hash=check_md5hash,
                                         add_reciprocal_rels=add_reciprocal_rels))
    else:
        return _load_dataset(fb15k_237,
                             data_home=None,
                             check_md5hash=check_md5hash,
                             add_reciprocal_rels=add_reciprocal_rels)


def load_yago3_10(check_md5hash=False, clean_unseen=True, add_reciprocal_rels=False):
    """Load the YAGO3-10 dataset

    The dataset is a split of YAGO3 :cite:`mahdisoltani2013yago3`,
    and has been first presented in :cite:`DettmersMS018`.

    The YAGO3-10 dataset is loaded from file if it exists at the ``AMPLIGRAPH_DATA_HOME`` location.
    If ``AMPLIGRAPH_DATA_HOME`` is not set the the default  ``~/ampligraph_datasets`` is checked.

    If the dataset is not found at either location it is downloaded and placed in ``AMPLIGRAPH_DATA_HOME``
    or ``~/ampligraph_datasets``.

    It is divided in three splits:

    - ``train``
    - ``valid``
    - ``test``

    ========= ========= ======= ======= ============ ===========
     Dataset  Train     Valid   Test    Entities     Relations
    ========= ========= ======= ======= ============ ===========
    YAGO3-10  1,079,040 5,000   5,000   123,182       37
    ========= ========= ======= ======= ============ ===========

    Parameters
    ----------
    check_md5hash : boolean
        If ``True`` check the md5hash of the files. Defaults to ``False``.

    clean_unseen : bool
        If ``True``, filters triples in validation and test sets that include entities not present in the training set.

    add_reciprocal_rels : bool
        Flag which specifies whether to add reciprocal relations. For every <s, p, o> in the dataset
        this creates a corresponding triple with reciprocal relation <o, p_reciprocal, s>. (default: False).

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
    yago3_10 = DatasetMetadata(
        dataset_name='YAGO3-10',
        filename='YAGO3-10.zip',
        url='https://s3-eu-west-1.amazonaws.com/ampligraph/datasets/YAGO3-10.zip',
        train_name='train.txt',
        valid_name='valid.txt',
        test_name='test.txt',
        train_checksum='a9da8f583ec3920570eeccf07199229a',
        valid_checksum='2d679a906f2b1ac29d74d5c948c1ad09',
        test_checksum='14bf97890b2fee774dbce5f326acd189'
    )

    if clean_unseen:
        return _clean_data(_load_dataset(yago3_10,
                                         data_home=None,
                                         check_md5hash=check_md5hash,
                                         add_reciprocal_rels=add_reciprocal_rels))
    else:
        return _load_dataset(yago3_10,
                             data_home=None,
                             check_md5hash=check_md5hash,
                             add_reciprocal_rels=add_reciprocal_rels)


def load_wn11(check_md5hash=False, clean_unseen=True, add_reciprocal_rels=False):
    """Load the WordNet11 (WN11) dataset

    WordNet was originally proposed in `WordNet: a lexical database for English` :cite:`miller1995wordnet`.

    WN11 dataset is loaded from file if it exists at the ``AMPLIGRAPH_DATA_HOME`` location.
    If ``AMPLIGRAPH_DATA_HOME`` is not set the the default  ``~/ampligraph_datasets`` is checked.

    If the dataset is not found at either location, it is downloaded and placed in ``AMPLIGRAPH_DATA_HOME``
    or ``~/ampligraph_datasets``.

    It is divided in three splits:

    - ``train``
    - ``valid``
    - ``test``

    Both the validation and test splits are associated with labels (binary ndarrays),
    with `True` for positive statements and `False` for  negatives:

    - ``valid_labels``
    - ``test_labels``

    ========= ========= ========== ========== ======== ======== ============ ===========
     Dataset  Train     Valid Pos  Valid Neg  Test Pos Test Neg Entities     Relations
    ========= ========= ========== ========== ======== ======== ============ ===========
    WN11      110361    2606       2609       10493    10542    38588        11
    ========= ========= ========== ========== ======== ======== ============ ===========

    Parameters
    ----------
    check_md5hash : boolean
        If ``True`` check the md5hash of the files. Defaults to ``False``.

    clean_unseen : bool
        If ``True``, filters triples in validation and test sets that include entities not present in the training set.

    add_reciprocal_rels : bool
        Flag which specifies whether to add reciprocal relations. For every <s, p, o> in the dataset
        this creates a corresponding triple with reciprocal relation <o, p_reciprocal, s>. (default: False).

    Returns
    -------

    splits : dict
        The dataset splits: {'train': train, 'valid': valid, 'valid_labels': valid_labels,
        'test': test, 'test_labels': test_labels}.
        Each split containing a dataset is an ndarray of shape [n, 3].
        The labels are ndarray of shape [n].

    Examples
    -------

    >>> from ampligraph.datasets import load_wn11
    >>> X = load_wn11()
    >>> X["valid"][0]
    array(['__genus_xylomelum_1', '_type_of', '__dicot_genus_1'], dtype=object)
    >>> X["valid_labels"][0:3]
    array([ True, False,  True])

    """
    wn11 = DatasetMetadata(
        dataset_name='wordnet11',
        filename='wordnet11.zip',
        url='https://s3-eu-west-1.amazonaws.com/ampligraph/datasets/wordnet11.zip',
        train_name='train.txt',
        valid_name='dev.txt',
        test_name='test.txt',
        train_checksum='2429c672c89e33ad4fa8e1a3ade416e4',
        valid_checksum='87bf86e225e79294a2524089614b96aa',
        test_checksum='24113b464f8042c339e3e6833c1cebdf'
    )

    dataset = _load_dataset(wn11, data_home=None,
                            check_md5hash=check_md5hash,
                            add_reciprocal_rels=add_reciprocal_rels)

    valid_labels = dataset['valid'][:, 3]
    test_labels = dataset['test'][:, 3]

    dataset['valid'] = dataset['valid'][:, 0:3]
    dataset['test'] = dataset['test'][:, 0:3]

    dataset['valid_labels'] = valid_labels == '1'
    dataset['test_labels'] = test_labels == '1'

    if clean_unseen:
        clean_dataset, valid_idx, test_idx = _clean_data(dataset, return_idx=True)
        clean_dataset['valid_labels'] = dataset['valid_labels'][valid_idx]
        clean_dataset['test_labels'] = dataset['test_labels'][test_idx]
        return clean_dataset
    else:
        return dataset


def load_fb13(check_md5hash=False, clean_unseen=True, add_reciprocal_rels=False):
    """Load the Freebase13 (FB13) dataset

    FB13 is a subset of Freebase :cite:`bollacker2008freebase`
    and was initially presented in
    `Reasoning With Neural Tensor Networks for Knowledge Base Completion` :cite:`socher2013reasoning`.

    FB13 dataset is loaded from file if it exists at the ``AMPLIGRAPH_DATA_HOME`` location.
    If ``AMPLIGRAPH_DATA_HOME`` is not set the the default  ``~/ampligraph_datasets`` is checked.

    If the dataset is not found at either location, it is downloaded and placed in ``AMPLIGRAPH_DATA_HOME``
    or ``~/ampligraph_datasets``.

    It is divided in three splits:

    - ``train``
    - ``valid``
    - ``test``

    Both the validation and test splits are associated with labels (binary ndarrays),
    with `True` for positive statements and `False` for  negatives:

    - ``valid_labels``
    - ``test_labels``

    ========= ========= ========== ========== ======== ======== ============ ===========
     Dataset  Train     Valid Pos  Valid Neg  Test Pos Test Neg Entities     Relations
    ========= ========= ========== ========== ======== ======== ============ ===========
    FB13      316232    5908       5908       23733    23731    75043        13
    ========= ========= ========== ========== ======== ======== ============ ===========

    Parameters
    ----------
    check_md5hash : boolean
        If ``True`` check the md5hash of the files. Defaults to ``False``.

    clean_unseen : bool
        If ``True``, filters triples in validation and test sets that include entities not present in the training set.

    add_reciprocal_rels : bool
        Flag which specifies whether to add reciprocal relations. For every <s, p, o> in the dataset
        this creates a corresponding triple with reciprocal relation <o, p_reciprocal, s>. (default: False).

    Returns
    -------

    splits : dict
        The dataset splits: {'train': train, 'valid': valid, 'valid_labels': valid_labels,
        'test': test, 'test_labels': test_labels}.
        Each split containing a dataset is an ndarray of shape [n, 3].
        The labels are ndarray of shape [n].

    Examples
    -------

    >>> from ampligraph.datasets import load_fb13
    >>> X = load_fb13()
    >>> X["valid"][0]
    array(['cornelie_van_zanten', 'gender', 'female'], dtype=object)
    >>> X["valid_labels"][0:3]
    array([True, False, True], dtype=object)

    """
    fb13 = DatasetMetadata(
        dataset_name='freebase13',
        filename='freebase13.zip',
        url='https://s3-eu-west-1.amazonaws.com/ampligraph/datasets/freebase13.zip',
        train_name='train.txt',
        valid_name='dev.txt',
        test_name='test.txt',
        train_checksum='9099ebcd85ab3ce723cfaaf34f74dceb',
        valid_checksum='c4ef7b244baa436a97c2a5e57d4ba7ed',
        test_checksum='f9af2eac7c5a86996c909bdffd295528'
    )

    dataset = _load_dataset(fb13,
                            data_home=None,
                            check_md5hash=check_md5hash,
                            add_reciprocal_rels=add_reciprocal_rels)

    valid_labels = dataset['valid'][:, 3]
    test_labels = dataset['test'][:, 3]

    dataset['valid'] = dataset['valid'][:, 0:3]
    dataset['test'] = dataset['test'][:, 0:3]

    dataset['valid_labels'] = valid_labels == '1'
    dataset['test_labels'] = test_labels == '1'

    if clean_unseen:
        clean_dataset, valid_idx, test_idx = _clean_data(dataset, return_idx=True)
        clean_dataset['valid_labels'] = dataset['valid_labels'][valid_idx]
        clean_dataset['test_labels'] = dataset['test_labels'][test_idx]
        return clean_dataset
    else:
        return dataset


def load_all_datasets(check_md5hash=False):
    load_wn18(check_md5hash)
    load_wn18rr(check_md5hash)
    load_fb15k(check_md5hash)
    load_fb15k_237(check_md5hash)
    load_yago3_10(check_md5hash)
    load_wn11(check_md5hash)
    load_fb13(check_md5hash)


def load_from_rdf(folder_name, file_name, rdf_format='nt', data_home=None, add_reciprocal_rels=False):
    """Load an RDF file

    Loads an RDF knowledge graph using rdflib_ APIs.
    Multiple RDF serialization formats are supported (nt, ttl, rdf/xml, etc).
    The entire graph will be loaded in memory, and converted into an rdflib `Graph` object.

    .. _rdflib: https://rdflib.readthedocs.io/

    .. warning::
        Large RDF graphs should be serialized to ntriples beforehand and loaded with ``load_from_ntriples()`` instead.

    .. note::
        It is recommended to use :meth:`ampligraph.evaluation.train_test_split_no_unseen` to split custom
        knowledge graphs into train, validation, and test sets. Using this function will lead to validation, test sets
        that do not include triples with entities that do not occur in the training set.


    Parameters
    ----------
    folder_name: str
        Base folder where the file is stored.
    file_name : str
        File name.
    rdf_format : str
        The RDF serialization format (nt, ttl, rdf/xml - see rdflib documentation).
    data_home : str
       The path to the folder that contains the datasets.
    add_reciprocal_rels : bool
        Flag which specifies whether to add reciprocal relations. For every <s, p, o> in the dataset
        this creates a corresponding triple with reciprocal relation <o, p_reciprocal, s>. (default: False).

    Returns
    -------
        triples : ndarray , shape [n, 3]
            the actual triples of the file.
    """

    logger.debug('Loading rdf data from {}.'.format(file_name))
    data_home = _get_data_home(data_home)
    from rdflib import Graph
    g = Graph()
    g.parse(os.path.join(data_home, folder_name, file_name), format=rdf_format, publicID='http://test#')
    triples = pd.DataFrame(np.array(g))
    triples = triples.drop_duplicates()
    if add_reciprocal_rels:
        triples = _add_reciprocal_relations(triples)

    return triples.values


def load_from_ntriples(folder_name, file_name, data_home=None, add_reciprocal_rels=False):
    """Load RDF ntriples

    Loads an RDF knowledge graph serialized as ntriples, without building an RDF graph in memory.
    This function should be preferred over ``load_from_rdf()``,
    since it does not load the graph into an rdflib model (and it is therefore faster by order of magnitudes).
    Nevertheless, it requires a ntriples_ serialization as in the example below:

    .. _ntriples: https://www.w3.org/TR/n-triples/.

    .. code-block:: text

        _:alice <http://xmlns.com/foaf/0.1/knows> _:bob .
        _:bob <http://xmlns.com/foaf/0.1/knows> _:alice .

    .. note::
        It is recommended to use :meth:`ampligraph.evaluation.train_test_split_no_unseen` to split custom
        knowledge graphs into train, validation, and test sets. Using this function will lead to validation, test sets
        that do not include triples with entities that do not occur in the training set.


    Parameters
    ----------
    folder_name: str
        base folder where the file is stored.
    file_name : str
        file name
    data_home : str
       The path to the folder that contains the datasets.
    add_reciprocal_rels : bool
        Flag which specifies whether to add reciprocal relations. For every <s, p, o> in the dataset
        this creates a corresponding triple with reciprocal relation <o, p_reciprocal, s>. (default: False).

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

    if add_reciprocal_rels:
        df = _add_reciprocal_relations(df)

    return df.values
