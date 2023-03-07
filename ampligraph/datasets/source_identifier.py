# Copyright 2019-2023 The AmpliGraph Authors. All Rights Reserved.
#
# This file is Licensed under the Apache License, Version 2.0.
# A copy of the Licence is available in LICENCE, or at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
"""Data source identifier.

This module provides the main class and the supporting functions for automatic
identification of data source (whether it is csv, tar.gz or numpy array)
and provides adequate loader for the data source identified.
"""
import logging
from collections.abc import Iterable
from itertools import chain, islice

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def load_csv(data_source, chunk_size=None, sep="\t", verbose=False, **kwargs):
    """CSV data loader.

    Parameters
    ---------
    data_source: str
        csv file with data, separated by ``sep``.
    chunk_size: int
        The size of chunk to be used while reading the data. If used, the returned type is
        an iterator and not a numpy array.
    sep: str
        Separator in the csv file, e.g. line "1,2,3\n" has ``sep=","``, while "1 2 3\n" has ``sep=" "``.

    Returns
    -------
    data: ndarray or iter
        Either a numpy array with data or a lazy iterator if ``chunk_size`` was provided.
    """
    data = pd.read_csv(
        data_source, sep=sep, chunksize=chunk_size, header=None, **kwargs
    )
    logger.debug("data type: {}".format(type(data)))
    logger.debug("CSV loaded, into iterator data.")

    if isinstance(data, pd.DataFrame):
        return data.values
    else:
        return data


def load_json(data_source, orient="records", chunksize=None):
    """json files data loader.

    Parameters
    ----------
    data_source : str
        Path to a .json file.
    orient : str
        Indicates the expected .json file format. The default ``orient="records"`` assumes the knowledge graph is
        stored as a list like `[{subject_1: value, predicate_1: value, object_1: value}, ...,
        {subject_n: value, predicate_n: value, object_n: value}]`. If looking for more options check the
        `Pandas <https://pandas.pydata.org/docs/reference/api/pandas.read_json.html>`_ website.
    chunksize : int
        The size of chunk to be used while reading the data. If used, the returned type is
        an iterator and not a numpy array.


    Returns
    -------
    data : ndarray or iter
        Either a numpy array with data or a lazy iterator if ``chunk_size`` was provided.
    """
    if chunksize is not None:
        data = pd.read_json(
            data_source, orient=orient, lines=True, chunksize=chunksize
        )
    else:
        data = pd.read_json(data_source, orient=orient)
    logger.debug("data type: {}".format(type(data)))
    logger.debug("JSON loaded into iterator data.")

    return data.values


def chunks(iterable, chunk_size=1):
    """Chunks generator."""
    iterator = iter(iterable)
    for first in iterator:
        yield np.array(list(chain([first], islice(iterator, chunk_size - 1))))


def load_gz(data_source, chunk_size=None, verbose=False):
    """Gz data loader. Reads compressed file."""
    raise NotImplementedError


def load_tar(data_source, chunk_size=None, verbose=False):
    """Tar data loader. Reads compressed file."""
    raise NotImplementedError


class DataSourceIdentifier:
    """Class that recognizes the type of given file and provides with an
    adequate loader.

    Properties
    ----------
    supported_types: dict
         Dictionary of supported types along with their adequate loaders, to support a new data type, this
         dictionary needs to be updated with the file extension as key and the loading function name as value.

    Example
    -------
    >>>identifier = DataSourceIdentifier("data.csv")
    >>>loader = identifier.fetch_loader()
    >>>X = loader("data.csv")
    """

    def __init__(self, data_source, verbose=False):
        """Initialise DataSourceIdentifier.

        Parameters
        ----------
        data_source: str
             Name of a file to be recognized.
        """
        self.verbose = verbose
        self.data_source = data_source
        self.supported_types = {
            "csv": load_csv,
            "txt": load_csv,
            "gz": load_csv,
            "json": load_json,
            "tar": load_tar,
            "iter": chunks,
        }
        self._identify()

    def fetch_loader(self):
        """Returns adequate loader required to read identified file."""
        logger.debug(
            "Return adequate loader that provides loading of data source."
        )
        return self.supported_types[self.src]

    def get_src(self):
        """Returns identified source type."""
        return self.src

    def _identify(self):
        """Identifies the data file type based on the file name."""
        if isinstance(self.data_source, str):
            self.src = (
                self.data_source.split(".")[-1]
                if "." in self.data_source
                else None
            )
            if self.src is not None and self.src not in self.supported_types:
                logger.debug(
                    "File type not supported! Supported types: {}".format(
                        ", ".join(self.supported_types)
                    )
                )
                self.src = None
        else:
            logger.debug("data_source is an object")
            if isinstance(self.data_source, Iterable):
                self.src = "iter"
                logger.debug("data_source is an iterable")
            else:
                logger.error("Object type not supported")
