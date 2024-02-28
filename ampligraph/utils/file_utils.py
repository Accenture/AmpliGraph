# Copyright 2019-2023 The AmpliGraph Authors. All Rights Reserved.
#
# This file is Licensed under the Apache License, Version 2.0.
# A copy of the Licence is available in LICENCE, or at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#

"""
This module contains utility functions to download and interact with files (specifically datasets
and pretrained model) stored online.
"""

import logging
import os
import hashlib
import zipfile
import urllib
from pathlib import Path


AMPLIGRAPH_ENV_NAME = "AMPLIGRAPH_DATA_HOME"

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def _get_data_home(file_home=None, file_type='datasets'):
    """
    Get the location of the data folder to use.

    Automatically determine the file folder to use.
    If ``file_home`` is provided, a check is performed to see if the path exists and creates one if it does not.
    If ``file_home`` is `None` the ``AMPLIGRAPH_ENV_NAME`` is used.
    If ``AMPLIGRAPH_ENV_NAME`` is not set, the default environment ``~/ampligraph_datasets`` or
    ``~/ampligraph_models`` are used, depending on the value assigned to ``file_type``.

    Parameters
    ----------
    file_home: str
       The path to the folder that contains the model.
    file_type: str
        Whether the file we are trying to load is a dataset (``file_type='datasets'``) or
        a model (``file_type='models'``).

    Returns
    -------
    str
        The path to the file directory.
    """
    assert file_type in ["datasets", "models"],\
        f"The file_type provided has to be either 'datasets' or 'models', but you passed {file_type}!"
    if file_home is None:
        file_home = os.environ.get(
            AMPLIGRAPH_ENV_NAME, os.path.join("~", "ampligraph_" + file_type)
        )
    file_home = os.path.expanduser(file_home)
    if not os.path.exists(file_home):
        os.makedirs(file_home)
    logger.debug("model_home is set to {}".format(file_home))
    return file_home


def _unzip_file(remote, source, destination, check_md5hash=False):
    """Unzip a file from a source location to a destination folder.

    Parameters
    ----------

    source: str
        The path to the zipped file
    destination: str
        The destination directory to unzip the files to.

    """

    # TODO - add error checking
    with zipfile.ZipFile(source, "r") as zip_ref:
        logger.debug("Unzipping {} to {}".format(source, destination))
        zip_ref.extractall(destination)
    if check_md5hash:
        for file_name, remote_checksum in [
            [remote.train_name, remote.train_checksum],
            [remote.valid_name, remote.valid_checksum],
            [remote.test_name, remote.test_checksum],
            [remote.test_human_name, remote.test_human_checksum],
            [remote.test_human_ids_name, remote.test_human_ids_checksum],
            [remote.mapper_name, remote.mapper_checksum],
            [remote.valid_negatives_name, remote.valid_negatives_checksum],
            [remote.test_negatives_name, remote.test_negatives_checksum],
        ]:
            file_path = os.path.join(
                destination, remote.dataset_name, file_name
            )
            checksum = _md5(file_path)
            if checksum != remote_checksum:
                os.remove(source)
                msg = (
                    "{} has an md5 checksum of ({}) which is different from the expected ({}), "
                    "the file may be corrupted.".format(
                        file_path, checksum, remote_checksum
                    )
                )
                logger.error(msg)
                raise IOError(msg)
    os.remove(source)


def _md5(file_path):
    md5hash = hashlib.md5()
    chunk_size = 4096
    with open(file_path, "rb") as f:
        content_buffer = f.read(chunk_size)
        while content_buffer:
            md5hash.update(content_buffer)
            content_buffer = f.read(chunk_size)
    return md5hash.hexdigest()


def _fetch_remote_data(remote, download_dir, data_home, check_md5hash=False):
    """Download a remote file.

    Parameters
    ----------

    remote: DatasetMetadata or ModelMetadata
        Named tuple containing remote dataset meta information: dataset name, dataset filename,
        url, train filename, validation filename, test filename, train checksum, valid checksum, test checksum.
    download_dir: str
        The location to download the file to.
    data_home: str
        The location to save the dataset.
    check_md5hash: bool
        Whether to check the MD5 hash of the dataset file.

    """

    file_path = "{}.zip".format(download_dir)
    if not Path(file_path).exists():
        urllib.request.urlretrieve(remote.url, file_path)
        # TODO - add error checking
    _unzip_file(remote, file_path, data_home, check_md5hash)


def _fetch_file(remote, data_home=None, check_md5hash=False, file_type='datasets'):
    """Get a file.

    Gets the directory of a file. If the file is not found it is downloaded automatically.

    Parameters
    ----------
    remote: DatasetMetadata or ModelMetadata
        Named tuple containing remote datasets meta information: dataset name, dataset filename,
        url, train filename, validation filename, test filename, train checksum, valid checksum, test checksum.
    data_home: str
        The location to save the file to.
    check_md5hash: bool
        Whether to check the MD5 hash of the file.

    Returns
    ------
    str
        The location of the file.
    """
    data_home = _get_data_home(data_home, file_type=file_type)
    if file_type == "datasets":
        file_dir = os.path.join(data_home, remote.dataset_name)
    elif file_type == "models":
        file_dir = os.path.join(data_home, remote.pretrained_model_name)

    if not os.path.exists(file_dir):
        if remote.url is None:
            msg = f"No file found at {file_dir} and no url provided."
            logger.error(msg)
            raise Exception(msg)

        _fetch_remote_data(remote, file_dir, data_home, check_md5hash)
    return file_dir
