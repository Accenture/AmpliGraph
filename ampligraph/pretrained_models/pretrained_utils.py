# Copyright 2019-2023 The AmpliGraph Authors. All Rights Reserved.
#
# This file is Licensed under the Apache License, Version 2.0.
# A copy of the Licence is available in LICENCE, or at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
import logging
from collections import namedtuple
from ampligraph.utils.file_utils import _fetch_file
from ampligraph.utils.model_utils import restore_model

AMPLIGRAPH_ENV_NAME = "AMPLIGRAPH_DATA_HOME"

ModelMetadata = namedtuple(
    "ModelMetadata",
    [
        "scoring_type",
        "dataset",
        "pretrained_model_name",
        "url",
        "model_checksum"
    ],
    defaults=(None, None, None, None, None),
)

MODEL_URLS = {
    "YAGO310_ROTATE": "https://ndownloader.figshare.com/files/66253883",
    "YAGO310_TRANSE": "https://ndownloader.figshare.com/files/66253871",
    "YAGO310_HOLE": "https://ndownloader.figshare.com/files/66253889",
    "YAGO310_COMPLEX": "https://ndownloader.figshare.com/files/66253892",
    "YAGO310_DISTMULT": "https://ndownloader.figshare.com/files/66253886",
    "FB15K-237_ROTATE": "https://ndownloader.figshare.com/files/66253835",
    "FB15K-237_TRANSE": "https://ndownloader.figshare.com/files/66253820",
    "FB15K-237_HOLE": "https://ndownloader.figshare.com/files/66253847",
    "FB15K-237_DISTMULT": "https://ndownloader.figshare.com/files/66253850",
    "FB15K-237_COMPLEX": "https://ndownloader.figshare.com/files/66253865",
    "WN18_DISTMULT": "https://ndownloader.figshare.com/files/66253841",
    "WN18_TRANSE": "https://ndownloader.figshare.com/files/66253832",
    "WN18_ROTATE": "https://ndownloader.figshare.com/files/66253859",
    "WN18_COMPLEX": "https://ndownloader.figshare.com/files/66253874",
    "WN18_HOLE": "https://ndownloader.figshare.com/files/66253877",
    "FB15K_HOLE": "https://ndownloader.figshare.com/files/66253817",
    "FB15K_COMPLEX": "https://ndownloader.figshare.com/files/66253829",
    "FB15K_TRANSE": "https://ndownloader.figshare.com/files/66253826",
    "FB15K_DISTMULT": "https://ndownloader.figshare.com/files/66253838",
    "FB15K_ROTATE": "https://ndownloader.figshare.com/files/66253856",
    "WN18RR_TRANSE": "https://ndownloader.figshare.com/files/66253862",
    "WN18RR_ROTATE": "https://ndownloader.figshare.com/files/66253880",
    "WN18RR_DISTMULT": "https://ndownloader.figshare.com/files/66253868",
    "WN18RR_HOLE": "https://ndownloader.figshare.com/files/66253844",
    "WN18RR_COMPLEX": "https://ndownloader.figshare.com/files/66253853",
}

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def load_pretrained_model(dataset, scoring_type, data_home=None):
    """
    Function to load a pretrained model.

    This function allows downloading and loading one of the AmpliGraph pre-trained
    model on benchmark datasets.

    Parameters
    ----------
    dataset: str
        Specify the dataset on which the pre-trained model was built. The possible
        value is one of `["fb15k-237", "wn18rr", "yago310", "fb15k", "wn18rr"]`.
    scoring_type: str
        The scoring function used when training the model. The possible value is one of
        `["TransE", "DistMult", "ComplEx", "HolE", "RotatE"]`.

    Return
    ------
    model: ScoringBasedEmbeddingModel
        The pre-trained :class:`~ampligraph.latent_features.ScoringBasedEmbeddingModel`.

    Example
    -------
    >>> from ampligraph.datasets import load_fb15k_237
    >>> from ampligraph.pretrained_models import load_pretrained_model
    >>> from ampligraph.evaluation.metrics import mrr_score, hits_at_n_score, mr_score
    >>>
    >>> dataset = load_fb15k_237()
    >>> model = load_pretrained_model(dataset_name="fb15k-237", scoring_type="ComplEx")
    >>> ranks = model.evaluate(
    >>>     dataset['test'],
    >>>     corrupt_side='s,o',
    >>>     use_filter={'train': dataset['train'],
    >>>                 'valid': dataset['valid'],
    >>>                 'test': dataset['test']}
    >>> )
    >>> print(f"mr_score: {mr_score(ranks)}")
    >>> print(f"mrr_score: {mrr_score(ranks)}")
    >>> print(f"hits@1: {hits_at_n_score(ranks, 1)}")
    >>> print(f"hits@10: {hits_at_n_score(ranks, 10)}")
    """
    assert dataset in ["fb15k-237", "wn18rr", "yago310", "fb15k", "wn18rr"], \
        "The dataset you specified is not one of the available ones! Try with one of " \
        "the following: ['fb15k-237', 'wn18rr', 'yago310', 'fb15k', 'wn18rr']."
    assert scoring_type in ["TransE", "DistMult", "ComplEx", "HolE", "RotatE"], \
        "The scoring type you provided is not one of the available ones! Try with one of " \
        "the following: ['TransE', 'DistMult', 'ComplEx', 'HolE', 'RotatE']."

    model_name = scoring_type.upper()
    dataset_name = dataset.upper()
    pretrained_model_name = dataset_name + "_" + model_name
    filename = pretrained_model_name
    url = MODEL_URLS[filename]

    metadata = ModelMetadata(
        scoring_type=scoring_type,
        dataset=dataset,
        pretrained_model_name=pretrained_model_name,
        url=url
    )

    # with this command we download the .zip file and unzip it, so that, in the
    # desired folder, we'll have the model ready to be loaded.
    model_path = _fetch_file(metadata, data_home, file_type='models')

    return restore_model(model_path)
