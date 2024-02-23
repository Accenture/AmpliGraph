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
        f"The dataset you specified is not one of the available ones! Try with one of" \
        f"the following: ['fb15k-237', 'wn18rr', 'yago310', 'fb15k', 'wn18rr']."
    assert scoring_type in ["TransE", "DistMult", "ComplEx", "HolE", "RotatE"], \
        f"The scoring type you provided is not one of the available ones! Try with one of" \
        f"the following: ['TransE', 'DistMult', 'ComplEx', 'HolE', 'RotatE']."

    model_name = scoring_type.upper()
    dataset_name = dataset.upper()
    pretrained_model_name = dataset_name + "_" + model_name
    filename = pretrained_model_name + ".zip"
    url = "https://ampligraph.s3.eu-west-1.amazonaws.com/pretrained-models-v2.0/" + filename

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





