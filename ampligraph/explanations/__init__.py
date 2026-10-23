# Copyright 2019-2021 The AmpliGraph Authors. All Rights Reserved.
#
# This file is Licensed under the Apache License, Version 2.0.
# A copy of the Licence is available in LICENCE, or at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
"""The module includes explainers for the predictions of knowledge graph embedding models, and visualizers for their explanations."""
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

from .explainers import *
try:
    import pygraphviz
    import graphviz
    from .visualizers import *
except ImportError:
    logger.info("Without pygraphviz and graphviz you won't be able to use visualizers.")
from .utils import *
