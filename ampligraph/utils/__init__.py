# Copyright 2019 The AmpliGraph Authors. All Rights Reserved.
#
# This file is Licensed under the Apache License, Version 2.0.
# A copy of the Licence is available in LICENCE, or at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
"""This module contains utility functions for neural knowledge graph embedding models.

"""

from .model_utils import save_model, restore_model, create_tensorboard_visualizations, write_metadata_tsv

__all__ = ['save_model', 'restore_model', 'create_tensorboard_visualizations', 'write_metadata_tsv']
