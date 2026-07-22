# Copyright 2019-2026 The AmpliGraph Authors. All Rights Reserved.
#
# This file is Licensed under the Apache License, Version 2.0.0.
# A copy of the Licence is available in LICENCE, or at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
"""AmpliGraph is a library for relational learning on knowledge graphs."""
import logging.config
import os

# Silence TensorFlow's (C++) info/warning logging
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

__version__ = '2.2.0'
__all__ = ['datasets', 'latent_features', 'discovery', 'evaluation', 'utils', 'pretrained_models']

logging.config.fileConfig(
    os.path.join(os.path.dirname(__file__), "logger.conf"),
    disable_existing_loggers=False,
)
