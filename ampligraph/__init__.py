# Copyright 2019-2021 The AmpliGraph Authors. All Rights Reserved.
#
# This file is Licensed under the Apache License, Version 2.0.
# A copy of the Licence is available in LICENCE, or at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
"""AmpliGraph is a library for relational learning on knowledge graphs."""
import logging.config
import pkg_resources

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

__version__ = '1.4.0'
__all__ = ['datasets', 'latent_features', 'discovery', 'evaluation', 'utils']

logging.config.fileConfig(pkg_resources.resource_filename(__name__, 'logger.conf'), disable_existing_loggers=False)
