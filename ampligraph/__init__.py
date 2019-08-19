# Copyright 2019 The AmpliGraph Authors. All Rights Reserved.
#
# This file is Licensed under the Apache License, Version 2.0.
# A copy of the Licence is available in LICENCE, or at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
"""Explainable Link Prediction is a library for relational learning on knowledge graphs."""
import logging.config
import pkg_resources

__version__ = '1.2-dev'
__all__ = ['datasets', 'latent_features', 'discovery', 'evaluation', 'utils']

logging.config.fileConfig(pkg_resources.resource_filename(__name__, 'logger.conf'), disable_existing_loggers=False)
