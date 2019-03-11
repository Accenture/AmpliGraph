"""Explainable Link Prediction is a library for relational learning on knowledge graphs."""
import logging.config
import os

__version__ = '0.3-dev'
__all__ = ['datasets', 'latent_features', 'evaluation']

curr_dir, _ = os.path.split(__file__)
logging.config.fileConfig(fname=os.path.join(curr_dir,'logger.conf'), disable_existing_loggers=False)
