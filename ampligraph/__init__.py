"""Explainable Link Prediction is a library for relational learning on knowledge graphs."""
import logging.config
import os

__version__ = '0.3-dev'
__all__ = ['datasets', 'latent_features', 'evaluation']

logging.config.fileConfig(fname=os.path.join(os.path.abspath(os.path.dirname(__file__)),'logger.conf'), disable_existing_loggers=False)
