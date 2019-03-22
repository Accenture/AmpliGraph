"""Explainable Link Prediction is a library for relational learning on knowledge graphs."""
import logging.config
import pkg_resources

__version__ = '1.0.1'
__all__ = ['datasets', 'latent_features', 'evaluation']

logging.config.fileConfig(pkg_resources.resource_filename(__name__, 'logger.conf'), disable_existing_loggers=False)
