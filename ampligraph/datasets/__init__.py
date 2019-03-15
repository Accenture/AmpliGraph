"""Helper functions to load knowledge graphs from disk."""

from .datasets import load_from_csv, load_from_rdf, load_fb15k, load_wn18, load_fb15k_237, load_from_ntriples, \
    load_wn11, load_fb13, load_yago3_10, load_wn18rr

__all__ = ['load_from_csv', 'load_from_rdf', 'load_from_ntriples', 'load_wn11', 'load_wn18', 'load_fb15k',
           'load_fb13', 'load_fb15k_237',
           'load_wn11', 'load_fb13', 'load_yago3_10', 'load_wn18rr']
