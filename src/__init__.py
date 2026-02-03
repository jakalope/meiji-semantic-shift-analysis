"""
Edo-Meiji Polysemy Analysis Package

A computational linguistics research toolkit for analyzing semantic changes
in Japanese literature between the Edo (1603-1868) and Meiji (1868-1912) periods.

Modules:
    utils: Common utility functions (logging, device management, I/O)
    data_preprocess: Text preprocessing and tokenization
    embedding_extraction: BERT contextual embedding extraction
    polysemy_clustering: Clustering-based polysemy analysis
    compare_eras: Statistical comparison between historical eras
"""

__version__ = '0.1.0'
__author__ = 'Jacob Askeland'
__all__ = [
    'utils',
    'data_preprocess',
    'embedding_extraction',
    'polysemy_clustering',
    'compare_eras'
]
