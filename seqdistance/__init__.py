from __future__ import division
"""
Optimized genetic distance/similarity module for python.
========================================================
Includes string versions as well as numba and numpy optimized
functions for computing genetic distances using substitution matrices.
"""
import os
import sys

from . import matrices
from . import metrics
from . import plotting

from .tools import *

from .metrics import NB_SUCCESS

BADAA = '-*BX#Z'
"""FULL_AALPHABET is used for indexing substitution matrices so that they are all consistent."""
FULL_AALPHABET = 'ABCDEFGHIKLMNPQRSTVWXYZ-'
AALPHABET = 'ACDEFGHIKLMNPQRSTVWY'

__all__ = ['metrics',
           'plotting',
           'matrices',
           'NB_SUCCESS',
           'isvalidpeptide',
           'removeBadAA',
           'seq2vec',
           'seqs2mat',
           'mat2seqs',
           'vec2seq',
           'string2byte',
           'seq_similarity',
           'seq_distance',
           'hamming_distance',
           'unalign_similarity',
           'distance_rect',
           'distance_df',
           'BADAA',
           'FULL_AALPHABET',
           'AALPHABET']