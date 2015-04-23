from __future__ import division
"""
Optimized genetic distance/similarity module for python.
========================================================
Includes string versions as well as numba and numpy optimized
functions for computing genetic distances using substitution matrices.
"""

global NB_SUCCESS 
global BADAA
global FULL_AALPHABET
global AALPHABET

NB_SUCCESS = False
BADAA = '-*BX#Z'
"""FULL_AALPHABET is used for indexing substitution matrices so that they are all consistent."""
FULL_AALPHABET = 'ABCDEFGHIKLMNPQRSTVWXYZ-'
AALPHABET = 'ACDEFGHIKLMNPQRSTVWY'

try:
    import numba as nb
    import nbmetrics
    NB_SUCCESS = True
except WindowsError:
    NB_SUCCESS = False
    print 'Could not load numba\n(may be a path issue try starting python in C:\\)'
except ImportError:
    NB_SUCCESS = False
    print 'Could not load numba'

from tools import *
import strmetrics
import plotting
import npmetrics
import matrices

__all__ = ['nbmetrics',
           'npmetrics',
           'strmetrics',
           'plotting',
           'matrices',
           'NB_SUCCESS',
           'BADAA',
           'FULL_AALPHABET',
           'AALPHABET']