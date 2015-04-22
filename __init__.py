from __future__ import division
"""
Optimized genetic distance/similarity module for python.
========================================================
Includes string versions as well as numba and numpy optimized
functions for computing genetic distances using substitution matrices.
"""

"""Include code here to attempt to import numba and set flags about the success?"""

from tools import *
import strmetrics
import plotting
import npmetrics
import matrices

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


__all__ = ['nbmetrics',
           'npmetrics',
           'strmetrics',
           'plotting',
           'matrices']