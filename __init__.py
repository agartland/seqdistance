
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

import os
import sys

try:
    import numba as nb
    from . import nbmetrics
    print('seqdistance: Successfully imported numba version %s' % (nb.__version__))
    NB_SUCCESS = True
except OSError:
    try:
        """On Windows it is neccessary to be on the same drive as the LLVM DLL
        in order to import numba without generating a "Windows Error 161: The specified path is invalid."""
        curDir = os.getcwd()
        targetDir = os.path.splitdrive(sys.executable)[0]
        os.chdir(targetDir)
        import numba as nb
        from . import nbmetrics
        print('seqdistance: Successfully imported numba version %s' % (nb.__version__))
        NB_SUCCESS = True
    except OSError:
        NB_SUCCESS = False
        print('seqdistance: Could not load numba\n(may be a path issue try starting python in C:\\)')
    finally:
        os.chdir(curDir)
except ImportError:
    NB_SUCCESS = False
    print('seqdistance: Could not load numba')

from .tools import *
from . import strmetrics
from . import plotting
from . import npmetrics
from . import matrices

__all__ = ['npmetrics',
           'strmetrics',
           'plotting',
           'matrices',
           'NB_SUCCESS',
           'BADAA',
           'FULL_AALPHABET',
           'AALPHABET']
if NB_SUCCESS:
    __all__.append('nbmetrics')