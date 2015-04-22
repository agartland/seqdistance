"""Substitution matrices and associated functions.
Several of these are originally imported from BioPython.

Substitution matrices are similarity and not distance matrices.
Matrices are represented in two ways in this module and there are
converters that go from one to the other:

    (1) Parameters called "subst" are dicts with tuple keys
        representing the substitution and keys with the similarity
        ('X', 'Y') : s
        
    (2) Parameters called "substMat" are square 2d np.ndarrays (dtype = np.float64)
        The indices align with those in the FULL_ALPHABET.
        TODO: Use pd.DataFrames instead to avoid losing the index. But
              this would not be a feature used by numba or numpy optimized routines,
              just the end user.

TODO:
    (1) Add suppot for a DNA alphabet and associated matrices.

"""

from Bio.SubsMat.MatrixInfo import blosum90, ident, blosum62
import numpy as np
from copy import deepcopy
from tools import AALPHABET, FULL_AALPHABET
import itertools

__all__ = ['nanGapScores',
           'nanZeroGapScores',
           'binGapScores',
           'blosum90GapScores',
           'binarySubst',
           'binaryMat',
           'identMat',
           'blosum62Mat',
           'blosum90Mat',
           'addGapScores',
           'blosum90',
           'ident',
           'blosum62']
           
           
def subst2mat(subst, alphabet = FULL_AALPHABET):
    """Converts a substitution dictionary
    (like those from Bio) into a numpy 2d substitution matrix"""
    mat = np.nan * np.zeros((len(alphabet),len(alphabet)), dtype = np.float64)
    for (aa1,aa2),v in subst.items():
        mat[alphabet.index(aa1),alphabet.index(aa2)] = v
    return mat

def addGapScores(subst, gapScores = None, minScorePenalty = False, returnMat = False):
    """Add gap similarity scores for each AA (Could be done once for a set of sequences to improve speed)
    if gapScores is None then it will use defaults:
        gapScores = {('-','-'):1,
                     ('-','X'):0,
                     ('X','-'):0}
    OR for blosum90 default is:
        blosum90GapScores = {('-','-'):5,
                             ('-','X'):-11,
                             ('X','-'):-11}
    """
    if minScorePenalty:
        gapScores = {('-','-') : 1,
                     ('-','X') : np.min(subst.values()),
                     ('X','-') : np.min(subst.values())}
    elif gapScores is None:
        if subst is binarySubst:
            print 'Using default binGapScores for binarySubst'
            gapScores = binGapScores
        elif subst is blosum90:
            print 'Using default blosum90 gap scores'
            gapScores = blosum90GapScores
        else:
            raise Exception('Cannot determine which gap scores to use!')
    su = deepcopy(subst)
    uAA = np.unique([k[0] for k in subst.keys()])
    su.update({('-',aa) : gapScores[('-','X')] for aa in uAA})
    su.update({(aa,'-') : gapScores[('X','-')] for aa in uAA})
    su.update({('-','-') : gapScores[('-','-')]})

    if returnMat:
        return subst2mat(su)
    return su


"""Many different ways of handling gaps. Remember that these are SIMILARITY scores"""
nanGapScores = {('-','-'):np.nan,
                ('-','X'):np.nan,
                ('X','-'):np.nan}

nanZeroGapScores = {('-','-'):np.nan,
                     ('-','X'):0,
                     ('X','-'):0}
"""Default for addGapScores()"""
binGapScores = {('-','-'):1,
                ('-','X'):0,
                ('X','-'):0}
"""Arbitrary/reasonable values (extremes for blosum90 I think)"""
blosum90GapScores = {('-','-'):5,
                     ('-','X'):-11,
                     ('X','-'):-11}

binarySubst = {(aa1,aa2):np.float64(aa1==aa2) for aa1,aa2 in itertools.product(AALPHABET, AALPHABET)}

identMat = subst2mat(ident)
blosum90Mat = subst2mat(blosum90)
blosum62Mat = subst2mat(blosum62)
binaryMat = subst2mat(binarySubst)