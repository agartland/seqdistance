"""Substitution matrices and associated functions.
Several of these are originally imported from BioPython.

Substitution matrices are similarity and not distance matrices.
Matrices are represented in two ways in this module and there are
converters that go from one to the other:

    (1) Parameters called "subst" are dicts with tuple keys
        representing the substitution and keys with the similarity
        ('X', 'Y') : s
        
    (2) Parameters called "substMat" are square 2d np.ndarrays (dtype = np.float64)
        The indices align with those in the FULL_AALPHABET.
        TODO: Use pd.DataFrames instead to avoid losing the index. But
              this would not be a feature used by numba or numpy optimized routines,
              just the end user.

TODO:
    (1) Add suppot for a DNA alphabet and associated matrices.

"""
from Bio.SubsMat.MatrixInfo import blosum90, ident, blosum62
import numpy as np
from copy import deepcopy
import itertools

from seqdistance import BADAA, FULL_AALPHABET, AALPHABET

def subst2mat(subst, alphabet = None):
    """Converts a substitution dictionary
    (like those from Bio) into a numpy 2d substitution matrix.

    Assumes the matrix is symetrical,
    but if its not this will still produce a good copy.

    Missing substitutions are nan.

    Return type is float64"""
    if alphabet is None:
        alphabet = FULL_AALPHABET
    mat = np.nan * np.zeros((len(alphabet),len(alphabet)), dtype = np.float64)
    ij = np.zeros((len(subst),2),dtype=np.int)
    for ki,((aa1,aa2),v) in enumerate(subst.items()):
        i,j = alphabet.index(aa1),alphabet.index(aa2)
        ij[ki,:] = [i,j]
        mat[i,j] = v
    for ki in range(ij.shape[0]):
        """Go back through all the assignments and make the symetrical assignments
        if the value is still nan (i.e. otherwise unspecified)"""
        i,j = ij[ki,0], ij[ki,1]
        if np.isnan(mat[j,i]):
            mat[j,i] = mat[i,j]
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