"""
Numba optimized functions for computing similarities and distances.

These routines are only neccesary for computing thousands of distances
between sequences or short peptides.

For single distances or smaller distances these functions may
effectively be slower due to conversions from strings to vector
representations.

Templates:

d = nb_metric(seqVec1, seqVec2, substMat, **kwargs)

pwdist = nb_dist_rect(seqMat1, seqMat2, substMat, **kwargs)


"""

"""NOTE: The default kwargs used here will not jit with numba as of version 0.18.2"""

import numpy as np

"""Can assume NB_SUCCESS is True if this module is imported at all."""
from . import NB_SUCCESS
from . import nb

__all__ = ['nb_hamming_distance',
           'nb_seq_similarity',
           'nb_seq_distance',
           'nb_coverage_distance']

@nb.jit(nb.float64(nb.char[:],nb.char[:]), nopython = True)
def nb_hamming_distance(str1,str2):
    assert str1.shape[0] == str2.shape[0]

    tot = 0
    for s1,s2 in zip(str1,str2):
        if s1 != s2:
            tot += 1
    return tot

@nb.jit(nb.float64(nb.int8[:],nb.int8[:],nb.float64[:,:],nb.boolean,nb.boolean), nopython = True)
def nb_seq_similarity(seq1, seq2, substMat, normed = True, asDistance = False):
    """Computes sequence similarity based on the substitution matrix."""
    assert seq1.shape[0] == seq2.shape[0]

    site12N = 0.
    if normed:
        site11N = 0.
        site22N = 0.

        sim12 = 0.
        sim11 = 0.
        sim22 = 0.
        for i in range(seq1.shape[0]):
            cur12 = substMat[seq1[i],seq2[i]]
            cur11 = substMat[seq1[i],seq1[i]]
            cur22 = substMat[seq2[i],seq2[i]]
            if not np.isnan(cur12):
                sim12 += cur12
                site12N += 1.
            if not np.isnan(cur11):
                sim11 += cur11
                site11N += 1.
            if not np.isnan(cur22):
                sim22 += cur22
                site22N += 1.
        if site11N == 0 or site22N == 0:
            sim12 = np.nan
        else:
            sim12 = 2*sim12/((sim11/site11N) + (sim22/site22N))
    else:
        sim12 = 0.
        for i in range(seq1.shape[0]):
            cur12 = substMat[seq1[i],seq2[i]]
            if not np.isnan(cur12):
                sim12 += cur12
                site12N += 1.

    if asDistance:
        if normed:
            if site12N == 0.:
                sim12 = np.nan
            else:
                sim12 = (site12N - sim12)/site12N
        else:
            sim12 = site12N - sim12
    return sim12

@nb.jit(nb.float64(nb.int8[:],nb.int8[:],nb.float64[:,:],nb.boolean), nopython = True)
def nb_seq_distance(seq1, seq2, substMat, normed = False):
    """Compare two sequences and return the distance from one to the other
    If the seqs are of different length then it raises an exception

    Returns a scalar [0, siteN] where siteN ignores nan similarities which may depend on gaps
    Optionally returns normed = True distance:
        [0, 1]

    Note that either way the distance is "normed", its either per site (True) or total normed (False):
        [0, siteN]"""
    return nb_seq_similarity(seq1, seq2, substMat, normed, True)

@nb.jit(nb.float64(nb.int8[:],nb.int8[:],nb.int8), nopython = True)
def nb_coverage_distance(epitope, peptide, mmTolerance = 0):
    """Determines whether pepitide covers epitope
    and can handle epitopes and peptides of different lengths.

    To be a consistent distance matrix:
        covered = 0
        not-covered = 1

    If epitope is longer than peptide it is not covered.
    Otherwise coverage is determined based on a mmTolerance

    Parameters
    ----------
    epitope : np.array
    peptide : np.array
    mmTolerance : int
        Number of mismatches tolerated
        If dist <= mmTolerance then it is covered

    Returns
    -------
    covered : int
        Covered (0) or not-covered (1)"""

    LEpitope, LPeptide = len(epitope), len(peptide)
    if LEpitope > LPeptide:
        return 1

    for starti in range(LPeptide-LEpitope+1):
        mm = 0
        for k in range(LEpitope):
            if epitope[k] != peptide[starti + k]:
                mm = mm + 1
                if mm > mmTolerance:
                    """If this peptide is already over the tolerance then goto next one"""
                    break
        if mm <= mmTolerance:
            """If this peptide is below tolerance then return covered (0)"""
            return 0
    """If no peptides meet mmTolerance then return not covered"""
    return 1