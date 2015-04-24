"""
Numpy optimized functions for computing similarities and distances.

These routines are only neccesary for computing distances between
long sequences since the distance computationis vectorized. For
large numbers of distances these functions are not very useful
because the pairwise distance function still uses python for-loops.

For single distances or smaller distances these functions may
effectively be slower due to conversions from strings to vector
representations.

Templates:

d = np_metric(seqVec1, seqVec2, substMat, **kwargs)

pwdist = np_dist_rect(seqMat1, seqMat2, substMat, **kwargs)

"""

import numpy as np

__all__ = ['np_seq_similarity',
           'np_hamming_distance',
           'np_seq_distance',
           'np_coverage_distance']

def np_hamming_distance(seqVec1, seqVec2):
    """Hamming distance between str1 and str2."""
    assert seqVec1.shape[0] == seqVec2.shape[0], "Inputs must have the same length."
    return (seqVec1 != seqVec2).sum()

def np_seq_similarity(seq1, seq2, substMat, normed = True, asDistance = False):
    """Computes sequence similarity based on the substitution matrix."""
    assert seq1.shape[0] == seq2.shape[0], "Sequences must be the same length (%d != %d)." % (seq1.shape[0],seq2.shape[0])

    """Similarity between seq1 and seq2 using the substitution matrix subst"""
    sim12 = substMat[seq1,seq2]

    if normed or asDistance:
        siteN = (~np.isnan(sim12)).sum()

    if normed:
        sim11 = np.nansum(substMat[seq1,seq1])/siteN
        sim22 = np.nansum(substMat[seq2,seq2])/siteN
        tot12 = 2*np.nansum(sim12)/(sim11+sim22)
    else:
        tot12 = np.nansum(sim12)

    if asDistance:
        """Distance between seq1 and seq2 using the substitution matrix subst
            because seq_similarity returns a total similarity with max of siteN (not per site), we use
                d = siteN - sim
            which is a total normed distance, not a per site distance"""
        if normed:
            tot12 = (siteN - tot12)/siteN
        else:
            tot12 = siteN - tot12
    return tot12

def np_seq_distance(seqVec1, seqVec2, substMat = None, normed = False):
    """Compare two sequences and return the distance from one to the other
    If the seqs are of different length then it raises an exception

    Returns a scalar [0, siteN] where siteN ignores nan similarities which may depend on gaps
    Optionally returns normed = True distance:
        [0, 1]

    Note that either way the distance is "normed", its either per site (True) or total normed (False):
        [0, siteN]"""
    return np_seq_similarity(seqVec1, seqVec2, substMat = substMat, normed = normed, asDistance = True)
    
def np_coverage_distance(epitope, peptide, mmTolerance = 0):
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
    assert type(epitope) == np.ndarray
    assert type(peptide) == np.ndarray

    LEpitope, LPeptide = len(epitope), len(peptide)
    if LEpitope > LPeptide:
        return 1

    for starti in range(LPeptide-LEpitope+1):
        if np.sum(np.not_equal(epitope,peptide[starti:starti+LEpitope])) <= mmTolerance:
            return 0
    return 1