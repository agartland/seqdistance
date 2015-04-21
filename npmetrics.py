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

"""TODO: Should some of thes functions be designated as private (with "_")?"""

import numpy as nb

__all__ = ['np_seq_similarity']

def np_seq_similarity(seq1, seq2, substMat, normed, asDistance):
    """Computes sequence similarity based on the substitution matrix."""
    if seq1.shape[0] != seq2.shape[0]:
        raise IndexError, "Sequences must be the same length (%d != %d)." % (seq1.shape[0],seq2.shape[0])

    """Similarity between seq1 and seq2 using the substitution matrix subst"""
    sim12 = substMat[seq1,seq2]

    if normed or asDistance:
        siteN = (~np.isnan(sim12)).sum()
        sim11 = np.nansum(substMat[seq1,seq1])/siteN
        sim22 = np.nansum(substMat[seq1,seq1])/siteN
        tot12 = np.nansum(2*sim12)/(sim11+sim22)
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