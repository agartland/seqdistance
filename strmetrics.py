"""
Metrics and tools that rely only on string computation
to compute similarities and distances.

Metric template:

d = str_metric(seq1, seq2, **kwargs)


"""
import matrices

__all__ = ['str_hamming_distance',
           'trunc_hamming',
           'dichot_hamming',
           'str_coverage_distance',
           'str_seq_distance',
           'str_seq_similarity']

import itertools
import operator
import numpy as np

def str_hamming_distance(str1, str2):
    """Hamming distance between str1 and str2."""
    assert len(str1) == len(str2), "Inputs must have the same length."
    return np.sum([i for i in itertools.imap(operator.__ne__, str1, str2)])
  
def str_trunc_hamming(seq1, seq2, maxDist=2):
    """Truncated hamming distance
    d = str_hamming() if d<maxDist else d = maxDist"""
    d = str_hamming_distance(seq1,seq2)
    return maxDist if d >= maxDist else d

def dichot_hamming(seq1, seq2, mmTolerance=1):
    """Dichotamized hamming distance.
    hamming <= mmTolerance is 0 and all others are 1"""
    d = str_hamming_distance(seq1,seq2)
    return 1 if d > mmTolerance else 0

def str_seq_similarity(seq1, seq2, subst = None, normed = True, asDistance = False):
    """Compare two sequences and return the similarity of one and the other
    If the seqs are of different length then it raises an exception

    FOR HIGHLY DIVERGENT SEQUENCES THIS NORMALIZATION DOES NOT GET TO [0,1] BECAUSE OF EXCESS NEGATIVE SCORES!
    Consider normalizing the matrix first by adding the min() so that min = 0 (but do not normalize per comparison)
    
    Return a nansum of site-wise similarities between two sequences based on a substitution matrix
        [0, siteN] where siteN ignores nan similarities which may depend on gaps
        sim12 = nansum(2*sim12/(nanmean(sim11) + nanmean(sim22))
    Optionally specify normed = False:
        [0, total raw similarity]
    This returns a score [0, 1] for binary and blosum based similarities
        otherwise its just the sum of the raw score out of the subst matrix
    
    Optionally return a distance instead of a similarity.
    Distance between seq1 and seq2 using the substitution matrix subst
    because seq_similarity returns a total similarity with max of siteN (not per site), we use
        d = siteN - sim
    which is a total normed distance, not a per site distance"""

    assert len(seq1) == len(seq2), "len of seq1 (%d) and seq2 (%d) are different" % (len(seq1),len(seq2))

    """if subst is matrices.binarySubst:
        dist = str_hamming_distance(seq1,seq2)
        sim = len(seq1) - dist
        if normed:
            sim = sim / len(seq1)
        return sim"""

    if subst is None:
        subst = matrices.addGapScores(matrices.binarySubst,matrices.binGapScores)

    """Site-wise similarity between seq1 and seq2 using the substitution matrix subst"""
    sim12 = np.array([i for i in itertools.imap(lambda a,b: subst.get((a,b),subst.get((b,a),np.nan)), seq1, seq2)])

    if normed or asDistance:
        siteN = np.sum(~np.isnan(sim12))

    if normed:
        sim11 = str_seq_similarity(seq1, seq1, subst=subst, normed=False) / siteN
        sim22 = str_seq_similarity(seq2, seq2, subst=subst, normed=False) / siteN
        tot12 = np.nansum(2*sim12/(sim11+sim22))
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

def str_seq_distance(seq1, seq2, subst = None, normed = True):
    """Compare two sequences and return the distance from one to the other
    If the seqs are of different length then it raises an exception

    Returns a scalar [0, siteN] where siteN ignores nan similarities which may depend on gaps
    Optionally returns normed = True distance:
        [0, 1]

    Note that either way the distance is "normed", its either per site (True) or total normed (False):
        [0, siteN]"""
    return str_seq_similarity(seq1, seq2, subst = subst, normed = normed, asDistance = True)

    
def str_coverage_distance(epitope, peptide, mmTolerance = 1):
    """Determines whether pepitde covers epitope
    and can handle epitopes and peptides of different lengths.

    To be a consistent distance matrix:
        covered = 0
        not-covered = 1

    If epitope is longer than peptide it is not covered.
    Otherwise coverage is determined based on a mmTolerance

    Parameters
    ----------
    epitope : str
    peptide : str
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

    min_dist = np.array([np.sum([i for i in itertools.imap(operator.__ne__, epitope, peptide[starti:starti+LEpitope])]) for starti in range(LPeptide-LEpitope+1)]).min()
    
    return 0 if min_dist <= mmTolerance else 1