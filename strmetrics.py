"""
Metrics and tools that rely only on string computation
to compute similarities and distances.

Metric template:

d = str_metric(seq1, seq2, **kwargs)


"""

__all__ = ['hamming_distance',
           'trunc_hamming',
           'dichot_hamming',
           'aamismatch_distance',
           'coverageDistance']
           

import itertools
import operator

def hamming_distance(str1, str2, noConvert = False, **kwargs):
    """Hamming distance between str1 and str2.
    Only finds distance over the length of the shorter string.
    **kwargs are so this can be plugged in place of a seq_distance() metric"""
    if noConvert:
        return np.sum([i for i in itertools.imap(operator.__ne__, str1, str2)])

    if isinstance(str1,basestring):
        str1 = string2byte(str1)
    if isinstance(str2,basestring):
        str2 = string2byte(str2)
    return nb_hamming_distance(str1, str2)

def aamismatch_distance(seq1,seq2, **kwargs):
    if isinstance(seq1,basestring):
        seq1 = seq2vec(seq1)

    if isinstance(seq2,basestring):
        seq2 = seq2vec(seq2)
    dist12 = nb_seq_similarity(seq1, seq2, substMat = binaryMat, normed = False, asDistance = True)
    return dist12
    
def trunc_hamming(seq1,seq2,maxDist=2,**kwargs):
    """Truncated hamming distance
    d = hamming() if d<maxDist else d = maxDist"""
    d = hamming_distance(seq1,seq2)
    return maxDist if d >= maxDist else d

def dichot_hamming(seq1,seq2,mmTolerance=1,**kwargs):
    """Dichotamized hamming distance.
    hamming <= mmTolerance is 0 and all others are 1"""
    d = hamming_distance(seq1,seq2)
    return 1 if d > mmTolerance else 0
def coverageDistance(epitope,peptide, mmTolerance = 1,**kwargs):
    """Determines whether pepitde covers epitope
    and can handle epitopes and peptides of different lengths.

    To be a consistent distance matrix:
        covered = 0
        not-covered = 1

    If epitope is longer than peptide it is not covered.
    Otherwise coverage is determined based on a mmTolerance

    Can accomodate strings or np.arrays (but not a mix).

    Parameters
    ----------
    epitope : str or np.array
    peptide : str or np.array
    mmTolerance : int
        Number of mismatches tolerated
        If dist <= mmTolerance then it is covered

    Returns
    -------
    covered : int
        Covered (0) or not-covered (1)"""

    tEpitope, tPeptide = type(epitope), type(peptide)
    assert tEpitope == tPeptide

    LEpitope, LPeptide = len(epitope), len(peptide)
    if LEpitope > LPeptide:
        return 1

    if isinstance(epitope, basestring):
        min_dist = array([np.sum([i for i in itertools.imap(operator.__ne__, epitope, peptide[starti:starti+LEpitope])]) for starti in range(LPeptide-LEpitope+1)]).min()
    else:
        min_dist = array([(epitope != peptide[starti:starti+LEpitope]).sum() for starti in range(LPeptide-LEpitope+1)]).min()
    
    return 0 if min_dist <= mmTolerance else 1