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

"""TODO: Should some of thes functions be designated as private (with "_")?"""

import numpy as nb
import numba as nb

__all__ = ['nb_hamming_distance',
           'nb_seq_similarity',
           'distRect']

@nb.jit(nb.int8(nb.char[:],nb.char[:]), nopython = True)
def nb_hamming_distance(str1,str2):
    tot = 0
    for s1,s2 in zip(str1,str2):
        if s1 != s2:
            tot += 1
    return tot

#@nb.jit(nb.float64(nb.int8[:],nb.int8[:],nb.float64[:,:],nb.boolean,nb.boolean), nopython = True)
@nb.jit(nopython = True)
def nb_seq_similarity(seq1, seq2, substMat, normed, asDistance):
    """Computes sequence similarity based on the substitution matrix."""
    if seq1.shape[0] != seq2.shape[0]:
        raise IndexError

    if normed or asDistance:
        sim12 = 0.
        siteN = 0.
        sim11 = 0.
        sim22 = 0.
        for i in range(seq1.shape[0]):
            cur12 = substMat[seq1[i],seq2[i]]
            cur11 = substMat[seq1[i],seq1[i]]
            cur22 = substMat[seq2[i],seq2[i]]
            if not np.isnan(cur12):
                sim12 += cur12
                siteN += 1.
            if not np.isnan(cur11):
                sim11 += cur11
            if not np.isnan(cur22):
                sim22 += cur22
        sim12 = 2*sim12/((sim11/siteN) + (sim22/siteN))
    else:
        sim12 = 0.
        siteN = 0.
        for i in range(seq1.shape[0]):
            if not np.isnan(substMat[seq1[i],seq2[i]]):
                sim12 += substMat[seq1[i],seq2[i]]
                siteN += 1.

    if asDistance:
        if normed:
            sim12 = (siteN - sim12)/siteN
        else:
            sim12 = siteN - sim12
    return sim12
    
def distRect_factory(nb_metric): 
    """Can be passed a numba jit'd distance function and
    will return a jit'd function for computing all pairwise distances using that function"""
    @nb.jit(nb.boolean(nb.float64[:,:],nb.int8[:,:],nb.int8[:,:],nb.float64[:,:],nb.boolean),nopython=True) 
    def nb_distRect(pwdist,rows,cols,substMat,symetric): 
        n = rows.shape[0] 
        m = cols.shape[0]
        for i in range(n): 
            for j in range(m): 
                if not symetric:
                    pwdist[i,j] = nb_seq_similarity(rows[i, :], cols[j, :],substMat=substMat, normed=False, asDistance=True)
                else:
                    if j<=i:
                        pwdist[i,j] = nb_seq_similarity(rows[i, :], cols[j, :],substMat=substMat, normed=False, asDistance=True)
                        pwdist[j,i] = pwdist[i,j]
        return True 
    return nb_distRect

def distRect(row_vecs, col_vecs, substMat, nb_metric, normalize=False, symetric=False):
    """These conversion will go in a wrapper function with the uniquing business
    if subst is None:
        substMat = subst2mat(addGapScores(binarySubst,binGapScores))
    else:
        substMat = subst2mat(subst)

    if nb_metric is None:
        nb_metric = nb_seq_similarity

    row_vecs = seqs2mat(row_seqs)
    col_vecs = seqs2mat(col_seqs)"""

    nb_drect = distRect_factory(nb_metric)
    pwdist = np.zeros((row_vecs.shape[0],col_vecs.shape[0]),dtype=np.float64)
    success = nb_drect(pwdist, row_vecs, col_vecs, substMat, symetric)
    assert success

    if normalize:
        pwdist = pwdist - pwdist.min()
    return pwdist

