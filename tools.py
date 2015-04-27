"""
Generalized tools for computing distances and generating
pairwise distance matrices. Generally these tools take sequences
as strings and substitution matrices as dicts and use whatever
optimized functions are available (making neccessary conversions).

Conversions from string to vector based representations of sequences
is negligible for large datasets. If the dataset is small and the 
numba or numpy optized routines seem unnecessary these tools allow
you to specify that string-based metrics be used.

Module also contains alphabet constants and tools for sequence and
matrix conversions.

Sequence as a vector:
Genetic code string is represented as a 1d numpy int8 vector with
values based on each residue's position in the FULL_AALPHABET.

Sequence alignment as a matrix:
A genetic alignment (assumes same length for all sequences)
is represented as a 2d numpy int8 matrix [seqs x sites] with values
based on each residue's position in the FULL_AALPHABET.

"""
import numpy as np
import re
from Bio import pairwise2
import matrices
import npmetrics
import strmetrics
from . import FULL_AALPHABET
from . import BADAA
from . import NB_SUCCESS

if NB_SUCCESS:
    from . import nb
    import nbmetrics
else:
    nb = None

__all__ = ['isvalidpeptide',
           'removeBadAA',
           'seq2vec',
           'seqs2mat',
           'string2byte',
           'seq_similarity',
           'seq_distance',
           'hamming_distance',
           'unalign_similarity',
           'distance_rect']

def _unique_rows(a, return_index = False, return_inverse = False, return_counts = False):
    """Performs np.unique on whole rows of matrix a using a "view".
    See http://stackoverflow.com/a/16971324/74616"""
    dummy,uniqi,inv_uniqi,counts = np.unique(a.view(a.dtype.descr * a.shape[1]), return_index = True, return_inverse = True, return_counts = True)
    out = [a[uniqi,:]]
    if return_index:
        out.append(uniqi)
    if return_inverse:
        out.append(inv_uniqi)
    if return_counts:
        out.append(counts)
    return tuple(out)
           
def isvalidpeptide(mer, badaa = None):
    """Test if the mer contains an BAD amino acids in global BADAA
    typically -*BX#Z"""
    if badaa is None:
        badaa = BADAA
    if not mer is None:
        return not re.search('[%s]' % badaa,mer)
    else:
        return False
def removeBadAA(mer, badaa = None):
    """Remove badaa amino acids from the mer, default badaa is -*BX#Z"""
    if badaa is None:
        badaa = BADAA
    if not mer is None:
        return re.sub('[%s]' % badaa,'',mer)
    else:
        return mer
def string2byte(s):
    """Convert string to byte array since numba can't handle strings"""
    if isinstance(s,basestring):
        s = np.array(s)
    dtype = s.dtype
    if dtype is np.dtype('byte'):
        return s # it's already a byte array
    shape = list(s.shape)
    n = dtype.itemsize
    shape.append(n)
    return s.ravel().view(dtype='byte').reshape(shape)

def seq2vec(seq):
    """Convert AA sequence into numpy vector of integers for fast comparison"""
    vec = np.zeros(len(seq), dtype = np.int8)
    for aai,aa in enumerate(seq):
        vec[aai] = FULL_AALPHABET.index(aa)
    return vec
def seqs2mat(seqs):
    """Convert a collection of AA sequences into a
    numpy matrix of integers for fast comparison.

    Requires all seqs to have the same length."""
    L1 = len(seqs[0])
    mat = np.zeros((len(seqs),L1), dtype = np.int8)
    for si,s in enumerate(seqs):
        assert L1 == len(s), "All sequences must have the same length: L1 = %d, but L%d = %d" % (L1,si,len(s))
        for aai,aa in enumerate(s):
            mat[si,aai] = FULL_AALPHABET.index(aa)
    return mat

def hamming_distance(str1, str2, asStrings = False):
    """Hamming distance between str1 and str2.
    Inputs are required to be the same length.
    
    REMOVED **kwargs which were there so that this could be plugged in place of a seq_distance() metric"""

    if asStrings:
        assert isinstance(str1, basestring), "Seq1 is not a string."
        assert isinstance(str2, basestring), "Seq1 is not a string."
        return strmetrics.str_hamming_distance(str1, str2)

    if isinstance(str1,basestring):
        seqVec1 = seq2vec(str1)
    else:
        seqVec1 = str1

    if isinstance(str2,basestring):
        seqVec2 = seq2vec(str2)
    else:
        seqVec2 = str2

    if NB_SUCCESS:
        return nbmetrics.nb_hamming_distance(seqVec1, seqVec2)
    else:
        return npmetrics.np_hamming_distance(seqVec1, seqVec2)

def seq_similarity(seq1, seq2, subst = None, normed = True, asDistance = False, asStrings = False):
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
        otherwise its just the sum of the raw score out of the subst matrix"""

    if subst is None:
        print 'Using default binarySubst matrix with binGaps for seq_similarity'
        subst = matrices.addGapScores(matrices.binarySubst, matrices.binGapScores)

    if asStrings:
        assert isinstance(str1, basestring), "Seq1 is not a string."
        assert isinstance(str2, basestring), "Seq2 is not a string."
        assert isinstance(subst, dict), "Subst is not a dict."
        return strmetrics.str_seq_similarity(seq1, seq2, subst = subst, normed = normed, asDistance = asDistance)

    if isinstance(seq1,basestring):
        seq1 = seq2vec(seq1)
    if isinstance(seq2,basestring):
        seq2 = seq2vec(seq2)
    if isinstance(subst,dict):
        subst = subst2mat(subst)

    if NB_SUCCESS:
        return nbmetrics.nb_seq_similarity(seq1, seq2, subst, normed, asDistance)
    else:
        return npmetrics.np_seq_similarity(seq1, seq2, subst, normed, asDistance)

def seq_distance(seq1, seq2, subst = None, normed = True, asStrings = False):
    """Compare two sequences and return the distance from one to the other
    If the seqs are of different length then it raises an exception

    Returns a scalar [0, siteN] where siteN ignores nan similarities which may depend on gaps
    Optionally returns normed = True distance:
        [0, 1]

    Note that either way the distance is "normed", its either per site (True) or total normed (False):
        [0, siteN]"""
    return seq_similarity(seq1, seq2, subst = subst, normed = normed, asDistance = True, asStrings = asStrings)


def unalign_similarity(seq1, seq2, subst = None):
    """Compare two sequences by aligning them first with pairwise alignment
       and return the distance from one to the other"""
    
    if subst is None:
        subst = matrices.blosum90

    res = pairwise2.align.globaldx(seq1, seq2, subst)
    return res[0][2]

def _distance_rect_factory(metric,nargs): 
    """Can be passed a numba jit'd distance function and
    will return a jit'd function for computing all pairwise distances using that function"""

    def distance_rect0(pwdist,symetric,seq_vecs1,seq_vecs2): 
        n,m = pwdist.shape
        for i in range(n): 
            for j in range(m): 
                if not symetric:
                    pwdist[i,j] = metric(seq_vecs1[i,:], seq_vecs2[j,:])
                else:
                    if j <= i:
                        pwdist[i,j] = metric(seq_vecs1[i,:], seq_vecs2[j,:])
                        pwdist[j,i] = pwdist[i,j]
        return True
    def distance_rect1(pwdist,symetric,seq_vecs1,seq_vecs2,arg1): 
        n,m = pwdist.shape
        for i in range(n): 
            for j in range(m): 
                if not symetric:
                    pwdist[i,j] = metric(seq_vecs1[i,:], seq_vecs2[j,:], arg1)
                else:
                    if j <= i:
                        pwdist[i,j] = metric(seq_vecs1[i,:], seq_vecs2[j,:], arg1)
                        pwdist[j,i] = pwdist[i,j]
        return True
    def distance_rect2(pwdist,symetric,seq_vecs1,seq_vecs2,arg1,arg2): 
        n,m = pwdist.shape
        for i in range(n): 
            for j in range(m): 
                if not symetric:
                    pwdist[i,j] = metric(seq_vecs1[i,:], seq_vecs2[j,:], arg1, arg2)
                else:
                    if j <= i:
                        pwdist[i,j] = metric(seq_vecs1[i,:], seq_vecs2[j,:], arg1, arg2)
                        pwdist[j,i] = pwdist[i,j]
        return True
    def distance_rect3(pwdist,symetric,seq_vecs1,seq_vecs2,arg1,arg2,arg3): 
        n,m = pwdist.shape
        for i in range(n): 
            for j in range(m): 
                if not symetric:
                    pwdist[i,j] = metric(seq_vecs1[i,:], seq_vecs2[j,:], arg1, arg2, arg3)
                else:
                    if j <= i:
                        pwdist[i,j] = metric(seq_vecs1[i,:], seq_vecs2[j,:], arg1, arg2, arg3)
                        pwdist[j,i] = pwdist[i,j]
        return True

    distance_rect = [distance_rect0,distance_rect1,distance_rect2,distance_rect3][nargs]
    
    if isinstance(metric,nb.targets.registry.CPUOverloaded):
        #return distance_rect
        return nb.jit(distance_rect, nopython=True) 
    else:
        return distance_rect

def distance_rect(row_seqs, col_seqs, metric, args = (), normalize = False, symetric = False):
    """Returns a rectangular matrix with rows and columns of the sequences in row_seqs and col_seqs.
    
    Optionally will normalize by subtracting off the min() to eliminate negative distances.
    (however, this may not be a very good way to normalize many times)

    If symetric is True then only calculates dist[i,j] and assumes dist[j,i] == dist[i,j]

    Parameters
    ----------
    row_seqs : collection
        Genetic sequences to compare.
    col_seqs : collection
        Genetic sequences to compare.
    metric : function or numba jit'd function with params seq1, seq2 (int8) and args
        Function will be called to compute each pairwise metric.
    args : tuple
        Ordinal arguments to be passed to the metric after seq1 and seq2.
        Typically includes subst as a dict or a substMat, normed, asDistance, mmTolerance, etc.
    normalize : bool
        If true (default: False), subtracts off min() to eliminate negative values
        (Could be improved/expanded)
    symetric : bool
        If True (default: False), then it assumes row_seqs and col_seqs are identical
        and speeds up computation.

    Returns
    -------
    pw : ndarray of shape [len(row_seqs), len(col_seqs)]
        Contains all pairwise metrics for seqs."""
    
    nargs = len(args)

    argList = list(args)
    if nargs > 0:
        if type(args[0]) is dict:
            argList[0] = matrices.subst2mat(args[0])
        elif args[0] is None:
            argList[0] = matrices.subst2mat(matrices.addGapScores(matrices.binarySubst, matrices.binGapScores))

    if not isinstance(row_seqs,np.ndarray) or not row_seqs.dtype == np.int8:
        row_vecs = seqs2mat(row_seqs)
        col_vecs = seqs2mat(col_seqs)
    else:
        row_vecs = row_seqs
        col_vecs = col_seqs

    if row_vecs.shape[0] != col_vecs.shape[0]:
        symetric = False

    """Only compute distances on unique sequences. De-uniquify with inv_uniqi later"""
    uRowVecs, row_uniqi, row_inv_uniqi = _unique_rows(row_vecs,return_index=True,return_inverse=True)
    uColVecs, col_uniqi, col_inv_uniqi = _unique_rows(col_vecs,return_index=True,return_inverse=True)

    drectFunc = _distance_rect_factory(metric, nargs)
    
    pw = np.zeros((uRowVecs.shape[0],uColVecs.shape[0]),dtype = np.float64)

    success = drectFunc(pw, symetric, uRowVecs, uColVecs, *tuple(argList))
    assert success

    if normalize:
        pw = pw - pw.min()

    """De-uniquify such that dist is now shape [len(seqs1), len(seqs2)]"""
    pw = pw[row_inv_uniqi,:][:,col_inv_uniqi]

    return pw

