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
values based on each residue's position in the FULL_ALPHABET.

Sequence alignment as a matrix:
A genetic alignment (assumes same length for all sequences)
is represented as a 2d numpy int8 matrix [seqs x sites] with values
based on each residue's position in the FULL_ALPHABET.

"""
import numpy as np
import re
from Bio import pairwise2


__all__ = ['BADAA',
           'AALPHABET',
           'AA2CODE',
           'CODE2AA',
           'isvalidpeptide',
           'removeBadAA',
           'seq2vec',
           'seq_similarity',
           'seq_distance',
           'unalign_similarity',
           '_test_seq_similarity',
           'calcDistanceMatrix',
           'calcDistanceRectangle',
           'aamismatch_distance']
           
BADAA = '-*BX#Z'
FULL_AALPHABET = 'ABCDEFGHIKLMNPQRSTVWXYZ-'
AALPHABET = 'ACDEFGHIKLMNPQRSTVWY'
AA2CODE = {aa:i for i,aa in enumerate(FULL_AALPHABET)}
AA2CODE.update({'-':23})
CODE2AA = {i:aa for i,aa in enumerate(FULL_AALPHABET)}
CODE2AA.update({23:'-'})

def isvalidpeptide(mer,badaa=None):
    """Test if the mer contains an BAD amino acids in global BADAA
    typically -*BX#Z"""
    if badaa is None:
        badaa = BADAA
    if not mer is None:
        return not re.search('[%s]' % badaa,mer)
    else:
        return False
def removeBadAA(mer,badaa=None):
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
        vec[aai] = AA2CODE[aa]
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
            mat[si,aai] = AA2CODE[aa]
    return mat

def aamismatch_distance(seq1,seq2, **kwargs):
    """No substitution matrix used."""
    if isinstance(seq1,basestring):
        seq1 = seq2vec(seq1)

    if isinstance(seq2,basestring):
        seq2 = seq2vec(seq2)
    dist12 = nb_seq_similarity(seq1, seq2, substMat = binaryMat, normed = False, asDistance = True)
    return dist12

def hamming_distance(str1, str2, noConvert = False, **kwargs):
    """Hamming distance between str1 and str2.
    Only finds distance over the length of the shorter string.
    **kwargs are so this can be plugged in place of a seq_distance() metric"""
    if noConvert:
        return str_hamming_distance(str1,str2)

    if isinstance(str1,basestring):
        str1 = string2byte(str1)
    if isinstance(str2,basestring):
        str2 = string2byte(str2)
    return nb_hamming_distance(str1, str2)

def seq_similarity(seq1, seq2, subst = None, normed = True, asDistance = False):
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
        subst = addGapScores(binarySubst, binGapScores)

    if isinstance(subst,dict):
        subst = subst2mat(subst)

    if isinstance(seq1,basestring):
        seq1 = seq2vec(seq1)

    if isinstance(seq2,basestring):
        seq2 = seq2vec(seq2)

    result = np_seq_similarity(seq1, seq2, substMat = subst, normed = normed, asDistance = asDistance)
    return result


def seq_distance(seq1, seq2, subst = None, normed = True):
    """Compare two sequences and return the distance from one to the other
    If the seqs are of different length then it raises an exception

    Returns a scalar [0, siteN] where siteN ignores nan similarities which may depend on gaps
    Optionally returns normed = True distance:
        [0, 1]

    Note that either way the distance is "normed", its either per site (True) or total normed (False):
        [0, siteN]"""
    return seq_similarity(seq1, seq2, subst = subst, normed = normed, asDistance = True)


def unalign_similarity(seq1,seq2,subst=None):
    """Compare two sequences by aligning them first with pairwise alignment
       and return the distance from one to the other"""
    
    if subst is None:
        subst = blosum90

    res = pairwise2.align.globaldx(seq1,seq2,subst)
    return res[0][2]

def _test_seq_similarity(subst=None,normed=True):
    def test_one(s,sname,n,seq1,seq2):
        print seq1
        print seq2
        try:
            sim = seq_similarity_old(seq1, seq2, subst=s, normed=n)
            print 'Similarity: %f' % sim
        except:
            print 'Similarity: %s [%s]' % (sys.exc_info()[0],sys.exc_info()[1])
        
        #dist = seq_distance(seq1,seq2,subst=s)
        try:
            dist = seq_distance(seq1, seq2, subst=s)
            print 'Distance: %f' % dist
        except:
            print 'Distance: %s [%s]' % (sys.exc_info()[0],sys.exc_info()[1])
        print

    seqs = ['AAAA',
            'AAAA',
            'KKKK',
            'AAKA',
            '-AAA',
            '-A-A']
    if subst is None:
        subst = [addGapScores(binarySubst,binGapScores),
                 addGapScores(binarySubst,nanZeroGapScores),
                 addGapScores(blosum90,blosum90GapScores),
                 addGapScores(blosum90,nanGapScores)]
        names = ['addGapScores(binarySubst,binGapScores)',
                 'addGapScores(binarySubst,nanZeroGapScores)',
                 'addGapScores(blosum90,blosum90GapScores)',
                 'addGapScores(blosum90,nanGapScores)']
        for s,sname in zip(subst,names):
            print 'Using %s normed = %s' % (sname,normed)
            for seq1,seq2 in itertools.combinations(seqs,2):
                test_one(s,sname,normed,seq1,seq2)
    else:
        for seq1,seq2 in itertools.combinations(seqs,2):
            test_one(subst,'supplied subst',normed,seq1,seq2)

def calcDistanceMatrix(seqs,normalize=False,symetric=True,metric=None,**kwargs):
    """Returns a square distance matrix with rows and columns of the unique sequences in seqs
    By default will normalize by subtracting off the min() to at least get rid of negative distances
    However, I don't really think this is the best option. 
    If symetric is True then only calculates dist[i,j] and assumes dist[j,i] == dist[i,j]
    Additional kwargs are passed to the distanceFunc (e.g. subst, gapScores, normed)

    Parameters
    ----------
    seqs : list/iterator
        Genetic sequences to compare.
    normalize : bool
        If true (default: False), subtracts off dist.min() to eliminate negative distances
        (Could be improved/expanded)
    symetric : bool
        If True (default), then it assumes dist(A,B) == dist(B,A) and speeds up computation.
    metric : function with params seq1, seq2 and possibly additional kwargs
        Function will be called to compute each pairwise distance.
    kwargs : additional keyword arguments
        Will be passed to each call of metric()

    Returns
    -------
    dist : ndarray of shape [len(seqs), len(seqs)]
        Contains all pairwise distances for seqs.
    """
    return calcDistanceRectangle_old(seqs,seqs,normalize=normalize,symetric=symetric,metric=metric,**kwargs)

def calcDistanceRectangle_old(row_seqs,col_seqs,normalize=False,symetric=False,metric=None,convertToNP=False,**kwargs):
    """Returns a rectangular distance matrix with rows and columns of the unique sequences in row_seqs and col_seqs
    By default will normalize by subtracting off the min() to at least get rid of negative distances
    However, I don't really think this is the best option. 
    If symetric is True then only calculates dist[i,j] and assumes dist[j,i] == dist[i,j]
    
    Additional kwargs are passed to the distanceFunc (e.g. subst, gapScores, normed)

    Parameters
    ----------
    row_seqs : list/iterator
        Genetic sequences to compare.
    col_seqs : list/iterator
        Genetic sequences to compare.
    normalize : bool
        If true (default: False), subtracts off dist.min() to eliminate negative distances
        (Could be improved/expanded)
    symetric : bool
        If True (default), then it assumes dist(A,B) == dist(B,A) and speeds up computation.
    metric : function with params seq1, seq2 and possibly additional kwargs
        Function will be called to compute each pairwise distance.
    convertToNP : bool (default: False)
        If True then strings are converted to np.arrays for speed,
        but metric will also need to accomodate the arrays as opposed to strings
    kwargs : additional keyword arguments
        Will be passed to each call of metric()

    Returns
    -------
    dist : ndarray of shape [len(row_seqs), len(col_seqs)]
        Contains all pairwise distances for seqs.
    """
    if not 'normed' in kwargs.keys():
        kwargs['normed'] = False
    if metric is None:
        metric = seq_distance

    """Only compute distances on unique sequences. De-uniquify with inv_uniqi later"""
    row_uSeqs,row_uniqi,row_inv_uniqi = np.unique(row_seqs,return_index=True,return_inverse=True)
    col_uSeqs,col_uniqi,col_inv_uniqi = np.unique(col_seqs,return_index=True,return_inverse=True)

    if convertToNP:
        R = [seq2vec(s) for s in row_uSeqs]
        C = [seq2vec(s) for s in col_uSeqs]
    else:
        R = row_uSeqs
        C = col_uSeqs

    dist = np.zeros((len(row_uSeqs),len(col_uSeqs)))
    for i,j in itertools.product(range(len(row_uSeqs)),range(len(col_uSeqs))):
        if not symetric:
            """If not assumed symetric, compute all distances"""
            dist[i,j] = metric(R[i],C[j],**kwargs)
        else:
            if j<i:
                tmp = metric(R[i],C[j],**kwargs)
                dist[i,j] = tmp
                dist[j,i] = tmp
            elif j>i:
                pass
            elif j==i:
                dist[i,j] = metric(R[i],C[j],**kwargs)

    if normalize:
        dist = dist - dist.min()
    """De-uniquify such that dist is now shape [len(seqs), len(seqs)]"""
    dist = dist[row_inv_uniqi,:][:,col_inv_uniqi]
    return dist

def calcDistanceRectangle(row_seqs, col_seqs, subst=None, nb_metric=None, normalize=False, symetric=False):
    """Returns a rectangular distance matrix with rows and columns of the unique sequences in row_seqs and col_seqs
    By default will normalize by subtracting off the min() to at least get rid of negative distances
    However, I don't really think this is the best option. 
    If symetric is True then only calculates dist[i,j] and assumes dist[j,i] == dist[i,j]

    TODO:
    (1) Wrap this function around dist rect functins below.
    (2) Define a coverage nb_metric
    (3) Come up with a back-up plan for when numba import fails...
        (Not jit'ing is not a good option because it will be super slow!
            There need to be numpy equivalent functions as back-up...)

    
    Additional kwargs are passed to the distanceFunc (e.g. subst, gapScores, normed)

    Parameters
    ----------
    row_seqs : list/iterator
        Genetic sequences to compare.
    col_seqs : list/iterator
        Genetic sequences to compare.
    subst : dict or ndarray
        Similarity matrix for use by the metric. Can be subst or substMat (i.e. dict or ndarray)
    nb_metric : numba jit'd function with params seq_vecs1, seq_vecs2 (int8) and substMat (and others?)
        Function will be called to compute each pairwise distance.
    normalize : bool
        If true (default: False), subtracts off dist.min() to eliminate negative distances
        (Could be improved/expanded)
    symetric : bool
        If True (default), then it assumes dist(A,B) == dist(B,A) and speeds up computation.
    

    Returns
    -------
    dist : ndarray of shape [len(row_seqs), len(col_seqs)]
        Contains all pairwise distances for seqs.
    """
    if not 'normed' in kwargs.keys():
        kwargs['normed'] = False
    if metric is None:
        metric = seq_distance

    """Only compute distances on unique sequences. De-uniquify with inv_uniqi later"""
    row_uSeqs,row_uniqi,row_inv_uniqi = np.unique(row_seqs,return_index=True,return_inverse=True)
    col_uSeqs,col_uniqi,col_inv_uniqi = np.unique(col_seqs,return_index=True,return_inverse=True)

    if convertToNP:
        R = [seq2vec(s) for s in row_uSeqs]
        C = [seq2vec(s) for s in col_uSeqs]
    else:
        R = row_uSeqs
        C = col_uSeqs

    dist = zeros((len(row_uSeqs),len(col_uSeqs)))
    for i,j in itertools.product(range(len(row_uSeqs)),range(len(col_uSeqs))):
        if not symetric:
            """If not assumed symetric, compute all distances"""
            dist[i,j] = metric(R[i],C[j],**kwargs)
        else:
            if j<i:
                tmp = metric(R[i],C[j],**kwargs)
                dist[i,j] = tmp
                dist[j,i] = tmp
            elif j>i:
                pass
            elif j==i:
                dist[i,j] = metric(R[i],C[j],**kwargs)

    if normalize:
        dist = dist - dist.min()
    """De-uniquify such that dist is now shape [len(seqs), len(seqs)]"""
    dist = dist[row_inv_uniqi,:][:,col_inv_uniqi]
    return dist
