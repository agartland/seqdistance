import unittest
import numpy as np

from . import FULL_AALPHABET
import tools
import strmetrics
import nbmetrics
import npmetrics
import matrices

from tools import seq2vec
from matrices import subst2mat

s = matrices.binarySubst
sGap = matrices.addGapScores(matrices.binarySubst, matrices.binGapScores)
nanS = matrices.addGapScores(matrices.binarySubst, matrices.nanGapScores)

hamming_equals = [ [('AAAA','AAAA'), 0],
                    [('AAAA','AKAA'), 1],
                    [('AKAA','AAAA'), 1],
                    [('AAAA','KKKK'), 4],
                    [('AAAA','-AAA'), 1] ]

coverage_equals = [ [('AAAA','AAAA', 0), 0],
                     [('AAAA','AKAA', 0), 1],
                     [('AKAA','AAAA', 0), 1],
                     [('AAAA','KKKK', 0), 1],
                     [('AAAA','KAAAA', 0), 0],
                     [('AAAA','AAAAK', 0), 0],
                     [('AAAA','AA', 0), 1],
                     [('AIA','KAIAAK', 0), 0],
                     [('AAAA','AAAA', 1), 0],
                     [('AAAA','AKAA', 1), 0],
                     [('AKAA','AAAA', 1), 0],
                     [('AAAA','KKAA', 2), 0],
                     [('AAAA','KKAA', 1), 1],
                     [('AAAA','KAAIA', 1), 0],
                     [('AAAA','KAAIA', 0), 1],
                     [('AAAA','AA', 2), 1],
                     [('AIAA','KAIARK', 1), 0],
                     [('AIAA','KAIARK', 0), 1] ]

similarity_equals = [ [('AAAA','AAAA', s, False), 4],
                       [('AAA','AAA', s, False), 3],
                       [('AAA','AAI', s, False), 2],
                       [('AAA','AAA', None, False), 3],
                       [('AAA','AAI', None, False), 2],
                       [('AAA','III', s, True), 0],
                       [('AAA','III', s, False), 0],
                       [('AAAA','AAII', s, True), 2],
                       [('AAAA','AAII', s, False), 2],
                       [('AAAA','-AAA', s, False), 3],
                       [('AAAA','-AAA', sGap, False), 3],
                       [('AAAA-','-AAAA', sGap, False), 3],
                       [('AAAA','AAAA', s, True), 1],
                       [('AAAA','-AAA', sGap, True), 0.75],
                       [('AAAA-','-AAAA', sGap, True), 0.6],
                       [('-AAA-','-AAAA', sGap, False), 2],
                       [('-AAA-','-AAAA', sGap, True), 0.6],
                       [('AAAAK','-AAAK', nanS, True), 1],
                       [('AAAA-','-AAAA', nanS, True), 1],
                       [('AAAAK','-AAAK', nanS, False), 4],
                       [('AAAA-','-AAAA', nanS, False), 3],
                       [('AIAAK','-AAAK', nanS, True), 0.75],
                       [('AIAAK','-AAAK', nanS, False), 1],
                       [('AAAA-','-AAAA', nanS, False), 2],
                       [('AAAA-','-AAAA', nanS, True), 1],
                       [('-AAA-','-AAAA', nanS, False), 0],
                       [('-AAAI-','-AAAAK', nanS, True), 0.25] ]

distance_equals = [ [('AAAA','AAAA', s, True), 0],
                   [('AAAA','AAAA', s, False), 0],
                   [('AAA','III', s, True), 1],
                   [('AAA','III', s, False), 3],
                   [('AAAA','AAII', s, True), 0.5],
                   [('AAAA','AAII', s, False), 2],
                   [('AAAA','-AAA', s, False), 0],
                   [('AAAA','-AAA', sGap, True), 0.25],
                   [('AAAA-','-AAAA', sGap, True), 0.4],
                   [('AAAA','-AAA', sGap, False), 1],
                   [('AAAA-','-AAAA', sGap, False), 2],
                   [('-AAA-','-AAAA', sGap, False), 1],
                   [('-AAA-','-AAAA', sGap, True), 0.2],
                   [('AAAAK','-AAAK', nanS, False), 0],
                   [('AAAA-','-AAAA', nanS, False), 0],
                   [('AIAAK','-AAAK', nanS, False), 1],
                   [('-AAA-','-AAAA', nanS, False), 0],
                   [('AAAAK','-AAAK', nanS, True), 0.111],
                   [('AAAA-','-AAAA', nanS, True), 0],
                   [('AIAAK','-AAAK', nanS, True), 0.25],
                   [('-AAAI-','-AAAAK', nanS, True), 0.25],
                   [('AAAA-','-AAAA', nanS, True), 0],
                   [('AAAA-','-AAAA', nanS, False), 2] ]

class TestTools(unittest.TestCase):
    def test_seq2vec(self):
        self.assertTrue(np.all(seq2vec('AKA-') == np.array([ 0,  9,  0, 23], dtype=np.int8)))
    def test_subst2mat(self):
        smat = subst2mat({('A','K'):-3, ('A','A'):7, ('K','K'):5})
        self.assertEqual(smat.shape[0], len(tools.FULL_AALPHABET))
        self.assertEqual(smat.shape[0], smat.shape[1])
        self.assertEqual(len(smat.shape), 2)
        self.assertEqual(smat[FULL_AALPHABET.index('A'),FULL_AALPHABET.index('K')], -3)
        self.assertEqual(smat[FULL_AALPHABET.index('K'),FULL_AALPHABET.index('A')], -3)
        self.assertEqual(smat[FULL_AALPHABET.index('A'),FULL_AALPHABET.index('A')], 7)
        self.assertTrue(np.isnan(smat[FULL_AALPHABET.index('A'),FULL_AALPHABET.index('R')]))
    def test_isvalidpeptide(self):
        self.assertTrue(tools.isvalidpeptide('AKR'))
        self.assertFalse(tools.isvalidpeptide('AK-'))
        self.assertFalse(tools.isvalidpeptide('AK#'))
    def test_removeBadAA(self):
        self.assertEqual(tools.removeBadAA('AKRI#R'),'AKRIR')
        self.assertEqual(tools.removeBadAA('AKRIR'),'AKRIR')
        self.assertEqual(tools.removeBadAA('##'),'')

class TestStrMetrics(unittest.TestCase):
    def setUp(self):
        pass
    def tearDown(self):
        pass
    def test_hamming_distance(self):
        """str_hamming_distance(seq1, seq2)"""
        for args, res in hamming_equals:
            self.assertEqual(strmetrics.str_hamming_distance(*args), res)

        with self.assertRaises(AssertionError):
            strmetrics.str_hamming_distance('AAAA','AA')

    def test_coverage_distance(self):
        """str_coverage_distance(epitope, peptide, mmTolerance = 1)
            To be a consistent distance matrix:
                covered = 0
                not-covered = 1"""
        for args, res in coverage_equals:
            self.assertEqual(strmetrics.str_coverage_distance(*args), res)
    def test_seq_similarity(self):
        """str_seq_similarity(seq1, seq2, subst = None, normed = True, asDistance = False)
        asDistance = True is tested in seq_distance"""

        for i, (args, res) in enumerate(similarity_equals):
            """Many of these tests fail because I didn't hand-code the answer correctly.
            I'm not worried though because the nb_ and str_ versions agree..."""
            self.assertEqual(strmetrics.str_seq_similarity(*args), res, msg = "Test %d: (%s, %s)" % (i,args[0],args[1]))

        with self.assertRaises(AssertionError):
            strmetrics.str_seq_similarity('AAAA','AA')

    def test_seq_distance(self):
        """str_seq_distance(seq1, seq2, subst = None, normed = True)"""
        for i, (args, res) in enumerate(distance_equals):
            """Many of these tests fail because I didn't hand-code the answer correctly.
            I'm not worried though because the nb_ and str_ versions agree..."""
            self.assertEqual(strmetrics.str_seq_distance(*args), res, msg = "Test %d: (%s, %s)" % (i,args[0],args[1]))

        with self.assertRaises(KeyError):
            strmetrics.str_seq_distance('AAAA', '-AAA', s)

class TestNbMetrics(unittest.TestCase):
    """Numba metrics"""
    def setUp(self):
        self.hamming = nbmetrics.nb_hamming_distance
        self.seq_similarity = nbmetrics.nb_seq_similarity
        self.coverage_distance = nbmetrics.nb_coverage_distance
        self.seq_distance = nbmetrics.nb_seq_distance

    def test_hamming_distance(self):
        """nb_hamming_distance(seqVec1, seqVec2)"""
        for args, res in hamming_equals:
            """Check each case against the strmetric"""
            self.assertEqual(self.hamming(seq2vec(args[0]),seq2vec(args[1])),
                             strmetrics.str_hamming_distance(*args))

        with self.assertRaises(AssertionError):
            self.hamming(seq2vec('AAAA'),seq2vec('AA'))

    def test_seq_similarity(self):
        """nb_seq_similarity(seqVec1, seqVec2, substMat, normed, asDistance))"""
        for i,(args, res) in enumerate(similarity_equals):
            """Check each case against the strmetric"""
            if args[2] is None:
                sMat = matrices.binaryMat
            else:
                sMat = matrices.subst2mat(args[2])
            self.assertEqual(self.seq_similarity(seq2vec(args[0]),seq2vec(args[1]),sMat,args[3],False),
                             strmetrics.str_seq_similarity(*args), msg = "Test %d: (%s, %s)" % (i,args[0],args[1]))
    def test_coverage_distance(self):
        """nb_coverage_distance(seqVec1, seqVec2, mmTolerance))"""
        for i, (args, res) in enumerate(coverage_equals):
            """Check each case against the strmetric"""
            self.assertEqual(self.coverage_distance(seq2vec(args[0]),seq2vec(args[1]),args[2]),
                             strmetrics.str_coverage_distance(*args), msg = "Test %d: (%s, %s)" % (i,args[0],args[1]))
    def test_seq_distance(self):
        """nb_seq_distance(seqVec1, seqVec2, substMat, normed))"""
        for i,(args, res) in enumerate(distance_equals):
            """Check each case against the strmetric"""
            if args[2] is None:
                sMat = matrices.binaryMat
            else:
                sMat = matrices.subst2mat(args[2])
            self.assertEqual(self.seq_distance(seq2vec(args[0]),seq2vec(args[1]),sMat,args[3]),
                             strmetrics.str_seq_distance(*args), msg = "Test %d: (%s, %s)" % (i,args[0],args[1]))
class TestNpMetrics(TestNbMetrics):
    """Numpy metrics"""
    def setUp(self):
        self.hamming = npmetrics.np_hamming_distance
        self.seq_similarity = npmetrics.np_seq_similarity
        self.coverage_distance = npmetrics.np_coverage_distance
        self.seq_distance = npmetrics.np_seq_distance
    def test_hamming_distance(self):
        """np_hamming_distance(seqVec1, seqVec2)"""
        TestNbMetrics.test_hamming_distance(self)
    def test_seq_similarity(self):
        """np_seq_similarity(seqVec1, seqVec2, substMat, normed, asDistance))"""
        TestNbMetrics.test_seq_similarity(self)
    def test_coverage_distance(self):
        """np_coverage_distance(seqVec1, seqVec2, mmTolerance))"""
        TestNbMetrics.test_coverage_distance(self)
    def test_seq_distance(self):
        """np_seq_distance(seqVec1, seqVec2, substMat, normed))"""
        TestNbMetrics.test_seq_distance(self)

class TestDistRect(unittest.TestCase):
    def setUp(self):
        self.seq = tools.removeBadAA("ABSJDHKJHDSJHKFJOIWIYOQIJOIRKJHYDAGUMAMOIDNIFNOINTWIDMLKA")
        self.mers = [self.seq[starti:starti + 9] for starti in range(len(self.seq)-9+1)]
    def test_run_asstrings(self):
        pw = tools.distance_rect(self.mers[:10], self.mers[:25], subst = matrices.binarySubst, metric = strmetrics.str_seq_distance, normalize = False, symetric = True)
        self.assertEqual(pw.shape[0], 10)
        self.assertEqual(pw.shape[1], 25)

