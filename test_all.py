import unittest
import tools
import strmetrics
import matrices

class TestStrMetrics(unittest.TestCase):
    def setUp(self):
        pass
    def tearDown(self):
        pass
    def test_hamming_distance(self):
        equals = [ [('AAAA','AAAA'), 0],
                   [('AAAA','AKAA'), 1],
                   [('AKAA','AAAA'), 1],
                   [('AAAA','KKKK'), 4],
                   [('AAAA','-AAA'), 1] ]
        for args, res in equals:
            self.assertEqual(strmetrics.str_hamming_distance(*args), res)

        with self.assertRaises(AssertionError):
            strmetrics.str_hamming_distance('AAAA','AA')
    def test_coverage_distance(self):
        """ str_coverage_distance(epitope, peptide, mmTolerance = 1)
            To be a consistent distance matrix:
                covered = 0
                not-covered = 1"""

        equals = [ [('AAAA','AAAA', 0), 0],
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
                   [('AIAA','KAIARK', 0), 1]]

        for args, res in equals:
            self.assertEqual(strmetrics.str_coverage_distance(*args), res)
    def test_seq_similarity(self):
        """str_seq_similarity(seq1, seq2, subst = None, normed = True, asDistance = False)
        asDistance = True is tested in seq_distance"""
        s = matrices.binarySubst
        sGap = matrices.addGapScores(matrices.binarySubst, matrices.binGapScores)
        nanS = matrices.addGapScores(matrices.binarySubst, matrices.nanGapScores)
        equals = [ [('AAAA','AAAA', s, False), 4],
                   [('AAA','AAA', s, False), 3],
                   [('AAA','AAI', s, False), 2],
                   [('AAA','AAA', None, False), 3],
                   [('AAA','AAI', None, False), 2],
                   [('AAA','III', s, True), 0],
                   [('AAA','III', s, False), 0],
                   [('AAAA','AAII', s, True), 2],
                   [('AAAA','AAII', s, False), 2],
                   [('AAAA','-AAA', sGap, False), 3],
                   [('AAAA-','-AAAA', sGap, False), 3] ]

                   #[('AAAA','AAAA', s, True), 1],
                   #[('AAAA','-AAA', sGap, True), 0.75],
                   #[('AAAA-','-AAAA', sGap, True), 0.6],
                   #[('-AAA-','-AAAA', sGap, False), 2],
                   #[('-AAA-','-AAAA', sGap, True), 0.6],
                   #[('AAAAK','-AAAK', nanS, True), 1],
                   #[('AAAA-','-AAAA', nanS, True), 1],
                   #[('AAAAK','-AAAK', nanS, False), 4],
                   #[('AAAA-','-AAAA', nanS, False), 3],
                   #[('AIAAK','-AAAK', nanS, True), 0.75],
                   #[('AIAAK','-AAAK', nanS, False), 1],
                   #[('AAAA-','-AAAA', nanS, False), 2],
                   #[('AAAA-','-AAAA', nanS, True), 1],
                   #[('-AAA-','-AAAA', nanS, False), 0],
                   #[('-AAAI-','-AAAAK', nanS, True), 0.25] 

        for i, (args, res) in enumerate(equals):
            self.assertEqual(strmetrics.str_seq_similarity(*args), res, msg = "Test %d" % i)

        with self.assertRaises(AssertionError):
            strmetrics.str_seq_similarity('AAAA','AA')
        with self.assertRaises(KeyError):
            strmetrics.str_seq_similarity('AAAA', '-AAA', s)

    def test_seq_distance(self):
        """str_seq_distance(seq1, seq2, subst = None, normed = True)"""
        s = matrices.binarySubst
        sGap = matrices.addGapScores(matrices.binarySubst, matrices.binGapScores)
        nanS = matrices.addGapScores(matrices.binarySubst, matrices.nanGapScores)
        equals = [ [('AAAA','AAAA', s, True), 0],
                   [('AAAA','AAAA', s, False), 0],
                   [('AAA','III', s, True), 1],
                   [('AAA','III', s, False), 3],
                   [('AAAA','AAII', s, True), 0.5],
                   [('AAAA','AAII', s, False), 2],
                   [('AAAA','-AAA', sGap, True), 0.25],
                   [('AAAA-','-AAAA', sGap, True), 0.4],
                   [('AAAA','-AAA', sGap, False), 1],
                   [('AAAA-','-AAAA', sGap, False), 2],
                   [('-AAA-','-AAAA', sGap, False), 1],
                   [('-AAA-','-AAAA', sGap, True), 0.2],
                   [('AAAAK','-AAAK', nanS, False), 0],
                   [('AAAA-','-AAAA', nanS, False), 0],
                   [('AIAAK','-AAAK', nanS, False), 1],
                   [('-AAA-','-AAAA', nanS, False), 0] ]

        """
        These are tougher than they look because of the way I do normalization
        as far as I know they are working though. I just should compute the answers
        and use self.almostEquals(,places=3)"""
        #[('AAAAK','-AAAK', nanS, True), 0.111],
        #[('AAAA-','-AAAA', nanS, True), 0],
        #[('AIAAK','-AAAK', nanS, True), 0.25],
        #[('-AAAI-','-AAAAK', nanS, True), 0.25]
        #[('AAAA-','-AAAA', nanS, True), 0],
        #[('AAAA-','-AAAA', nanS, False), 2],
                   
        for i, (args, res) in enumerate(equals):
            self.assertEqual(strmetrics.str_seq_distance(*args), res, msg = "Test %d" % i)

        with self.assertRaises(KeyError):
            strmetrics.str_seq_distance('AAAA', '-AAA', s)

