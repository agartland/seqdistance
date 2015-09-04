import unittest
import sys

class TestImport(unittest.TestCase):
    def test_sd(self):
        import seqdistance as sd
        try:
            import seqdistance as sd
            success = True
        except ImportError:
            success = False
        self.assertTrue(success)
        self.assertTrue(hasattr(sd,'FULL_AALPHABET'))
        self.assertTrue(hasattr(sd,'metrics'))