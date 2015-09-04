global NB_SUCCESS

try:
    import numba as nb
    print 'seqdistance: Successfully imported numba version %s' % (nb.__version__)
    NB_SUCCESS = True
except OSError:
    NB_SUCCESS = False
    try:
        """On Windows it is neccessary to be on the same drive as the LLVM DLL
        in order to import numba without generating a "Windows Error 161: The specified path is invalid."""
        curDir = os.getcwd()
        targetDir = os.path.splitdrive(sys.executable)[0]
        os.chdir(targetDir)
        import numba as nb
        import nbmetrics
        print 'seqdistance: Successfully imported numba version %s' % (nb.__version__)
        NB_SUCCESS = True
    except OSError:
        NB_SUCCESS = False
        print 'seqdistance: Could not load numba\n(may be a path issue try starting python in C:\\)'
    finally:
        os.chdir(curDir)
        
except ImportError:
    NB_SUCCESS = False
    print 'seqdistance: Could not load numba'

from .strmetrics import *
from .npmetrics import *

__all__ = ['str_hamming_distance',
           'trunc_hamming',
           'dichot_hamming',
           'str_coverage_distance',
           'str_seq_distance',
           'str_seq_similarity',
           'np_seq_similarity',
           'np_hamming_distance',
           'np_seq_distance',
           'np_coverage_distance']

if NB_SUCCESS:
    from .nbmetrics import *
    __all__.append(['nb_hamming_distance',
                   'nb_seq_similarity',
                   'nb_seq_distance',
                   'nb_coverage_distance'])