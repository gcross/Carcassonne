# Imports {{{
from . import *
from ..data import *
# }}}

class TestNDArrayData(TestCase): # {{{
    @with_checker # increaseDimension w/ vectors {{{
    def test_increaseDimension_vectors(self,m=irange(1,10),dm=irange(0,5)):
        A = NDArrayData.newRandom(m)
        B = NDArrayData.newRandom(m)
        new_A, matrix = A.increaseDimension(0,by=dm)
        self.assertEqual(new_A.shape,(m+dm,))
        new_B = B.absorbMatrixAt(0,matrix)
        self.assertEqual(new_B.shape,(m+dm,))
        self.assertAlmostEqual(
            A.contractWith(B,(0,),(0,)).extractScalar(),
            new_A.contractWith(new_B,(0,),(0,)).extractScalar(),
        )
    # }}}
# }}}
