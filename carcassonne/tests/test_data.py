# Imports {{{
from . import *
from ..data import *
# }}}

class TestNDArrayData(TestCase): # {{{
    @with_checker # newEnlargener {{{
    def test_newEnlargener(self,old=irange(1,10),new=irange(10,20)):
        m1, m2 = NDArrayData.newEnlargener(old,new)
        self.assertEqual(m1.shape,(new,old))
        self.assertEqual(m2.shape,(new,old))
        self.assertDataAlmostEqual(m1.contractWith(m2,(0,),(0,)),m1.newIdentity(old))
        self.assertDataAlmostEqual(m2.contractWith(m1,(0,),(0,)),m2.newIdentity(old))
    # }}}
    @with_checker # absorbMatrixAt {{{
    def test_absorbMatrixAt(self,ndim=irange(1,5),n=irange(1,5)):
        axis = randint(0,ndim-1)
        tensor = NDArrayData.newRandom(*(randint(1,n) for _ in range(ndim)))
        matrix = NDArrayData.newRandom(tensor.shape[axis],tensor.shape[axis])
        new_axes = list(range(1,ndim))
        new_axes.insert(axis,0)
        self.assertDataAlmostEqual(
            tensor.absorbMatrixAt(axis,matrix),
            matrix.contractWith(tensor,(1,),(axis,)).join(*new_axes),
        )
    # }}}
# }}}
