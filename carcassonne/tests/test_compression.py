# Imports {{{
from numpy import complex128

from ..compression import *
from ..data import NDArrayData
from . import *
# }}}

class TestCompression(TestCase): # {{{
    @with_checker # test_computeProductCompressor {{{
    def test_computeProductCompressor(self,oldplus=irange(0,5),new=irange(1,5),l=irange(1,10),r=irange(1,10)):
        old = new + oldplus
        Lc1 = NDArrayData.newRandom(l,new,new)
        Lc1 += Lc1.transpose(0,2,1).conj()
        L = NDArrayData.newZeros((l,old,old),dtype=complex128)
        L._arr[:,:new,:new] = Lc1._arr

        Rc1 = NDArrayData.newRandom(new,new,r)
        Rc1 += Rc1.transpose(1,0,2).conj()
        R = NDArrayData.newZeros((old,old,r),dtype=complex128)
        R._arr[:new,:new,:] = Rc1._arr

        compressor = computeProductCompressor(L,R,new).transpose()

        Lc2 = L.absorbMatrixAt(1,compressor).absorbMatrixAt(2,compressor.conj())
        Rc2 = R.absorbMatrixAt(0,compressor.conj()).absorbMatrixAt(1,compressor)

        self.assertDataAlmostEqual(Lc1.contractWith(Rc1,(1,2),(0,1)),Lc2.contractWith(Rc2,(1,2),(0,1)))
    # }}}
# }}}
