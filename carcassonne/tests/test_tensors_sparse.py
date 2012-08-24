# Imports {{{
from paycheck import *

from ..sparse import mapSparseChunkValues
from ..tensors.dense import *
from ..tensors.sparse import *
from ..utils import *
from . import *
# }}}

class TestSparseCorner(TestCase): # {{{
    @with_checker
    def test_absorbFromLeft_nospecials(self, # {{{
        sa = irange(5,10),
        sb = irange(5,10),
        sc = irange(5,10),
        sd = irange(5,10),
        da = irange(1,5),
        db = irange(1,5),
        dc = irange(1,5),
        dd = irange(1,5),
        de = irange(1,5),
        direction = irange(0,3),
    ):
        A = randomSparseTensor((sa,sb),(da,db),4)
        B = randomSparseTensor((sc,sa,sd),(dc,da,dd,de),4)
        AD = NDArrayData(formDenseTensor(A,toArray=NDArrayData.toArray))
        BD = NDArrayData(formDenseTensor(B,toArray=NDArrayData.toArray))
        C1 = SparseCorner(mapSparseChunkValues(DenseCorner,A)).absorbFromLeft(direction,SparseSide(mapSparseChunkValues(DenseSide,B))).tensor
        C1D = NDArrayData(formDenseTensor(C1,toArray=lambda x: x.data.toArray(),shape=(dc,db*dd*de)))
        C2D = AD.contractWith(BD,(0,2),(1,4)).join(2,(0,3),4,(1,5,6))
        self.assertDataAlmostEqual(C1D,C2D)
    # }}}
    @with_checker
    def test_absorbFromRight_nospecials(self, # {{{
        sa = irange(5,10),
        sb = irange(5,10),
        sc = irange(5,10),
        sd = irange(5,10),
        da = irange(1,5),
        db = irange(1,5),
        dc = irange(1,5),
        dd = irange(1,5),
        de = irange(1,5),
        direction = irange(0,3),
    ):
        A = randomSparseTensor((sa,sb),(da,db),4)
        B = randomSparseTensor((sb,sc,sd),(db,dc,dd,de),4)
        AD = NDArrayData(formDenseTensor(A,toArray=NDArrayData.toArray))
        BD = NDArrayData(formDenseTensor(B,toArray=NDArrayData.toArray))
        C1 = SparseCorner(mapSparseChunkValues(DenseCorner,A)).absorbFromRight(direction,SparseSide(mapSparseChunkValues(DenseSide,B))).tensor
        C1D = NDArrayData(formDenseTensor(C1,toArray=lambda x: x.data.toArray(),shape=(da*dd*de,dc)))
        C2D = AD.contractWith(BD,(1,3),(0,3)).join((0,3),2,(1,5,6),4)
        self.assertDataAlmostEqual(C1D,C2D)
    # }}}
#}}}

class TestSparseSide(TestCase): # {{{
    @with_checker(number_of_calls=20)
    def test_absorbCenterSOS(self, # {{{
        sa = irange(5,6),
        sb = irange(5,6),
        sc = irange(5,6),
        sd = irange(5,6),
        se = irange(5,6),
        sf = irange(5,6),
        sg = irange(5,6),
        da = irange(1,2),
        db = irange(1,2),
        dc = irange(1,2),
        dd = irange(1,2),
        de = irange(1,2),
        df = irange(1,2),
        dg = irange(1,2),
        dp = irange(1,2),
        i = irange(0,3),
    ):
        A = randomSparseTensor((sa,sb,sc),(da,db,dc,dc),4)
        B_shape = replaceAt((dd,de,df,dg,dp),i,dc)
        B = NDArrayData.newRandom(*B_shape)
        C = randomSparseTensor(replaceAt((sd,se,sf,sg),i,sc),(dp,dp),4)
        AD = NDArrayData(formDenseTensor(A,toArray=NDArrayData.toArray))
        CD = NDArrayData(formDenseTensor(C,toArray=NDArrayData.toArray))
        D1 = SparseSide(mapSparseChunkValues(DenseSide,A)).absorbCenterSOS(i,B,C).tensor
        D1D = NDArrayData(formDenseTensor(D1,toArray = lambda x: x.data.toArray(),shape=(da*B_shape[L(i)]**2,db*B_shape[R(i)]**2,B_shape[O(i)],B_shape[O(i)])))
        D2D = formDataContractor(
            [
                Join(0,5,1,i),
                Join(0,2,2,i),
                Join(0,6,3,i),
                Join(1,4,2,5),
                Join(3,4,2,4),
            ],
            [
                [(0,0),(2,L(i))],
                [(0,1),(2,R(i))],
                [(2,O(i))],
                [(0,3),(1,L(i)),(3,L(i))],
                [(0,4),(1,R(i)),(3,R(i))],
                [(1,O(i))],
                [(3,O(i))],
            ]
        )(AD,B,CD,B.conj())
        self.assertDataAlmostEqual(D1D,D2D,atol=1e-3,rtol=1e-3)
    # }}}
# }}}
