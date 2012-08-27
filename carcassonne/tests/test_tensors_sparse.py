# Imports {{{
from numpy import any, isnan
from paycheck import *

from ..sparse import mapSparseChunkValues
from ..tensors.dense import *
from ..tensors.sparse import *
from ..tensors.sparse import formExpectationStage1, formExpectationStage2, formExpectationStage3
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
        C1 = absorbSparseSideIntoCornerFromLeft(direction,A,B)
        C1D = NDArrayData(formDenseTensor(C1,NDArrayData.toArray,shape=(dc,db*dd*de)))
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
        C1 = absorbSparseSideIntoCornerFromRight(direction,A,B)
        C1D = NDArrayData(formDenseTensor(C1,NDArrayData.toArray,shape=(da*dd*de,dc)))
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
        D1 = absorbSparseCenterSOSIntoSide(i,A,B,C)
        D1D = NDArrayData(formDenseTensor(D1,toArray=NDArrayData.toArray,shape=(da*B_shape[L(i)]**2,db*B_shape[R(i)]**2,B_shape[O(i)],B_shape[O(i)])))
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

class TestSparseStages(TestCase): # {{{
    @with_checker
    def test_stage_1(self, # {{{
        sa = irange(5,10),
        sb = irange(5,10),
        sc = irange(5,10),
        sd = irange(5,10),
        da = irange(1,5),
        db = irange(1,5),
        dc = irange(1,5),
        dd = irange(1,5),
        de = irange(1,5),
    ):
        A = randomSparseTensor((sa,sb),(da,db),4)
        B = randomSparseTensor((sb,sc,sd),(db,dc,dd,de),4)
        AD = NDArrayData(formDenseTensor(A,toArray=NDArrayData.toArray))
        BD = NDArrayData(formDenseTensor(B,toArray=NDArrayData.toArray))
        C1 = formExpectationStage1(A,B)
        C1D = NDArrayData(formDenseTensor(C1,toArray=NDArrayData.toArray,shape=(da,dc,dd,de)))
        C2D = AD.contractWith(BD,(1,3),(0,3)).join(0,2,3,1,4,5,6)
        self.assertDataAlmostEqual(C1D,C2D)
    # }}}
    @with_checker
    def test_stage_2(self, # {{{
        sa = irange(5,8),
        sb = irange(5,8),
        sc = irange(5,8),
        sd = irange(5,8),
        se = irange(5,8),
        da = irange(1,3),
        db = irange(1,3),
        dc = irange(1,3),
        dd = irange(1,3),
        de = irange(1,3),
    ):
        A = randomSparseTensor((sa,sb,sc),(da,db,dc,dc),4)
        B = randomSparseTensor((sd,sa,se),(dd,da,de,de),4)
        AD = NDArrayData(formDenseTensor(A,toArray=NDArrayData.toArray))
        BD = NDArrayData(formDenseTensor(B,toArray=NDArrayData.toArray))
        C1 = formExpectationStage2(A,B)
        C1D = NDArrayData(formDenseTensor(C1,toArray=NDArrayData.toArray,shape=(dd,db,dc,de,dc,de)))
        C2D = AD.contractWith(BD,(0,3),(1,4)).join(5,0,1,6,7,2,3,8,4,9)
        self.assertDataAlmostEqual(C1D,C2D)
    # }}}
    @with_checker
    def test_stage_3(self, # {{{
        sa = irange(5,6),
        sb = irange(5,6),
        sc = irange(5,6),
        sd = irange(5,6),
        se = irange(5,6),
        sf = irange(5,6),
        da = irange(1,2),
        db = irange(1,2),
        dc = irange(1,2),
        dd = irange(1,2),
        de = irange(1,2),
        df = irange(1,2),
        dp = irange(1,2),
    ):
        A = randomSparseTensor((sa,sb,sc,sd),(da,db,dc,dd,dc,dd),4)
        B = randomSparseTensor((sb,sa,se,sf),(db,da,de,df,de,df),4)
        C = randomSparseTensor((sc,sd,se,sf),(dp,dp),4)
        D = NDArrayData.newRandom(dc,dd,de,df,dp)
        AD = NDArrayData(formDenseTensor(A,toArray=NDArrayData.toArray))
        BD = NDArrayData(formDenseTensor(B,toArray=NDArrayData.toArray))
        CD = NDArrayData(formDenseTensor(C,toArray=NDArrayData.toArray))
        C1D = formExpectationStage3(A,B,C)(D)
        C2D = formDataContractor(
            [
                Join(0,(6,7),2,(0,1)),
                Join(3,5,2,4),
                Join(1,(6,7),2,(2,3)),
                Join(0,(0,1,4,5),1,(1,0,5,4)),
                Join(0,(2,3),3,(0,1)),
                Join(1,(2,3),3,(2,3)),
            ],[
                [(0,8)],
                [(0,9)],
                [(1,8)],
                [(1,9)],
                [(3,4)],
            ]
        )(AD,BD,D,CD)
        try:
            self.assertDataAlmostEqual(C1D,C2D,rtol=1e-3,atol=1e-3)
        except:
            if not any(isnan(C2D.toArray())):
                raise
    # }}}
# }}}
