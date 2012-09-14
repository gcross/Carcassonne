# Imports {{{
from numpy import dot, multiply
from paycheck import *

from ..data import NDArrayData
from ..tensors.dense import *
from ..tensors.dense import formNormalizationStage1, formNormalizationStage2, formNormalizationStage3
from . import *
# }}}

class TestDenseCorner(TestCase): # {{{
    @with_checker
    def test_absorbFromLeft(self, # {{{
        a = irange(1,10),
        b = irange(1,10),
        c = irange(1,10),
        d = irange(1,10),
        e = irange(1,10),
        f = irange(1,10),
        g = irange(1,10),
        h = irange(1,10),
    ) :
        A = NDArrayData.newRandom(a,b,c,d)
        B = NDArrayData.newRandom(e,f,a,b,g,h)
        C1 = absorbDenseSideIntoCornerFromLeft(A,B)
        C2 = A.contractWith(B,(0,1),(2,3)).join(2,3,(0,4),(1,5))
        self.assertDataAlmostEqual(C1,C2)
    # }}}
    @with_checker
    def test_absorbFromRight(self, # {{{
        a = irange(1,10),
        b = irange(1,10),
        c = irange(1,10),
        d = irange(1,10),
        e = irange(1,10),
        f = irange(1,10),
        g = irange(1,10),
        h = irange(1,10),
    ) :
        A = NDArrayData.newRandom(a,b,c,d)
        B = NDArrayData.newRandom(c,d,e,f,g,h)
        C1 = absorbDenseSideIntoCornerFromRight(A,B)
        C2 = A.contractWith(B,(2,3,),(0,1)).join((0,4),(1,5),2,3)
        self.assertDataAlmostEqual(C1,C2)
    # }}}
# }}}

class TestDenseSide(TestCase): # {{{
    @with_checker
    def test_absorbCenterSS(self, # {{{
        a = irange(1,3),
        b = irange(1,3),
        c = irange(1,3),
        d = irange(1,3),
        e = irange(1,3),
        f = irange(1,3),
        g = irange(1,3),
        h = irange(1,3),
        j = irange(1,3),
        k = irange(1,3),
        i = irange(0,3),
    ):
        A = NDArrayData.newRandom(a,b,c,d,e,e)
        B_shape = replaceAt((f,g,h,j,k),i,e)
        B = NDArrayData.newRandom(*B_shape)
        C1 = absorbDenseCenterSSIntoSide(i,A,B)
        C2 = A.contractWith(
                B.contractWith(B.conj(),(4,),(4,)),
                (4,5),
                (i,i+4),
             ).join((0,4+LA(i)),(1,7+LA(i)),(2,4+RA(i)),(3,7+RA(i)),4+OA(i),7+OA(i))
        self.assertDataAlmostEqual(C1,C2)
    # }}}
    @with_checker
    def test_absorbCenterSOS(self, # {{{
        a = irange(1,3),
        b = irange(1,3),
        c = irange(1,3),
        d = irange(1,3),
        e = irange(1,3),
        f = irange(1,3),
        g = irange(1,3),
        h = irange(1,3),
        j = irange(1,3),
        k = irange(1,3),
        i = irange(0,3),
    ):
        A = NDArrayData.newRandom(a,b,c,d,e,e)
        B_shape = replaceAt((f,g,h,j,k),i,e)
        B = NDArrayData.newRandom(*B_shape)
        C = NDArrayData.newRandom(k,k)
        D1 = absorbDenseCenterSOSIntoSide(i,A,B,C)
        D2 = A.contractWith(
                (B).contractWith(C,(4,),(1,)).contractWith(B.conj(),(4,),(4,)),
                (4,5),
                (i,i+4),
             ).join((0,4+LA(i)),(1,7+LA(i)),(2,4+RA(i)),(3,7+RA(i)),4+OA(i),7+OA(i))
        self.assertDataAlmostEqual(D1,D2)
    # }}}
# }}}

class TestDenseStages(TestCase): # {{{
    @with_checker
    def test_stage_1(self, # {{{
        a = irange(1,10),
        b = irange(1,10),
        c = irange(1,10),
        d = irange(1,10),
        e = irange(1,10),
        f = irange(1,10),
        g = irange(1,10),
        h = irange(1,10),
    ):
        A = NDArrayData.newRandom(a,b,c,d)
        B = NDArrayData.newRandom(c,d,e,f,g,h)
        C1 = formNormalizationStage1(A,B)
        C2 = A.contractWith(B,(2,3),(0,1)).join((0,1),(2,3),4,5)
        self.assertDataAlmostEqual(C1,C2)
    # }}}
    @with_checker
    def test_stage_2(self, # {{{
        a = irange(1,10),
        b = irange(1,10),
        c = irange(1,10),
        d = irange(1,10),
        e = irange(1,10),
        f = irange(1,10),
        g = irange(1,10),
    ):
        A = NDArrayData.newRandom(a,b,c,d)
        B = NDArrayData.newRandom(e,a,f,g)
        C1 = formNormalizationStage2(A,B)
        C2 = A.contractWith(B,(0,),(1,)).join(3,0,1,4,2,5)
        self.assertDataAlmostEqual(C1,C2)
    # }}}
    @with_checker
    def test_stage_3(self, # {{{
        a = irange(1,5),
        b = irange(1,5),
        c = irange(1,5),
        d = irange(1,5),
        e = irange(1,5),
        f = irange(1,5),
        g = irange(1,5),
    ):
        A = NDArrayData.newRandom(a,b,c,d,c,d)
        B = NDArrayData.newRandom(b,a,e,f,e,f)
        C = NDArrayData.newRandom(c,d,e,f,g)
        D1 = formNormalizationStage3(A,B)(C)
        AB = A.contractWith(B,(0,1),(1,0)).join(0,1,4,5,2,3,6,7)
        D2 = AB.contractWith(C,(0,1,2,3),(0,1,2,3))
        if D1.hasNaN() or D2.hasNaN():
            return
        self.assertDataAlmostEqual(D1,D2)
    # }}}
# }}}
