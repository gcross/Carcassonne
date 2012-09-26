# Imports {{{
from numpy import dot, multiply
from paycheck import *

from ..data import NDArrayData
from ..tensors.dense import *
from ..tensors.dense import formNormalizationStage1, formNormalizationStage2, formNormalizationStage3
from . import *
# }}}

class TestDenseCorner(TestCase): # {{{
    @with_checker # test_absorbFromLeft {{{
    def test_absorbFromLeft(self,
        a = irange(1,5),
        b = irange(1,5),
        c = irange(1,5),
        d = irange(1,5),
        e = irange(1,5),
        f = irange(1,5),
        g = irange(1,5),
        h = irange(1,5),
        i = irange(1,5),
        j = irange(1,5),
        k = irange(1,5),
    ) :
        A = NDArrayData.newRandom(a,b,c,d,e,f)
        B = NDArrayData.newRandom(g,h,i,a,b,c,j,k)
        C1 = absorbDenseSideIntoCornerFromLeft(A,B)
        C2 = A.contractWith(B,(0,1,2),(3,4,5)).join(3,4,5,(0,6),(1,7),2)
        self.assertDataAlmostEqual(C1,C2)
    # }}}
    @with_checker # test_absorbFromRight {{{
    def test_absorbFromRight(self,
        a = irange(1,5),
        b = irange(1,5),
        c = irange(1,5),
        d = irange(1,5),
        e = irange(1,5),
        f = irange(1,5),
        g = irange(1,5),
        h = irange(1,5),
        i = irange(1,5),
        j = irange(1,5),
        k = irange(1,5),
    ) :
        A = NDArrayData.newRandom(a,b,c,d,e,f)
        B = NDArrayData.newRandom(d,e,f,g,h,i,j,k)
        C1 = absorbDenseSideIntoCornerFromRight(A,B)
        C2 = A.contractWith(B,(3,4,5),(0,1,2)).join((0,6),(1,7),2,3,4,5)
        self.assertDataAlmostEqual(C1,C2)
    # }}}
# }}}

class TestDenseSide(TestCase): # {{{
    @with_checker # test_absorbCenterSS {{{
    def test_absorbCenterSS(self,
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
        l = irange(1,3),
        m = irange(1,3),
        i = irange(0,3),
    ):
        A = NDArrayData.newRandom(a,b,c,d,e,f,g,g)
        B_shape = replaceAt((h,j,k,l,m),i,g)
        B = NDArrayData.newRandom(*B_shape)
        C1 = absorbDenseCenterSSIntoSide(i,A,B)
        C2 = A.contractWith(
                B.contractWith(B.conj(),(4,),(4,)),
                (6,7),
                (i,i+4),
             ).join(
                (0,6+LA(i)),
                (1,9+LA(i)),
                2,
                (3,6+RA(i)),
                (4,9+RA(i)),
                5,
                6+OA(i),
                9+OA(i)
             )
        self.assertDataAlmostEqual(C1,C2)
    # }}}
    @with_checker # test_absorbCenterSOS {{{
    def test_absorbCenterSOS(self,
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
        l = irange(1,3),
        m = irange(1,3),
        i = irange(0,3),
    ):
        A = NDArrayData.newRandom(a,b,c,d,e,f,g,g)
        B_shape = replaceAt((h,j,k,l,m),i,g)
        B = NDArrayData.newRandom(*B_shape)
        C = NDArrayData.newRandom(m,m)
        D1 = absorbDenseCenterSOSIntoSide(i,A,B,C)
        D2 = A.contractWith(
                (B).contractWith(C,(4,),(1,)).contractWith(B.conj(),(4,),(4,)),
                (6,7),
                (i,i+4),
             ).join(
                (0,6+LA(i)),
                (1,9+LA(i)),
                2,
                (3,6+RA(i)),
                (4,9+RA(i)),
                5,
                6+OA(i),
                9+OA(i)
             )
        self.assertDataAlmostEqual(D1,D2)
    # }}}
# }}}

class TestDenseStages(TestCase): # {{{
    @with_checker # test_stage_1 {{{
    def test_stage_1(self,
        a = irange(1,5),
        b = irange(1,5),
        c = irange(1,5),
        d = irange(1,5),
        e = irange(1,5),
        f = irange(1,5),
        g = irange(1,5),
        h = irange(1,5),
        j = irange(1,5),
        k = irange(1,5),
        l = irange(1,5),
    ):
        A = NDArrayData.newRandom(a,b,c,d,e,f)
        B = NDArrayData.newRandom(d,e,f,g,h,j,k,l)
        C1 = formNormalizationStage1(A,B)
        C2 = A.contractWith(B,(3,4,5),(0,1,2)).join((0,1,2),(3,4,5),6,7)
        self.assertDataAlmostEqual(C1,C2)
    # }}}
    @with_checker # test_stage_2 {{{
    def test_stage_2(self,
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
    @with_checker # test_stage_3 {{{
    def test_stage_3(self,
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
        C = NDArrayData.newRandom(g,g)
        D = NDArrayData.newRandom(c,d,e,f,g)
        E1 = formNormalizationStage3(A,B,C)(D)
        AB = A.contractWith(B,(0,1),(1,0)).join(0,1,4,5,2,3,6,7)
        E2 = AB.contractWith(D,(0,1,2,3),(0,1,2,3))
        if E1.hasNaN() or E2.hasNaN():
            return
        self.assertDataAlmostEqual(E1,E2)
    # }}}
# }}}
