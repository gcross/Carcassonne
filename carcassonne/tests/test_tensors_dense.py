# Imports {{{
from numpy import dot, multiply
from paycheck import *

from ..data import NDArrayData
import carcassonne.tensors.dense as CTD
from ..tensors.dense import *
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
    ) :
        A = NDArrayData.newRandom(a,b)
        B = NDArrayData.newRandom(c,a,d,e)
        C1 = DenseCorner(A).absorbFromLeft(DenseSide(B)).data
        C2 = A.contractWith(B,(0,),(1,)).join(1,(0,2,3))
        self.assertDataAlmostEqual(C1,C2)
    # }}}
    @with_checker
    def test_absorbFromRight(self, # {{{
        a = irange(1,10),
        b = irange(1,10),
        c = irange(1,10),
        d = irange(1,10),
        e = irange(1,10),
    ) :
        A = NDArrayData.newRandom(a,b)
        B = NDArrayData.newRandom(b,c,d,e)
        C1 = DenseCorner(A).absorbFromRight(DenseSide(B)).data
        C2 = A.contractWith(B,(1,),(0,)).join((0,2,3),1)
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
        i = irange(0,3),
    ):
        A = NDArrayData.newRandom(a,b,c,c)
        B_shape = replaceAt((d,e,f,g,h),i,c)
        B = NDArrayData.newRandom(*B_shape)
        C1 = DenseSide(A).absorbCenterSS(i,B).data
        C2 = A.join(0,1,(2,3)).contractWith(
                (B).contractWith(B.conj(),(4,),(4,)
                  ).join((0,4),(1,5),(2,6),(3,7)
                ),
                (2,),
                (i,),
             ).join((0,2+LA(i)),(1,2+RA(i)),2+OA(i)).splitAt(2,B_shape[O(i)],B_shape[O(i)])
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
        i = irange(0,3),
    ):
        A = NDArrayData.newRandom(a,b,c,c)
        B_shape = replaceAt((d,e,f,g,h),i,c)
        B = NDArrayData.newRandom(*B_shape)
        C = NDArrayData.newRandom(h,h)
        D1 = DenseSide(A).absorbCenterSOS(i,B,C).data
        D2 = A.join(0,1,(2,3)).contractWith(
                (B).contractWith(C,(4,),(1,)
                  ).contractWith(B.conj(),(4,),(4,)
                  ).join((0,4),(1,5),(2,6),(3,7)
                ),
                (2,),
                (i,),
             ).join((0,2+LA(i)),(1,2+RA(i)),2+OA(i)).splitAt(2,B_shape[O(i)],B_shape[O(i)])
        self.assertDataAlmostEqual(D1,D2)
    # }}}
# }}}

class TestDenseStage1(TestCase): # {{{
    @with_checker
    def test__init__(self, # {{{
        a = irange(1,10),
        b = irange(1,10),
        c = irange(1,10),
        d = irange(1,10),
        e = irange(1,10),
    ):
        A = NDArrayData.newRandom(a,b)
        B = NDArrayData.newRandom(b,c,d,e)
        C1 = CTD.DenseStage1(DenseCorner(A),DenseSide(B)).data
        C2 = A.contractWith(B,(1,),(0,))
        self.assertDataAlmostEqual(C1,C2)
    # }}}
# }}}

class TestDenseStage2(TestCase): # {{{
    @with_checker
    def test__init__(self, # {{{
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
        C1 = CTD.DenseStage2(Dummy(data=A),Dummy(data=B)).data
        C2 = A.contractWith(B,(0,),(1,)).join(3,0,1,4,2,5)
        self.assertDataAlmostEqual(C1,C2)
    # }}}
# }}}

class TestDenseStage3(TestCase): # {{{
    @with_checker
    def test__init__(self, # {{{
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
        D1 = CTD.DenseStage3(Dummy(data=A),Dummy(data=B))(C)
        AB = A.contractWith(B,(0,1),(1,0)).join(0,1,4,5,2,3,6,7)
        D2 = AB.contractWith(C,(0,1,2,3),(0,1,2,3))
        self.assertDataAlmostEqual(D1,D2)
    # }}}
# }}}
