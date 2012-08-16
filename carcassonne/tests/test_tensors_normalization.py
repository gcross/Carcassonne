# Imports {{{
from numpy import dot, multiply
from paycheck import *

from ..data import NDArrayData
from ..tensors.normalization import *
from . import *
# }}}

class TestNormalizationCorner(TestCase): # {{{
    @with_checker
    def test_absorbFromLeft(self, # {{{
        a = irange(1,10),
        b = irange(1,10),
        c = irange(1,10),
        d = irange(1,10),
    ) :
        A = NDArrayData.newRandom((a,b))
        B = NDArrayData.newRandom((c,a,d))
        C1 = NormalizationCorner(A).absorbFromLeft(NormalizationSide(0,B)).data
        C2 = A.contractWith(B,(0,),(1,)).join(((1,),(0,2)))
        self.assertDataAlmostEqual(C1,C2)
    # }}}
    @with_checker
    def test_absorbFromRight(self, # {{{
        a = irange(1,10),
        b = irange(1,10),
        c = irange(1,10),
        d = irange(1,10),
    ) :
        A = NDArrayData.newRandom((a,b))
        B = NDArrayData.newRandom((b,c,d))
        C1 = NormalizationCorner(A).absorbFromRight(NormalizationSide(0,B)).data
        C2 = A.contractWith(B,(1,),(0,)).join(((0,2),(1,)))
        self.assertDataAlmostEqual(C1,C2)
    # }}}
# }}}

class TestNormalizationSide(TestCase): # {{{
    @with_checker
    def test_absorbCenter_0(self, # {{{
        a = irange(1,3),
        b = irange(1,3),
        c = irange(1,3),
        d = irange(1,3),
        e = irange(1,3),
        f = irange(1,3),
        g = irange(1,3),
    ):
        A = NDArrayData.newRandom((a,b,c*c))
        B = NDArrayData.newRandom((c,d,e,f,g))
        C1 = NormalizationSide(0,A).absorbCenter(B).data
        C2 = A.contractWith(
                (B).contractWith(B.conj(),(4,),(4,)
                  ).join(((0,4),(1,5),(2,6),(3,7))
                ),
                (2,),
                (0,),
             ).join(((0,2),(1,4),(3,)))
        self.assertDataAlmostEqual(C1,C2)
    # }}}
    @with_checker
    def test_absorbCenter_1(self, # {{{
        a = irange(1,3),
        b = irange(1,3),
        c = irange(1,3),
        d = irange(1,3),
        e = irange(1,3),
        f = irange(1,3),
        g = irange(1,3),
    ):
        A = NDArrayData.newRandom((a,b,d*d))
        B = NDArrayData.newRandom((c,d,e,f,g))
        C1 = NormalizationSide(1,A).absorbCenter(B).data
        C2 = A.contractWith(
                (B).contractWith(B.conj(),(4,),(4,)
                  ).join(((0,4),(1,5),(2,6),(3,7))
                ),
                (2,),
                (1,),
             ).join(((0,3),(1,2),(4,)))
        self.assertDataAlmostEqual(C1,C2)
    # }}}
    @with_checker
    def test_absorbCenter_2(self, # {{{
        a = irange(1,3),
        b = irange(1,3),
        c = irange(1,3),
        d = irange(1,3),
        e = irange(1,3),
        f = irange(1,3),
        g = irange(1,3),
    ):
        A = NDArrayData.newRandom((a,b,e*e))
        B = NDArrayData.newRandom((c,d,e,f,g))
        C1 = NormalizationSide(2,A).absorbCenter(B).data
        C2 = A.contractWith(
                (B).contractWith(B.conj(),(4,),(4,)
                  ).join(((0,4),(1,5),(2,6),(3,7))
                ),
                (2,),
                (2,),
             ).join(((0,4),(1,3),(2,)))
        self.assertDataAlmostEqual(C1,C2)
    # }}}
    @with_checker
    def test_absorbCenter_3(self, # {{{
        a = irange(1,3),
        b = irange(1,3),
        c = irange(1,3),
        d = irange(1,3),
        e = irange(1,3),
        f = irange(1,3),
        g = irange(1,3),
    ):
        A = NDArrayData.newRandom((a,b,f*f))
        B = NDArrayData.newRandom((c,d,e,f,g))
        C1 = NormalizationSide(3,A).absorbCenter(B).data
        C2 = A.contractWith(
                (B).contractWith(B.conj(),(4,),(4,)
                  ).join(((0,4),(1,5),(2,6),(3,7))
                ),
                (2,),
                (3,),
             ).join(((0,2),(1,4),(3,)))
        self.assertDataAlmostEqual(C1,C2)
    # }}}
# }}}
# }}}
