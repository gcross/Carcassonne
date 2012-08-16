# Imports {{{
from numpy import dot, multiply
from paycheck import *

from ..data import NDArrayData
import carcassonne.tensors.normalization as TN
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
        e = irange(1,10),
    ) :
        A = NDArrayData.newRandom((a,b))
        B = NDArrayData.newRandom((c,a,d,e))
        C1 = NormalizationCorner(A).absorbFromLeft(NormalizationSide(0,B)).data
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
        A = NDArrayData.newRandom((a,b))
        B = NDArrayData.newRandom((b,c,d,e))
        C1 = NormalizationCorner(A).absorbFromRight(NormalizationSide(0,B)).data
        C2 = A.contractWith(B,(1,),(0,)).join((0,2,3),1)
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
        A = NDArrayData.newRandom((a,b,c,c))
        B = NDArrayData.newRandom((c,d,e,f,g))
        C1 = NormalizationSide(0,A).absorbCenter(B).data
        C2 = A.join(0,1,(2,3)).contractWith(
                (B).contractWith(B.conj(),(4,),(4,)
                  ).join((0,4),(1,5),(2,6),(3,7)
                ),
                (2,),
                (0,),
             ).join((0,2),(1,4),3).splitAt(2,e,e)
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
        A = NDArrayData.newRandom((a,b,d,d))
        B = NDArrayData.newRandom((c,d,e,f,g))
        C1 = NormalizationSide(1,A).absorbCenter(B).data
        C2 = A.join(0,1,(2,3)).contractWith(
                (B).contractWith(B.conj(),(4,),(4,)
                  ).join((0,4),(1,5),(2,6),(3,7)
                ),
                (2,),
                (1,),
             ).join((0,3),(1,2),4).splitAt(2,f,f)
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
        A = NDArrayData.newRandom((a,b,e,e))
        B = NDArrayData.newRandom((c,d,e,f,g))
        C1 = NormalizationSide(2,A).absorbCenter(B).data
        C2 = A.join(0,1,(2,3)).contractWith(
                (B).contractWith(B.conj(),(4,),(4,)
                  ).join((0,4),(1,5),(2,6),(3,7)
                ),
                (2,),
                (2,),
             ).join((0,4),(1,3),2).splitAt(2,c,c)
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
        A = NDArrayData.newRandom((a,b,f,f))
        B = NDArrayData.newRandom((c,d,e,f,g))
        C1 = NormalizationSide(3,A).absorbCenter(B).data
        C2 = A.join(0,1,(2,3)).contractWith(
                (B).contractWith(B.conj(),(4,),(4,)
                  ).join((0,4),(1,5),(2,6),(3,7)
                ),
                (2,),
                (3,),
             ).join((0,2),(1,4),3).splitAt(2,d,d)
        self.assertDataAlmostEqual(C1,C2)
    # }}}
# }}}

class TestNormalizationStage1(TestCase): # {{{
    @with_checker
    def test__init__(self, # {{{
        a = irange(1,10),
        b = irange(1,10),
        c = irange(1,10),
        d = irange(1,10),
        e = irange(1,10),
    ):
        A = NDArrayData.newRandom((a,b))
        B = NDArrayData.newRandom((b,c,d,e))
        C1 = TN.NormalizationStage1(NormalizationCorner(A),NormalizationSide(0,B)).data
        C2 = A.contractWith(B,(1,),(0,))
        self.assertDataAlmostEqual(C1,C2)
    # }}}
# }}}

class TestNormalizationStage2(TestCase): # {{{
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
        A = NDArrayData.newRandom((a,b,c,d))
        B = NDArrayData.newRandom((e,a,f,g))
        C1 = TN.NormalizationStage2(Dummy(data=A),Dummy(data=B)).data
        C2 = A.contractWith(B,(0,),(1,)).join(3,0,1,4,2,5)
        self.assertDataAlmostEqual(C1,C2)
    # }}}
# }}}

class TestNormalizationStage3(TestCase): # {{{
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
        A = NDArrayData.newRandom((a,b,c,d,c,d))
        B = NDArrayData.newRandom((b,a,e,f,e,f))
        C = NDArrayData.newRandom((c,d,e,f,g))
        D1 = TN.NormalizationStage3(Dummy(data=A),Dummy(data=B))(C)
        AB = A.contractWith(B,(0,1),(1,0)).join(0,1,4,5,2,3,6,7)
        D2 = AB.contractWith(C,(0,1,2,3),(0,1,2,3))
        self.assertDataAlmostEqual(D1,D2)
    # }}}
# }}}