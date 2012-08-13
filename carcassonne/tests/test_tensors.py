# Imports {{{
from numpy import dot, multiply
from paycheck import *

from ..tensors import *
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
        A = crand(a,b)
        B = crand(c,a,d)
        C1 = NormalizationCorner._absorbFromLeft(A,B)
        C2 = tensordot(A,B,(0,1)).transpose(1,0,2).reshape(c,b*d)
    # }}}
    @with_checker
    def test_absorbFromRight(self, # {{{
        a = irange(1,10),
        b = irange(1,10),
        c = irange(1,10),
        d = irange(1,10),
    ) :
        A = crand(a,b)
        B = crand(b,c,d)
        C1 = NormalizationCorner._absorbFromRight(A,B)
        C2 = tensordot(A,B,(1,0)).transpose(0,2,1).reshape(a*d,c)
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
        A = crand(a,b,c*c)
        B = crand(c,d,e,f,g)
        C1 = NormalizationSide._absorbCenterFrom[0](A.reshape(a,b,c,c),B.conj(),B)
        C2 = tensordot(
                A,
                tensordot(B.conj(),B,(4,4)).transpose(0,2,4,6,1,3,5,7).reshape(c*c,d*d,e*e,f*f),
                (2,0)
             ).transpose(0,2,1,4,3).reshape(a*d*d,b*f*f,e*e)
    #}}}
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
        A = crand(a,b,d*d)
        B = crand(c,d,e,f,g)
        C1 = NormalizationSide._absorbCenterFrom[1](A.reshape(a,b,d,d),B.conj(),B)
        C2 = tensordot(
                A,
                tensordot(B.conj(),B,(4,4)).transpose(0,2,4,6,1,3,5,7).reshape(c*c,d*d,e*e,f*f),
                (2,1)
             ).transpose(0,2,1,4,3).reshape(a*c*c,b*e*e,f*f)
    #}}}
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
        A = crand(a,b,e*e)
        B = crand(c,d,e,f,g)
        C1 = NormalizationSide._absorbCenterFrom[2](A.reshape(a,b,e,e),B.conj(),B)
        C2 = tensordot(
                A,
                tensordot(B.conj(),B,(4,4)).transpose(0,2,4,6,1,3,5,7).reshape(c*c,d*d,e*e,f*f),
                (2,2)
             ).transpose(0,4,1,3,2).reshape(a*f*f,b*d*d,c*c)
    #}}}
    @with_checker
    def test_absorbCenter_4(self, # {{{
        a = irange(1,3),
        b = irange(1,3),
        c = irange(1,3),
        d = irange(1,3),
        e = irange(1,3),
        f = irange(1,3),
        g = irange(1,3),
    ):
        A = crand(a,b,f*f)
        B = crand(c,d,e,f,g)
        C1 = NormalizationSide._absorbCenterFrom[3](A.reshape(a,b,f,f),B.conj(),B)
        C2 = tensordot(
                A,
                tensordot(B.conj(),B,(4,4)).transpose(0,2,4,6,1,3,5,7).reshape(c*c,d*d,e*e,f*f),
                (2,3)
             ).transpose(0,2,1,4,3).reshape(a*c*c,b*e*e,d*d)
    #}}}
# }}}
