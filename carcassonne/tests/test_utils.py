# Imports {{{
from numpy import dot, multiply
from paycheck import *

from ..utils import *
from . import *
# }}}

class TestFormAbsorber(TestCase): # {{{
    @with_checker
    def test_outer_product(self, # {{{
        n=irange(1,10),
        m=irange(1,10),
    ):
        A = crand(n)
        B = crand(m)
        C1 = formAbsorber((),(),(FromLeft(0),FromRight(0)))(A,B)
        C2 = multiply.outer(A,B)
        self.assertAllClose(C1,C2)
# }}}
    @with_checker
    def test_r2_outer_product(self, # {{{
        l=irange(1,10),
        m=irange(1,10),
        n=irange(1,10),
        o=irange(1,10),
    ):
        A = crand(l,m)
        B = crand(n,o)
        C1 = formAbsorber((),(),(FromLeft(0),FromLeft(1),FromRight(0),FromRight(1)))(A,B)
        C2 = multiply.outer(A,B)
        self.assertAllClose(C1,C2)
# }}}
    @with_checker
    def test_r2_outer_product_with_interleave(self, # {{{
        l=irange(1,10),
        m=irange(1,10),
        n=irange(1,10),
        o=irange(1,10),
    ):
        A = crand(l,m)
        B = crand(n,o)
        C1 = formAbsorber((),(),(FromLeft(0),FromRight(0),FromLeft(1),FromRight(1)))(A,B)
        C2 = multiply.outer(A,B).transpose(0,2,1,3)
        self.assertAllClose(C1,C2)
# }}}
    @with_checker
    def test_r2_outer_product_with_interleave_and_merge(self, # {{{
        l=irange(1,10),
        m=irange(1,10),
        n=irange(1,10),
        o=irange(1,10),
    ):
        A = crand(l,m)
        B = crand(n,o)
        C1 = formAbsorber((),(),(FromBoth(1,0),FromBoth(0,1)))(A,B)
        C2 = multiply.outer(A,B).transpose(1,2,0,3).reshape(m*n,l*o)
        self.assertAllClose(C1,C2)
# }}}
    @with_checker
    def test_matvec(self, # {{{
        l=irange(1,10),
        m=irange(1,10),
        n=irange(1,10),
    ):
        A = crand(l,m)
        B = crand(m,n)
        C1 = formAbsorber((1,),(0,),(FromLeft(0),FromRight(1,)))(A,B)
        C2 = dot(A,B)
        self.assertAllClose(C1,C2)
# }}}
    @with_checker
    def test_matvec_with_merge(self, # {{{
        l=irange(1,10),
        m=irange(1,10),
        n=irange(1,10),
    ):
        A = crand(l,m)
        B = crand(m,n)
        C1 = formAbsorber((1,),(0,),(FromBoth(0,1),))(A,B)
        C2 = dot(A,B).ravel()
        self.assertAllClose(C1,C2)
# }}}
    @with_checker
    def test_r4_matvec_with_merge(self, # {{{
        l=irange(1,10),
        m=irange(1,10),
        n=irange(1,10),
        o=irange(1,10),
        p=irange(1,10),
        q=irange(1,10),
    ):
        A = crand(l,m,n,o)
        B = crand(o,n,p,q)
        C1 = formAbsorber((2,3,),(1,0,),(FromBoth(0,3),FromBoth(1,2),))(A,B)
        C2 = tensordot(A,B,((2,3),(1,0))).transpose(0,3,1,2).reshape(l*q,m*p)
        self.assertAllClose(C1,C2)
# }}}
# }}}

class TestFormContractor(TestCase): # {{{
    @with_checker(number_of_calls=10)
    def test_trivial_case_1D(self, # {{{
        d = irange(1,10),
    ):
        x = crand(d)
        self.assertAllEqual(x,formContractor(['A'],[],[[('A',0)]])(x))
    # }}}

    @with_checker(number_of_calls=100)
    def test_trivial_case_ND(self, # {{{
        n = irange(1,8),
    ):
        x = crand(*[randint(1,3) for _ in range(n)])
        self.assertAllEqual(x,formContractor(['A'],[],[[('A',i)] for i in range(n)])(x))
    # }}}

    @with_checker(number_of_calls=100)
    def test_trivial_case_ND_flattened(self, # {{{
        n = irange(1,8),
    ):
        x = crand(*[randint(1,3) for _ in range(n)])
        self.assertAllEqual(x.ravel(),formContractor(['A'],[],[[('A',i) for i in range(n)]])(x))
    # }}}

    @with_checker(number_of_calls=10)
    def test_matvec(self, # {{{
        m = irange(1,10),
        n = irange(1,10),
    ):
        M = crand(m,n)
        v = crand(n)
        self.assertAllEqual(
            dot(M,v),
            formContractor(
                ['M','v'],
                [(('M',1),('v',0))],
                [[('M',0)]]
            )(M,v)
        )
    # }}}

    @with_checker(number_of_calls=10)
    def test_matvec_transposed(self, # {{{
        m = irange(1,10),
        n = irange(1,10)
    ):
        M = crand(n,m)
        v = crand(n)
        self.assertAllEqual(
            dot(v,M),
            formContractor(
                ['v','M'],
                [(('M',0),('v',0))],
                [[('M',1)]]
            )(v,M)
        )
    # }}}
    
    @with_checker(number_of_calls=10)
    def test_matmat(self, # {{{
        m = irange(1,10),
        n = irange(1,10),
        o = irange(1,10),
    ):
        A = crand(m,n)
        B = crand(n,o)
        self.assertAllClose(
            dot(A,B),
            formContractor(
                ['A','B'],
                [(('A',1),('B',0))],
                [[('A',0)],[('B',1)]]
            )(A,B)
        )
    # }}}

    @with_checker(number_of_calls=10)
    def test_matmat_flattened(self, # {{{
        m = irange(1,10),
        n = irange(1,10),
        o = irange(1,10),
    ):
        A = crand(m,n)
        B = crand(n,o)
        self.assertAllClose(
            dot(A,B).ravel(),
            formContractor(
                ['A','B'],
                [(('A',1),('B',0))],
                [[('A',0),('B',1)]]
            )(A,B)
        )
    # }}}

    @with_checker(number_of_calls=10)
    def test_matmat_transposed(self, # {{{
        m = irange(1,10),
        n = irange(1,10),
        o = irange(1,10),
    ):
        A = crand(m,n)
        B = crand(n,o)
        self.assertAllClose(
            dot(A,B).transpose(),
            formContractor(
                ['A','B'],
                [(('A',1),('B',0))],
                [[('B',1)],[('A',0)]]
            )(A,B)
        )
    # }}}

    @with_checker(number_of_calls=10)
    def test_matmat_swapped_and_transposed(self, # {{{
        m = irange(1,10),
        n = irange(1,10),
        o = irange(1,10),
    ):
        A = crand(n,m)
        B = crand(o,n)
        self.assertAllClose(
            dot(B,A).transpose(),
            formContractor(
                ['A','B'],
                [(('A',0),('B',1))],
                [[('A',1)],[('B',0)]]
            )(A,B)
        )
    # }}}
    
    @with_checker(number_of_calls=10)
    def test_r3r3(self, # {{{
        a = irange(1,10),
        b = irange(1,10),
        c = irange(1,10),
        d = irange(1,10),
        e = irange(1,10),
    ):
        A = crand(a,b,c)
        B = crand(c,d,e)
        self.assertAllClose(
            dot(A.reshape(a*b,c),B.reshape(c,d*e)).reshape(a,b,d,e),
            formContractor(
                ['A','B'],
                [(('A',2),('B',0))],
                [[('A',0)],[('A',1)],[('B',1)],[('B',2)]]
            )(A,B)
        )
    # }}}

    @with_checker(number_of_calls=10)
    def test_r3r3_semi_flattened(self, # {{{
        a = irange(1,10),
        b = irange(1,10),
        c = irange(1,10),
        d = irange(1,10),
        e = irange(1,10),
    ):
        A = crand(a,b,c)
        B = crand(c,d,e)
        self.assertAllClose(
            dot(A.reshape(a*b,c),B.reshape(c,d*e)),
            formContractor(
                ['A','B'],
                [(('A',2),('B',0))],
                [[('A',0),('A',1)],[('B',1),('B',2)]]
            )(A,B)
        )
    # }}}

    @with_checker(number_of_calls=10)
    def test_r3r3_semi_transposed_and_flattened(self, # {{{
        a = irange(1,10),
        b = irange(1,10),
        c = irange(1,10),
        d = irange(1,10),
        e = irange(1,10),
    ):
        A = crand(a,b,c)
        B = crand(c,d,e)
        self.assertAllClose(
            dot(A.reshape(a*b,c),B.reshape(c,d*e)).transpose().reshape(d,e,a,b).transpose(1,0,2,3).reshape(e*d,a,b),
            formContractor(
                ['A','B'],
                [(('A',2),('B',0))],
                [[('B',2),('B',1)],[('A',0)],[('A',1)]]
            )(A,B)
        )
    # }}}

    @with_checker(number_of_calls=10)
    def test_r3r2(self, # {{{
        a = irange(1,10),
        b = irange(1,10),
        c = irange(1,10),
        d = irange(1,10),
    ):
        A = crand(a,b,c)
        B = crand(c,d)
        self.assertAllClose(
            dot(A.reshape(a*b,c),B.reshape(c,d)).reshape(a,b,d),
            formContractor(
                ['A','B'],
                [(('A',2),('B',0))],
                [[('A',0)],[('A',1)],[('B',1)]]
            )(A,B)
        )
    # }}}

    @with_checker(number_of_calls=10)
    def test_r3r3_semi_flattened(self, # {{{
        a = irange(1,10),
        b = irange(1,10),
        c = irange(1,10),
        d = irange(1,10),
    ):
        A = crand(a,b,c)
        B = crand(c,d)
        self.assertAllClose(
            dot(A.reshape(a*b,c),B.reshape(c,d)).reshape(a,b*d),
            formContractor(
                ['A','B'],
                [(('A',2),('B',0))],
                [[('A',0)],[('A',1),('B',1)]]
            )(A,B)
        )
    # }}}
    
    @with_checker(number_of_calls=10)
    def test_matmatmat(self, # {{{
        a = irange(1,10),
        b = irange(1,10),
        c = irange(1,10),
        d = irange(1,10),
    ):
        A = crand(a,b)
        B = crand(b,c)
        C = crand(c,d)
        self.assertAllClose(
            dot(dot(A,B),C),
            formContractor(
                ['A','B','C'],
                [
                    (('A',1),('B',0)),
                    (('B',1),('C',0)),
                ],
                [[('A',0)],[('C',1)]]
            )(A,B,C)
        )
    # }}}

    @with_checker(number_of_calls=10)
    def test_triangle(self, # {{{
        a = irange(1,10),
        b = irange(1,10),
        c = irange(1,10),
        d = irange(1,10),
        e = irange(1,10),
        f = irange(1,10),
    ):
        A = crand(a,e,b)
        B = crand(c,b,f)
        C = crand(d,a,c)
        AB = tensordot(A,B,(2,1))
        ABC = tensordot(AB,C,((0,2),(1,2)))
        self.assertAllClose(
            ABC,
            formContractor(
                ['A','B','C'],
                [
                    (('A',0),('C',1)),
                    (('A',2),('B',1)),
                    (('B',0),('C',2)),
                ],
                [[('A',1)],[('B',2)],[('C',0)]]
            )(A,B,C)
        )
    # }}}

    @with_checker(number_of_calls=10)
    def test_1(self, # {{{
        a = irange(1,10),
        b = irange(1,10),
        c = irange(1,10),
        d = irange(1,10),
        e = irange(1,10),
        f = irange(1,10),
    ):
        A = crand(a,b,c)
        B = crand(d,e,c,f)
        AB = tensordot(A,B,(2,2)).transpose(0,2,1,3,4).reshape(a*d,b*e,f)
        self.assertAllClose(
            AB,
            formContractor(
                ['A','B'],
                [
                    (('A',2),('B',2)),
                ],
                [
                    [('A',0),('B',0)],
                    [('A',1),('B',1)],
                    [('B',3)],
                ]
            )(A,B)
        )
    # }}}

    @with_checker(number_of_calls=10)
    def test_2(self, # {{{
        a = irange(1,5),
        b = irange(1,5),
        c = irange(1,5),
        d = irange(1,5),
        e = irange(1,5),
        f = irange(1,5),
        g = irange(1,5),
        h = irange(1,5),
        i = irange(1,5),
    ):
        A = crand(a,b,c,d)
        B = crand(e,f,d,h,i)
        AB = tensordot(A,B,(3,2)).transpose(0,3,1,4,2,5,6).reshape(a*e,b*f,c*h,i)
        self.assertAllClose(
            AB,
            formContractor(
                ['A','B'],
                [
                    (('A',3),('B',2)),
                ],
                [
                    [('A',0),('B',0)],
                    [('A',1),('B',1)],
                    [('A',2),('B',3)],
                    [('B',4)],
                ]
            )(A,B)
        )
    # }}}
# }}}

