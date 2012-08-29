# Imports {{{
from numpy import all, dot, multiply, zeros
import operator
from paycheck import *

from ..sparse import *
from . import *
# }}}

class TestFormSparseTensor(TestCase): # {{{
    def test_contract_empty_tensors(self): # {{{
        empty_tensor = SparseTensor((),{})
        def throwit():
            raise Exception("Should not have been called!")
        self.assertEqual(formSparseContractor((),(),(),throwit)(empty_tensor,empty_tensor),empty_tensor)
    # }}}
    def test_outer_product_trivial(self): # {{{
        def contractChunks(c1,c2):
            contractChunks.n += 1
            return None
        contractChunks.n = 0
        tensor = SparseTensor((),{():None})
        self.assertEqual(formSparseContractor((),(),(),contractChunks)(tensor,tensor),tensor)
        self.assertEqual(contractChunks.n,1)
    # }}}
    @with_checker
    def test_dot_product_overlapping_tensors(self,x=irange(1,10),y=irange(1,10)): # {{{
        x_tensor = SparseTensor((1,),{(None,):x})
        y_tensor = SparseTensor((1,),{(None,):y})
        result_tensor = SparseTensor((),{():x*y})
        self.assertEqual(formSparseContractor((0,),(0,),(),operator.mul)(x_tensor,y_tensor),result_tensor)
    # }}}
    @with_checker
    def test_dot_product_non_overlapping_tensors(self,x=irange(1,10),y=irange(1,10)): # {{{
        x_tensor = SparseTensor((1,),{(True,):x})
        y_tensor = SparseTensor((1,),{(False,):y})
        result_tensor = SparseTensor((),{})
        self.assertEqual(formSparseContractor((0,),(0,),(),operator.mul)(x_tensor,y_tensor),result_tensor)
    # }}}
    @with_checker
    def test_dot_product2_overlapping_tensors(self, # {{{
        i=irange(1,10),
        j=irange(1,10),
        x=irange(1,10),
        y=irange(1,10)
    ):
        x_tensor = SparseTensor((10,10),{(i,j):x})
        y_tensor = SparseTensor((10,10),{(j,i):y})
        result_tensor = SparseTensor((),{():x*y})
        self.assertEqual(formSparseContractor((0,1),(1,0),(),operator.mul)(x_tensor,y_tensor),result_tensor)
    # }}}
    @with_checker
    def test_dot_product2_non_overlapping_tensors(self, # {{{
        i=irange(1,10),
        j=irange(11,20),
        x=irange(1,10),
        y=irange(1,10)
    ):
        x_tensor = SparseTensor((20,20),{(i,j):x})
        y_tensor = SparseTensor((20,20),{(j,i):y})
        result_tensor = SparseTensor((),{})
        self.assertEqual(formSparseContractor((0,1),(0,1),(),operator.mul)(x_tensor,y_tensor),result_tensor)
    # }}}
    @with_checker
    def test_outer_product_r1(self, # {{{
        x_chunks={(irange(0,2),): float},
        y_chunks={(irange(0,2),): float},
    ):
        x = SparseTensor((3,),x_chunks)
        y = SparseTensor((3,),y_chunks)
        z = formSparseContractor((),(),(FromLeft(0),FromRight(0)),operator.mul)(x,y)
        xd = formDenseTensor(x)
        yd = formDenseTensor(y)
        zd = formDenseTensor(z)
        self.assertAllClose(zd,multiply.outer(xd,yd))
    # }}}
    @with_checker
    def test_outer_product_r2(self, # {{{
        x_chunks={(irange(0,2),)*2: float},
        y_chunks={(irange(0,2),)*2: float},
    ):
        x = SparseTensor((3,)*2,x_chunks)
        y = SparseTensor((3,)*2,y_chunks)
        z = formSparseContractor((),(),(FromLeft(0),FromLeft(1),FromRight(0),FromRight(1)),operator.mul)(x,y)
        xd = formDenseTensor(x)
        yd = formDenseTensor(y)
        zd = formDenseTensor(z)
        self.assertAllClose(zd,multiply.outer(xd,yd))
    # }}}
    @with_checker
    def test_outer_product_r2_with_interleave(self, # {{{
        x_chunks={(irange(0,2),)*2: float},
        y_chunks={(irange(0,2),)*2: float},
    ):
        x = SparseTensor((3,)*2,x_chunks)
        y = SparseTensor((3,)*2,y_chunks)
        z = formSparseContractor((),(),(FromLeft(0),FromRight(0),FromLeft(1),FromRight(1)),operator.mul)(x,y)
        xd = formDenseTensor(x)
        yd = formDenseTensor(y)
        zd = formDenseTensor(z)
        self.assertAllClose(zd,multiply.outer(xd,yd).transpose(0,2,1,3))
    # }}}
    @with_checker
    def test_matmul(self, # {{{
        x_chunks={(irange(0,2),)*2: float},
        y_chunks={(irange(0,2),)*2: float},
    ):
        x = SparseTensor((3,)*2,x_chunks)
        y = SparseTensor((3,)*2,y_chunks)
        z = formSparseContractor((1,),(0,),(FromLeft(0),FromRight(1)),operator.mul)(x,y)
        xd = formDenseTensor(x)
        yd = formDenseTensor(y)
        zd = formDenseTensor(z)
        self.assertAllClose(zd,dot(xd,yd))
    # }}}
    @with_checker
    def test_r4_matmul(self, # {{{
        x_chunks={(irange(0,2),)*4: float},
        y_chunks={(irange(0,2),)*4: float},
    ):
        x = SparseTensor((3,)*4,x_chunks)
        y = SparseTensor((3,)*4,y_chunks)
        z = formSparseContractor((1,3),(0,2),(FromLeft(0),FromRight(3),FromRight(1),FromLeft(2)),operator.mul)(x,y)
        xd = formDenseTensor(x)
        yd = formDenseTensor(y)
        zd = formDenseTensor(z)
        self.assertAllClose(zd,tensordot(xd,yd,((1,3),(0,2))).transpose(0,3,2,1))
    # }}}
    @with_checker
    def test_matmul_with_merge(self, # {{{
        x_chunks={(irange(0,2),)*2: float},
        y_chunks={(irange(0,2),)*2: float},
    ):
        x = SparseTensor((3,)*2,x_chunks)
        y = SparseTensor((3,)*2,y_chunks)
        z = formSparseContractor((1,),(0,),(FromBoth(0,1),),operator.mul)(x,y)
        xd = formDenseTensor(x)
        yd = formDenseTensor(y)
        zd = formDenseTensor(z)
        self.assertAllClose(zd,dot(xd,yd).ravel())
    # }}}
    @with_checker
    def test_matmul_with_merge_with_nontrivial_redirect_with_sum(self, # {{{
        x_chunks={(irange(0,2),)*2: float},
        y_chunks={(irange(0,2),)*2: float},
        i=irange(0,2),
        j=irange(0,2),
        k=irange(0,8),
    ):
        x = SparseTensor((3,)*2,x_chunks)
        y = SparseTensor((3,)*2,y_chunks)
        z = formSparseContractor((1,),(0,),(FromBoth(0,1,indices_to_redirect={(i,j):(k,True)}),),operator.mul)(x,y)
        xd = formDenseTensor(x)
        yd = formDenseTensor(y)
        zd = formDenseTensor(z)
        zc = dot(xd,yd)
        if k != i*3+j:
            zc[k//3,k%3] += zc[i,j]
            zc[i,j] = 0
            self.assertTrue(all(zd[i*3+j]==0))
        zc = zc.ravel()
        self.assertAllClose(zd,zc)
    # }}}
    @with_checker
    def test_outer_product_with_merge(self, # {{{
        x_chunks={(irange(0,2),): float},
        y_chunks={(irange(0,2),): float},
    ):
        x = SparseTensor((3,),x_chunks)
        y = SparseTensor((3,),y_chunks)
        z = formSparseContractor((),(),(FromBoth(0,0),),operator.mul)(x,y)
        xd = formDenseTensor(x)
        yd = formDenseTensor(y)
        zd = formDenseTensor(z)
        self.assertAllClose(zd,multiply.outer(xd,yd).ravel())
    # }}}
    @with_checker
    def test_outer_product_with_merge_with_trivial_redirect_with_sum(self, # {{{
        x_chunks={(irange(0,2),): float},
        y_chunks={(irange(0,2),): float},
        i=irange(0,2),
        j=irange(0,2),
    ):
        x = SparseTensor((3,),x_chunks)
        y = SparseTensor((3,),y_chunks)
        z = formSparseContractor((),(),(FromBoth(0,0,indices_to_redirect={(i,j):(i*3+j,True)}),),operator.mul)(x,y)
        xd = formDenseTensor(x)
        yd = formDenseTensor(y)
        zd = formDenseTensor(z)
        zc = multiply.outer(xd,yd)
        zc = zc.ravel()
        self.assertAllClose(zd,zc)
    # }}}
    @with_checker
    def test_outer_product_with_merge_with_nontrivial_redirect_with_sum(self, # {{{
        x_chunks={(irange(0,2),): float},
        y_chunks={(irange(0,2),): float},
        i=irange(0,2),
        j=irange(0,2),
        k=irange(0,8),
    ):
        x = SparseTensor((3,),x_chunks)
        y = SparseTensor((3,),y_chunks)
        z = formSparseContractor((),(),(FromBoth(0,0,indices_to_redirect={(i,j):(k,True)}),),operator.mul)(x,y)
        xd = formDenseTensor(x)
        yd = formDenseTensor(y)
        zd = formDenseTensor(z)
        zc = multiply.outer(xd,yd)
        if k != i*3+j:
            zc[k//3,k%3] += zc[i,j]
            zc[i,j] = 0
            self.assertTrue(all(zd[i*3+j]==0))
        zc = zc.ravel()
        self.assertAllClose(zd,zc)
    # }}}
    @with_checker
    def test_redirect_without_sum(self,
        a=int,
        b=int,
    ): # {{{
        x = SparseTensor((3,),{(0,):a,(1,):a,(2,):a})
        y = SparseTensor((3,),{(0,):b,(1,):b,(2,):b})
        z = formSparseContractor((),(),(FromBoth(0,0,indices_to_redirect={(i,j):(0,False) for i in range(3) for j in range(3)}),),operator.mul)(x,y)
        self.assertEqual(z.chunks,{(0,):a*b})
    # }}}
    @with_checker
    def test_application_of_redirects(self, # {{{
        a=float,
        b=float,
        c=float,
        d=float,
        e=float,
        f=float,
    ):
        LA = SparseTensor((2,1),{(0,0):a,(1,0):b})
        LB = SparseTensor((2,2),{(0,0):b,(1,1):a})
        LC = formSparseContractor((0,),(0,),(FromBoth(1,1,indices_to_redirect={(0,1):(0,False)}),),operator.mul)(LA,LB)

        RA = SparseTensor((2,1),{(0,0):c,(1,0):d})
        RB = SparseTensor((2,2),{(0,0):e,(1,1):f})
        RC = formSparseContractor((0,),(0,),(FromBoth(1,1,indices_to_redirect={(0,1):(0,True)}),),operator.mul)(RA,RB)

        self.assertAllClose(
            dot(formDenseTensor(LC),formDenseTensor(RC)),
            dot(
                tensordot(formDenseTensor(LA),formDenseTensor(LB),(0,0)).ravel(),
                tensordot(formDenseTensor(RA),formDenseTensor(RB),(0,0)).ravel(),
            )
        )
    # }}}
# }}}

class TestFormDenseTensor(TestCase): # {{{
    def test_empty_tensor(self): # {{{
        sparse_tensor = SparseTensor((),{})
        dense_tensor = formDenseTensor(sparse_tensor)
        self.assertEqual(dense_tensor,0)
    # }}}
# }}}
