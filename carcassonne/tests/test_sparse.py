# Imports {{{
import operator
from paycheck import *

from ..sparse import SparseTensor, formSparseContractor
from . import *
# }}}

class TestFormSparseTensor(TestCase): # {{{
    def test_contract_empty_tensors(self): # {{{
        empty_tensor = SparseTensor((),{})
        self.assertEqual(formSparseContractor((),(),(),None)(empty_tensor,empty_tensor),empty_tensor)
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
        x_tensor = SparseTensor((),{(None,):x})
        y_tensor = SparseTensor((),{(None,):y})
        result_tensor = SparseTensor((),{():x*y})
        self.assertEqual(formSparseContractor((0,),(0,),(),operator.mul)(x_tensor,y_tensor),result_tensor)
    # }}}
    @with_checker
    def test_dot_product_non_overlapping_tensors(self,x=irange(1,10),y=irange(1,10)): # {{{
        x_tensor = SparseTensor((),{(True,):x})
        y_tensor = SparseTensor((),{(False,):y})
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
        x_tensor = SparseTensor((),{(i,j):x})
        y_tensor = SparseTensor((),{(j,i):y})
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
        x_tensor = SparseTensor((),{(i,j):x})
        y_tensor = SparseTensor((),{(j,i):y})
        result_tensor = SparseTensor((),{})
        self.assertEqual(formSparseContractor((0,1),(0,1),(),operator.mul)(x_tensor,y_tensor),result_tensor)
    # }}}
# }}}
