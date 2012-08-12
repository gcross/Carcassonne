# Imports {{{
from paycheck import *

from ..sparse import SparseTensor, formSparseContractor
from . import *
# }}}


class TestNullSparseTensor(TestCase): # {{{
    def test_contract_empty_tensors(self): # {{{
        empty_tensor = SparseTensor((),{})
        self.assertEqual(formSparseContractor((),(),(),None)(empty_tensor,empty_tensor),empty_tensor)
    # }}}
    def test_outer_product_tensors(self): # {{{
        def contractChunks(c1,c2):
            contractChunks.n += 1
            return None
        contractChunks.n = 0
        tensor = SparseTensor((),{():None})
        self.assertEqual(formSparseContractor((),(),(),contractChunks)(tensor,tensor),tensor)
        self.assertEqual(contractChunks.n,1)
    # }}}
# }}}
