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
# }}}
