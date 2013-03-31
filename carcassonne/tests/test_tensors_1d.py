# Imports {{{
from . import *
from ..tensors._1d import *
# }}}

class TestFormExpectationMultiplier(TestCase): # {{{
    @with_checker # test_formMatrix_same_as_multiply {{{
    def test_formMatrix_same_as_multiply(self,
        lo=irange(1,5),ro=irange(1,5),
        ls=irange(1,5),rs=irange(1,5),
        p=irange(1,5)
    ):
        left_environment = NDArrayData.newRandom(lo,ls,ls)
        right_environment = NDArrayData.newRandom(ro,rs,rs)
        operator_center_data = NDArrayData.newRandom(lo,ro,p,p)
        state_center_data = NDArrayData.newRandom(ls,rs,p)
        multiply = formExpectationMultiplier(left_environment,right_environment,operator_center_data)
        multiply(state_center_data)
        matrix = multiply.formMatrix()
        self.assertDataAlmostEqual(multiply(state_center_data),matrix.contractWith(state_center_data.ravel(),1,0).split(ls,rs,p))
    # }}}
# }}}
