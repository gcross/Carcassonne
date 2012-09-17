# Imports {{{
from . import *
from ..system import System
# }}}

class TestTwoSiteOperator(TestCase):
    @with_checker # def test_no_steps {{{
    def test_no_steps(self,physical_dimension=irange(1,5)):
        Os = [NDArrayData.newRandomHermitian(physical_dimension,physical_dimension) for _ in range(4)]
        system = System.newTrivialWithSparseOperator(OO_UD=(Os[0],Os[1]),OO_LR=(Os[2],Os[3]))
        self.assertAlmostEqual(system.computeExpectation(),0)
    # }}}
