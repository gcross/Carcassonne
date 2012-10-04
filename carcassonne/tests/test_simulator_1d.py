# Imports {{{
from functools import partial
from paycheck import *

from . import *
from ..data import NDArrayData
from ..simulator import *
# }}}

class TestSimulator1D(TestCase): # {{{
    @ with_checker(number_of_calls=10) # test_on_magnetic_field {{{
    def test_on_magnetic_field(self,direction=choiceof((0,1))):
        simulator = Simulator1D(direction=direction,O=NDArrayData.Z)
        simulator.runUntilConverged(
            makeConvergenceThresholdTest(1e-7),
            lambda old_bandwidth: old_bandwidth+1,
            partial(makeConvergenceThresholdTest,1e-7)
        )
        self.assertAlmostEqual(simulator.computeOneSiteExpectation(),-1)
    # }}}
    @ with_checker(number_of_calls=10) # test_on_magnetic_field {{{
    def test_on_ferromagnetic_coupling(self,direction=choiceof((0,1))):
        simulator = Simulator1D(direction=direction,OO=[NDArrayData.Z,-NDArrayData.Z])
        simulator.runUntilConverged(
            makeConvergenceThresholdTest(1e-7),
            lambda old_bandwidth: old_bandwidth+1,
            partial(makeConvergenceThresholdTest,1e-7)
        )
        self.assertAlmostEqual(simulator.computeOneSiteExpectation(),-2)
    # }}}
# }}}
