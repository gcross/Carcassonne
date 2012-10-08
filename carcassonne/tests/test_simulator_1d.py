# Imports {{{
from functools import partial
from paycheck import *

from . import *
from ..data import NDArrayData
from ..policies import *
from ..system import System
# }}}

class TestSimulator1D(TestCase): # {{{
    @ with_checker(number_of_calls=10) # test_on_magnetic_field {{{
    def test_on_magnetic_field(self,direction=choiceof((0,1))):
        system = System.newTrivialWithSparseOperator(O=NDArrayData.Z)
        system.sweep_convergence_policy = RelativeCenterSiteExpectationDifferenceThresholdConvergencePolicy(1e-7)
        system.run_convergence_policy = RelativeCenterSiteExpectationDifferenceThresholdConvergencePolicy(1e-7)
        system.increase_bandwidth_policy = OneDirectionIncrementBandwidthPolicy(0)
        system.contraction_policy = RepeatPatternContractionPolicy([0,2])
        system.runUntilConverged()
        self.assertAlmostEqual(system.computeCenterSiteExpectation(),-1)
    # }}}
    @ with_checker(number_of_calls=10) # test_on_ferromagnetic_coupling {{{
    def test_on_ferromagnetic_coupling(self,direction=choiceof((0,1))):
        system = System.newTrivialWithSparseOperator(OO_LR=[NDArrayData.Z,-NDArrayData.Z])
        system.sweep_convergence_policy = RelativeCenterSiteExpectationDifferenceThresholdConvergencePolicy(1e-7)
        system.run_convergence_policy = RelativeCenterSiteExpectationDifferenceThresholdConvergencePolicy(1e-7)
        system.increase_bandwidth_policy = OneDirectionIncrementBandwidthPolicy(0)
        system.contraction_policy = RepeatPatternContractionPolicy([0,2])
        system.runUntilConverged()
        self.assertAlmostEqual(system.computeCenterSiteExpectation(),-2)
    # }}}
# }}}
