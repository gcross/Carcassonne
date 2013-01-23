# Imports {{{
from functools import partial
from math import pi, sqrt
from paycheck import *
from scipy.special import ellipeinc

from . import *
from ..data import NDArrayData
from ..policies import *
from ..system import System
# }}}

class TestSimulator2Din15D(TestCase): # {{{
    @ with_checker(number_of_calls=10) # test_on_magnetic_field {{{
    def test_on_magnetic_field(self):
        system = System.newTrivialWithSparseOperator(O=NDArrayData.Z)
        system.state_compression_policy = ConstantStateCompressionPolicy(1)
        system.sweep_convergence_policy = RelativeStateDifferenceThresholdConvergencePolicy(1e-5)
        system.run_convergence_policy = RelativeOneSiteExpectationDifferenceThresholdConvergencePolicy(1e-7)
        system.increase_bandwidth_policy = AllDirectionsIncrementBandwidthPolicy()
        system.contraction_policy = RepeatPatternContractionPolicy(range(4))
        system.runUntilConverged()
        self.assertAlmostEqual(system.computeOneSiteExpectation(),-1)
    # }}}
    @ with_checker(number_of_calls=10) # test_on_ferromagnetic_coupling {{{
    def dont_test_on_ferromagnetic_coupling(self,direction=choiceof((0,1))):
        if direction == 0:
            system = System.newTrivialWithSparseOperator(OO_LR=[NDArrayData.Z,-NDArrayData.Z])
        else:
            system = System.newTrivialWithSparseOperator(OO_UD=[NDArrayData.Z,-NDArrayData.Z])
        system.state_compression_policy = ConstantStateCompressionPolicy(1)
        system.sweep_convergence_policy = RelativeOneSiteExpectationDifferenceThresholdConvergencePolicy(1e-7)
        system.run_convergence_policy = RelativeOneSiteExpectationDifferenceThresholdConvergencePolicy(1e-7)
        system.increase_bandwidth_policy = AllDirectionsIncrementBandwidthPolicy()
        system.contraction_policy = RepeatPatternContractionPolicy(range(4))
        system.runUntilConverged()
        self.assertAlmostEqual(system.computeOneSiteExpectation(),-1)
    # }}}
    @ with_checker(number_of_calls=10) # test_on_transverseIsing {{{
    def dont_test_on_transverse_Ising(self,direction=choiceof((0,1))):
        if direction == 0:
            system = System.newTrivialWithSparseOperator(O=-NDArrayData.Z,OO_LR=[NDArrayData.X,-0.01*NDArrayData.X])
        else:
            system = System.newTrivialWithSparseOperator(O=-NDArrayData.Z,OO_UD=[NDArrayData.X,-0.01*NDArrayData.X])
        system.state_compression_policy = ConstantStateCompressionPolicy(1)
        system.sweep_convergence_policy = RelativeStateDifferenceThresholdConvergencePolicy(1e-5)
        system.run_convergence_policy = RelativeOneSiteExpectationDifferenceThresholdConvergencePolicy(1e-7)
        system.increase_bandwidth_policy = AllDirectionsIncrementBandwidthPolicy()
        system.contraction_policy = RepeatPatternContractionPolicy(range(4))
        system.runUntilConverged()
        self.assertAlmostEqual(system.computeOneSiteExpectation(),-1.0000250001562545)
    # }}}
# }}}

def computeTransverseIsingGroundStateEnergy(Gamma,J): # {{{
    lam = J/(2*Gamma)
    theta = 4*lam/(1+lam)**2
    return -Gamma*2/pi*(1+lam)*ellipeinc(pi/2,theta)
# }}}
