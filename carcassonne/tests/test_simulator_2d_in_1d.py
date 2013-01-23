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

class TestSimulator1D(TestCase): # {{{
    @ with_checker(number_of_calls=10) # test_on_magnetic_field {{{
    def dont_test_on_magnetic_field(self,direction=choiceof((0,1))):
        system = System.newTrivialWithSparseOperator(O=NDArrayData.Z)
        system.sweep_convergence_policy = RelativeStateDifferenceThresholdConvergencePolicy(1e-5)
        system.run_convergence_policy = RelativeOneSiteExpectationDifferenceThresholdConvergencePolicy(1e-7)
        system.increase_bandwidth_policy = OneDirectionIncrementBandwidthPolicy(direction)
        system.contraction_policy = RepeatPatternContractionPolicy([0+direction,2+direction])
        system.runUntilConverged()
        print(system.number_of_sweeps,system.number_of_iterations)
        self.assertAlmostEqual(system.computeOneSiteExpectation(),-1)
    # }}}
    @ with_checker(number_of_calls=10) # test_on_ferromagnetic_coupling {{{
    def dont_test_on_ferromagnetic_coupling(self,direction=choiceof((0,1))):
        if direction == 0:
            system = System.newTrivialWithSparseOperator(OO_LR=[NDArrayData.Z,-NDArrayData.Z])
        else:
            system = System.newTrivialWithSparseOperator(OO_UD=[NDArrayData.Z,-NDArrayData.Z])
        system.sweep_convergence_policy = RelativeOneSiteExpectationDifferenceThresholdConvergencePolicy(1e-7)
        system.run_convergence_policy = RelativeOneSiteExpectationDifferenceThresholdConvergencePolicy(1e-7)
        system.increase_bandwidth_policy = OneDirectionIncrementBandwidthPolicy(direction)
        system.contraction_policy = RepeatPatternContractionPolicy([0+direction,2+direction])
        system.runUntilConverged()
        print(system.number_of_sweeps,system.number_of_iterations)
        self.assertAlmostEqual(system.computeOneSiteExpectation(),-1)
    # }}}
    @ with_checker(number_of_calls=1) # test_on_transverseIsing {{{
    def test_on_transverse_Ising(self,direction=choiceof((0,1))):
        if direction == 0:
            system = System.newTrivialWithSparseOperator(O=-NDArrayData.Z,OO_LR=[NDArrayData.X,-0.01*NDArrayData.X])
        else:
            system = System.newTrivialWithSparseOperator(O=-NDArrayData.Z,OO_UD=[NDArrayData.X,-0.01*NDArrayData.X])
        system.sweep_convergence_policy = RelativeStateDifferenceThresholdConvergencePolicy(1e-5)
        system.run_convergence_policy = RelativeOneSiteExpectationDifferenceThresholdConvergencePolicy(1e-7)
        system.increase_bandwidth_policy = OneDirectionIncrementBandwidthPolicy(direction,2)
        class TestPolicy(RepeatPatternContractionPolicy):
            class BoundPolicy(RepeatPatternContractionPolicy.BoundPolicy):
                def apply(self):
                    #print(repr(self.system.state_center_data))
                    x = self.system.state_center_data
                    print((x.normalizeAxis(0)[0].absorbMatrixAt(0,x.normalizeAxis(2)[-1])-x).norm(),(x.normalizeAxis(2)[0].absorbMatrixAt(2,x.normalizeAxis(0)[-1])-x).norm())
                    #state_center_data = self.system.state_center_data
                    #left_normalized, _, left_remainder = state_center_data.normalizeAxis(0)
                    #right_normalized, _, right_remainder = state_center_data.normalizeAxis(1)
                    RepeatPatternContractionPolicy.BoundPolicy.apply(self)
        system.contraction_policy = TestPolicy([0+direction,2+direction])
        #system.contraction_policy = RepeatPatternContractionPolicy([0+direction,2+direction])
        system.runUntilConverged()
        print(system.number_of_sweeps,system.number_of_iterations)
        self.assertAlmostEqual(system.computeOneSiteExpectation(),-1.0000250001562545)
    # }}}
# }}}

def computeTransverseIsingGroundStateEnergy(Gamma,J): # {{{
    lam = J/(2*Gamma)
    theta = 4*lam/(1+lam)**2
    return -Gamma*2/pi*(1+lam)*ellipeinc(pi/2,theta)
# }}}
