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
    @ with_checker(number_of_calls=10) # test_on_transverseIsing {{{
    def test_on_transverse_Ising(self,direction=choiceof((0,1))):
        system = System.newTrivialWithSparseOperator(O=-NDArrayData.Z,OO_LR=[NDArrayData.X,-0.0001*NDArrayData.X])
        system.sweep_convergence_policy = RelativeCenterSiteExpectationDifferenceThresholdConvergencePolicy(1e-7)
        system.run_convergence_policy = RelativeCenterSiteExpectationDifferenceThresholdConvergencePolicy(1e-7)
        system.increase_bandwidth_policy = OneDirectionIncrementBandwidthPolicy(0)
        system.contraction_policy = RepeatPatternContractionPolicy([0,2])
        system.runUntilConverged()
        self.assertAlmostEqual(system.computeCenterSiteExpectation(),computeTransverseIsingGroundStateEnergy(1,0.000001))
    # }}}
# }}}


def computeTransverseIsingGroundStateEnergy(Gamma,J):
    lam = J/(2*Gamma)
    theta = 4*lam/(1+lam)**2
    return -Gamma*2/pi*(1+lam)*ellipeinc(pi/2,theta)
