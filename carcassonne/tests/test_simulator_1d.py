# Imports {{{
from functools import partial
from math import pi, sqrt
from numpy import ones
from paycheck import *
from scipy.special import ellipeinc

from . import *
from ..policies import *
from ..system._1d import System
# }}}

class TestSimulator1D(TestCase): # {{{
    @ with_checker(number_of_calls=10) # test_on_magnetic_field {{{
    def test_on_magnetic_field(self):
        system = \
            System(
                buildProductTensor([1,0]),
                buildProductTensor([0,1]),
                buildTensor((2,2,2,2),{
                   (0,0): Pauli.I,
                   (1,1): Pauli.I,
                   (0,1): Pauli.Z,
                }),
                ones((1,1,2)),
            )
        system.sweep_convergence_policy = RelativeStateDifferenceThresholdConvergencePolicy(1e-5)
        system.run_convergence_policy = RelativeOneSiteExpectationDifferenceThresholdConvergencePolicy(1e-7)
        system.increase_bandwidth_policy = OneDirectionIncrementBandwidthPolicy(0)
        system.contraction_policy = RepeatPatternContractionPolicy([0,1])
        system.runUntilConverged()
        self.assertAlmostEqual(abs(system.computeOneSiteExpectation()),1,places=5)
    # }}}
    @ with_checker(number_of_calls=10) # test_on_ferromagnetic_coupling {{{
    def test_on_ferromagnetic_coupling(self):
        system = \
            System(
                [1,0,0],
                [0,0,1],
                buildTensor((3,3,2,2),{
                    (0,0): Pauli.I,
                    (0,1): Pauli.Z,
                    (1,2): Pauli.Z,
                    (2,2): Pauli.I,
                }),
                ones((1,1,2)),
            )
        system.sweep_convergence_policy = RelativeStateDifferenceThresholdConvergencePolicy(1e-5)
        system.run_convergence_policy = RelativeOneSiteExpectationDifferenceThresholdConvergencePolicy(1e-7)
        system.increase_bandwidth_policy = OneDirectionIncrementBandwidthPolicy(0)
        system.contraction_policy = RepeatPatternContractionPolicy([0,1])
        system.runUntilConverged()
        self.assertAlmostEqual(abs(system.computeOneSiteExpectation()),1,places=5)
    # }}}
    @ with_checker(number_of_calls=10) # test_on_transverseIsing {{{
    def test_on_transverse_Ising(self):
        system = \
            System(
                [1,0,0],
                [0,0,1],
                buildTensor((3,3,2,2),{
                    (0,0): Pauli.I,
                    (0,2): Pauli.Z,
                    (0,1): -0.01*Pauli.X,
                    (1,2): Pauli.X,
                    (2,2): Pauli.I,
                }),
                ones((1,1,2)),
            )
        system.sweep_convergence_policy = RelativeStateDifferenceThresholdConvergencePolicy(1e-5)
        system.run_convergence_policy = RelativeOneSiteExpectationDifferenceThresholdConvergencePolicy(1e-7)
        system.increase_bandwidth_policy = OneDirectionIncrementBandwidthPolicy(0,2)
        system.contraction_policy = RepeatPatternContractionPolicy([0,1])
        system.runUntilConverged()
        self.assertAlmostEqual(system.computeOneSiteExpectation(),1.0000250001562545,places=6)
    # }}}
# }}}

def computeTransverseIsingGroundStateEnergy(Gamma,J):
    lam = J/(2*Gamma)
    theta = 4*lam/(1+lam)**2
    return -Gamma*2/pi*(1+lam)*ellipeinc(pi/2,theta)
