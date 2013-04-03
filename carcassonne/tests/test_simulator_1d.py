# Imports {{{
from functools import partial
from math import pi, sqrt
from numpy import ones
from paycheck import *

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
        system.setPolicy("sweep convergence",RelativeStateDifferenceThresholdConvergencePolicy(1e-5))
        system.setPolicy("run convergence",RelativeOneSiteExpectationDifferenceThresholdConvergencePolicy(1e-7))
        system.setPolicy("bandwidth increase",OneDirectionIncrementBandwidthIncreasePolicy(0))
        system.setPolicy("contraction",RepeatPatternContractionPolicy([0,1]))
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
        system.setPolicy("sweep convergence",RelativeOneSiteExpectationDifferenceThresholdConvergencePolicy(1e-7))
        system.setPolicy("run convergence",RelativeOneSiteExpectationDifferenceThresholdConvergencePolicy(1e-7))
        system.setPolicy("bandwidth increase",OneDirectionIncrementBandwidthIncreasePolicy(0))
        system.setPolicy("contraction",RepeatPatternContractionPolicy([0,1]))
        system.runUntilConverged()
        self.assertAlmostEqual(abs(system.computeOneSiteExpectation()),1,places=5)
    # }}}
    def test_on_transverse_Ising(self): # {{{
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
        system.setPolicy("sweep convergence",RelativeStateDifferenceThresholdConvergencePolicy(1e-5))
        system.setPolicy("run convergence",RelativeOneSiteExpectationDifferenceThresholdConvergencePolicy(1e-7))
        system.setPolicy("bandwidth increase",OneDirectionIncrementBandwidthIncreasePolicy(0,2))
        system.setPolicy("contraction",RepeatPatternContractionPolicy([0,1]))
        system.runUntilConverged()
        self.assertAlmostEqual(system.computeOneSiteExpectation(),1.0000250001562545,places=6)
    # }}}
# }}}
