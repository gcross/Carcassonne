# Imports {{{
from . import *
from ..policies import *
from ..system._1d2d import System
# }}}

class TestSimulator1D2D(TestCase): # {{{
    def test_on_magnetic_field(self): # {{{
        for direction in range(2):
            system = System.newSimple(direction,Pauli.Z)
            system.setPolicy("sweep convergence",RelativeStateDifferenceThresholdConvergencePolicy(1e-5))
            system.setPolicy("run convergence",RelativeEstimatedOneSiteExpectationDifferenceThresholdConvergencePolicy(1e-7))
            system.setPolicy("bandwidth increase",OneDirectionIncrementBandwidthIncreasePolicy(0,2))
            system.setPolicy("contraction",RepeatPatternContractionPolicy([0,1]))
            system.runUntilConverged()
            self.assertAlmostEqual(abs(system.computeEstimatedOneSiteExpectation()),1,places=5)
    # }}}
    def test_on_transverse_Ising(self): # {{{
        for direction in range(2):
            system = System.newSimple(direction,Pauli.Z,(-0.01*Pauli.X,Pauli.X),adaptive_state_threshold=True)
            system.checkEnvironments("preliminary check")
            system.setPolicy("sweep convergence",RelativeStateDifferenceThresholdConvergencePolicy(1e-5))
            system.setPolicy("run convergence",RelativeEstimatedOneSiteExpectationDifferenceThresholdConvergencePolicy(1e-7))
            system.setPolicy("bandwidth increase",OneDirectionIncrementBandwidthIncreasePolicy(0,2))
            system.setPolicy("contraction",RepeatPatternContractionPolicy([0,1]))
            system.runUntilConverged()
            self.assertAlmostEqual(system.computeOneSiteExpectation(),-1.0000250001562545,places=6)
    # }}}
    def test_on_Heisenberg(self): # {{{
        for direction in range(2):
            system = System.new(direction,OOs=[(Pauli.X,-Pauli.X),(Pauli.Y,-Pauli.Y),(Pauli.Z,Pauli.Z)],adaptive_state_threshold=True)
            system.checkEnvironments("preliminary check")
            system.setPolicy("sweep convergence",RelativeEstimatedOneSiteExpectationDifferenceThresholdConvergencePolicy(1e-5))
            system.setPolicy("run convergence",RelativeEstimatedOneSiteExpectationDifferenceThresholdConvergencePolicy(1e-3))
            system.setPolicy("bandwidth increase",OneDirectionIncrementBandwidthIncreasePolicy(0,2))
            system.setPolicy("contraction",RepeatPatternContractionPolicy([0,1]))
            system.runUntilConverged()
            self.assertAlmostEqual(system.computeEstimatedOneSiteExpectation()/4,-0.4431471805599,places=3)
    # }}}
# }}}
