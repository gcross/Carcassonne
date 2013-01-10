# Imports {{{
from . import *
from ..policies import *
from ..system._1d2d import System
# }}}

class TestSimulator1D2D(TestCase): # {{{
    def dont_test_on_magnetic_field(self): # {{{
        system = System.new(0,Pauli.Z)
        system.sweep_convergence_policy = RelativeStateDifferenceThresholdConvergencePolicy(1e-5)
        system.run_convergence_policy = RelativeOneSiteExpectationDifferenceThresholdConvergencePolicy(1e-7)
        system.increase_bandwidth_policy = OneDirectionIncrementBandwidthPolicy(0)
        system.contraction_policy = RepeatPatternContractionPolicy([0,1])
        system.runUntilConverged()
        self.assertAlmostEqual(abs(system.computeOneSiteExpectation()),1,places=5)
    # }}}
    def test_on_transverse_Ising(self): # {{{
        system = System.new(0,Pauli.Z,(-0.01*Pauli.X,Pauli.X))
        system.checkEnvironments("preliminary check")
        system.sweep_convergence_policy = RelativeStateDifferenceThresholdConvergencePolicy(1e-7)
        #system.sweep_convergence_policy = RelativeExpectationChangeDifferenceThresholdConvergencePolicy(0,1e-7)
        system.run_convergence_policy = RelativeExpectationChangeDifferenceThresholdConvergencePolicy(0,1e-7)
        system.increase_bandwidth_policy = OneDirectionIncrementBandwidthPolicy(0,2)
        system.contraction_policy = RepeatPatternContractionPolicy([0,1])
        system.runUntilConverged()
        print("FINAL EXP = {:.15}".format(system.computeOneSiteExpectation()))
        self.assertAlmostEqual(system.computeOneSiteExpectation(),-1.0000250001562545,places=6)
    # }}}
# }}}
