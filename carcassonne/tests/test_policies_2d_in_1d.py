# Imports {{{
from numpy import ones

from . import *
from ..policies import *
from ..system._2d import System
from ..utils import O
# }}}

class TestConvergencePolicies1D(TestCase): # {{{
    @ with_checker(number_of_calls=10) # test_one_site_expectation {{{
    def test_one_site_expectation(self,direction=choiceof((0,1))):
        if direction == 0:
            system = System.newTrivialWithSparseOperator(O=-NDArrayData.Z,OO_LR=[NDArrayData.X,-0.01*NDArrayData.X])
        else:
            system = System.newTrivialWithSparseOperator(O=-NDArrayData.Z,OO_UD=[NDArrayData.X,-0.01*NDArrayData.X])
        system.setPolicy("sweep convergence",RelativeOneSiteExpectationDifferenceThresholdConvergencePolicy(1e-7))
        system.setPolicy("run convergence",RelativeOneSiteExpectationDifferenceThresholdConvergencePolicy(1e-7))
        system.setPolicy("bandwidth increase",OneDirectionIncrementBandwidthIncreasePolicy(direction,2))
        system.setPolicy("contraction",RepeatPatternContractionPolicy([0+direction,2+direction]))
        system.runUntilConverged()
        self.assertAlmostEqual(system.computeOneSiteExpectation(),-1.0000250001562545)
    # }}}
    @ with_checker(number_of_calls=10) # test_estimated_one_site_expectation {{{
    def test_estimated_one_site_expectation(self,direction=choiceof((0,1)),estimate_direction=choiceof((0,1,2,3))):
        if direction == 0:
            system = System.newTrivialWithSparseOperator(O=-NDArrayData.Z,OO_LR=[NDArrayData.X,-0.01*NDArrayData.X])
        else:
            system = System.newTrivialWithSparseOperator(O=-NDArrayData.Z,OO_UD=[NDArrayData.X,-0.01*NDArrayData.X])
        system.setPolicy("sweep convergence",RelativeEstimatedOneSiteExpectationDifferenceThresholdConvergencePolicy(estimate_direction,1e-7))
        system.setPolicy("run convergence",RelativeOneSiteExpectationDifferenceThresholdConvergencePolicy(1e-7))
        system.setPolicy("bandwidth increase",OneDirectionIncrementBandwidthIncreasePolicy(direction,2))
        system.setPolicy("contraction",RepeatPatternContractionPolicy([0+direction,2+direction]))
        system.runUntilConverged()
        self.assertAlmostEqual(system.computeOneSiteExpectation(),-1.0000250001562545)
    # }}}
    @ with_checker(number_of_calls=10) # test_state_difference {{{
    def test_state_difference(self,direction=choiceof((0,1))):
        if direction == 0:
            system = System.newTrivialWithSparseOperator(O=-NDArrayData.Z,OO_LR=[NDArrayData.X,-0.01*NDArrayData.X])
        else:
            system = System.newTrivialWithSparseOperator(O=-NDArrayData.Z,OO_UD=[NDArrayData.X,-0.01*NDArrayData.X])
        system.setPolicy("sweep convergence",RelativeStateDifferenceThresholdConvergencePolicy(1e-5))
        system.setPolicy("run convergence",RelativeOneSiteExpectationDifferenceThresholdConvergencePolicy(1e-7))
        system.setPolicy("bandwidth increase",OneDirectionIncrementBandwidthIncreasePolicy(direction,2))
        system.setPolicy("contraction",RepeatPatternContractionPolicy([0+direction,2+direction]))
        system.runUntilConverged()
        self.assertAlmostEqual(system.computeOneSiteExpectation(),-1.0000250001562545)
    # }}}
    @ with_checker(number_of_calls=10) # test_periodicity {{{
    def test_periodicity(self,direction=choiceof((0,1))):
        if direction == 0:
            system = System.newTrivialWithSparseOperator(O=-NDArrayData.Z,OO_LR=[NDArrayData.X,-0.01*NDArrayData.X])
        else:
            system = System.newTrivialWithSparseOperator(O=-NDArrayData.Z,OO_UD=[NDArrayData.X,-0.01*NDArrayData.X])
        system.setPolicy("sweep convergence",PeriodicyThresholdConvergencePolicy(1e-7,0,2))
        system.setPolicy("run convergence",RelativeOneSiteExpectationDifferenceThresholdConvergencePolicy(1e-7))
        system.setPolicy("bandwidth increase",OneDirectionIncrementBandwidthIncreasePolicy(direction,2))
        system.setPolicy("contraction",RepeatPatternContractionPolicy([0+direction,2+direction]))
        system.runUntilConverged()
        self.assertAlmostEqual(system.computeOneSiteExpectation(),-1.0000250001562545)
    # }}}
# }}}
