# Imports {{{
from functools import partial
from math import pi, sqrt
from paycheck import *

from . import *
from ..data import NDArrayData
from ..policies import *
from ..system import System
# }}}

class TestSimulator1D(TestCase): # {{{
    @ with_checker(number_of_calls=10) # test_on_magnetic_field {{{
    def test_on_magnetic_field(self,direction=choiceof((0,1))):
        system = System.newTrivialWithSimpleSparseOperator(O=NDArrayData.Z)
        system.setPolicy("sweep convergence",RelativeStateDifferenceThresholdConvergencePolicy(1e-5))
        system.setPolicy("run convergence",RelativeOneSiteExpectationDifferenceThresholdConvergencePolicy(1e-7))
        system.setPolicy("bandwidth increase",OneDirectionIncrementBandwidthIncreasePolicy(direction))
        system.setPolicy("contraction",RepeatPatternContractionPolicy([0+direction,2+direction]))
        system.runUntilConverged()
        self.assertAlmostEqual(system.computeOneSiteExpectation(),-1)
    # }}}
    @ with_checker(number_of_calls=10) # test_on_ferromagnetic_coupling {{{
    def test_on_ferromagnetic_coupling(self,direction=choiceof((0,1))):
        if direction == 0:
            system = System.newTrivialWithSimpleSparseOperator(OO_LR=[NDArrayData.Z,-NDArrayData.Z])
        else:
            system = System.newTrivialWithSimpleSparseOperator(OO_UD=[NDArrayData.Z,-NDArrayData.Z])
        system.setPolicy("sweep convergence",RelativeOneSiteExpectationDifferenceThresholdConvergencePolicy(1e-7))
        system.setPolicy("run convergence",RelativeOneSiteExpectationDifferenceThresholdConvergencePolicy(1e-7))
        system.setPolicy("bandwidth increase",OneDirectionIncrementBandwidthIncreasePolicy(direction))
        system.setPolicy("contraction",RepeatPatternContractionPolicy([0+direction,2+direction]))
        system.runUntilConverged()
        self.assertAlmostEqual(system.computeOneSiteExpectation(),-1)
    # }}}
    @ with_checker(number_of_calls=10) # test_on_transverseIsing {{{
    def test_on_transverse_Ising(self,direction=choiceof((0,1))):
        if direction == 0:
            system = System.newTrivialWithSimpleSparseOperator(O=-NDArrayData.Z,OO_LR=[NDArrayData.X,-0.01*NDArrayData.X])
        else:
            system = System.newTrivialWithSimpleSparseOperator(O=-NDArrayData.Z,OO_UD=[NDArrayData.X,-0.01*NDArrayData.X])
        system.setPolicy("sweep convergence",RelativeStateDifferenceThresholdConvergencePolicy(1e-5))
        system.setPolicy("run convergence",RelativeOneSiteExpectationDifferenceThresholdConvergencePolicy(1e-7))
        system.setPolicy("bandwidth increase",OneDirectionIncrementBandwidthIncreasePolicy(direction,2))
        system.setPolicy("contraction",RepeatPatternContractionPolicy([0+direction,2+direction]))
        system.runUntilConverged()
        self.assertAlmostEqual(system.computeOneSiteExpectation(),-1.0000250001562545)
    # }}}
    def test_on_Heisenberg(self): # {{{
        for direction in range(2):
            if direction == 0:
                system = System.newTrivialWithSparseOperator(OO_LRs=[(NDArrayData.X,-NDArrayData.X),(NDArrayData.Y,-NDArrayData.Y),(NDArrayData.Z,NDArrayData.Z)])
            else:
                system = System.newTrivialWithSparseOperator(OO_UDs=[(NDArrayData.X,-NDArrayData.X),(NDArrayData.Y,-NDArrayData.Y),(NDArrayData.Z,NDArrayData.Z)])
            system.setPolicy("sweep convergence",RelativeEstimatedOneSiteExpectationDifferenceThresholdConvergencePolicy(1e-5,direction))
            system.setPolicy("run convergence",RelativeEstimatedOneSiteExpectationDifferenceThresholdConvergencePolicy(1e-4,direction))
            system.setPolicy("bandwidth increase",OneDirectionIncrementBandwidthIncreasePolicy(direction,2))
            system.setPolicy("contraction",RepeatPatternContractionPolicy([direction+2,direction+0]))
            system.runUntilConverged()
            self.assertAlmostEqual(system.computeEstimatedOneSiteExpectation(direction)/4,-0.4431471805599,places=3)
    # }}}
# }}}
