# Imports {{{
from functools import partial
from math import pi, sqrt
from paycheck import *

from . import *
from ..data import NDArrayData
from ..policies import *
from ..system import System
# }}}

class TestSimulator2Din15D(TestCase): # {{{
    @ with_checker(number_of_calls=10) # dont_test_on_magnetic_field {{{
    def dont_test_on_magnetic_field(self):
        system = System.newTrivialWithSimpleSparseOperator(O=NDArrayData.Z)
        system.setPolicy("state compression",ConstantStateCompressionPolicy(1))
        system.setPolicy("sweep convergence",RelativeOneSiteExpectationDifferenceThresholdConvergencePolicy(1e-7))
        system.setPolicy("run convergence",RelativeOneSiteExpectationDifferenceThresholdConvergencePolicy(1e-7))
        system.setPolicy("bandwidth increase",AllDirectionsIncrementBandwidthIncreasePolicy())
        system.setPolicy("contraction",RepeatPatternContractionPolicy(range(4)))
        system.runUntilConverged()
        self.assertAlmostEqual(system.computeOneSiteExpectation(),-1,places=6)
    # }}}
    @ with_checker(number_of_calls=1) # dont_test_on_ferromagnetic_coupling {{{
    def test_on_ferromagnetic_coupling(self,direction=choiceof((0,1))):
        direction = 0
        if direction == 0:
            system = System.newTrivialWithSimpleSparseOperator(OO_LR=[NDArrayData.Z,-NDArrayData.Z])
        else:
            system = System.newTrivialWithSimpleSparseOperator(OO_UD=[NDArrayData.Z,-NDArrayData.Z])
        system.setPolicy("state compression",ConstantStateCompressionPolicy(1))
        system.setPolicy("sweep convergence",RelativeOneSiteExpectationDifferenceThresholdConvergencePolicy(1e-7))
        #system.setPolicy("sweep convergence",RelativeStateDifferenceThresholdConvergencePolicy(1e-5))
        system.setPolicy("run convergence",RelativeOneSiteExpectationDifferenceThresholdConvergencePolicy(1e-7))
        system.setPolicy("bandwidth increase",AllDirectionsIncrementBandwidthIncreasePolicy())
        system.setPolicy("contraction",RepeatPatternContractionPolicy(range(4)))
        #system.setPolicy("post-optimization hook",HookPolicy(lambda system: print(system.state_center_data)))
        system.runUntilConverged()
        self.assertAlmostEqual(system.computeOneSiteExpectation(),-1,places=6)
    # }}}
    #@ with_checker(number_of_calls=5) # dont_test_on_transverseIsing {{{
    def dont_test_on_transverse_Ising(self,direction=choiceof((0,1))):
        for direction in [0,1]:
            if direction == 0:
                system = System.newTrivialWithSimpleSparseOperator(O=-NDArrayData.Z,OO_LR=[NDArrayData.X,-0.01*NDArrayData.X])
            else:
                system = System.newTrivialWithSimpleSparseOperator(O=-NDArrayData.Z,OO_UD=[NDArrayData.X,-0.01*NDArrayData.X])
            system.setPolicy("state compression",ConstantStateCompressionPolicy(1))
            system.setPolicy("sweep convergence",RelativeOneSiteExpectationDifferenceThresholdConvergencePolicy(1e-5))
            system.setPolicy("run convergence",RelativeOneSiteExpectationDifferenceThresholdConvergencePolicy(1e-7))
            system.setPolicy("bandwidth increase",AllDirectionsIncrementBandwidthIncreasePolicy())
            system.setPolicy("contraction",RepeatPatternContractionPolicy(range(4)))
            system.runUntilConverged()
            self.assertAlmostEqual(system.computeOneSiteExpectation(),-1.0000250001562545,places=6)
    # }}}
    def dont_test_on_Heisenberg(self): # {{{
        for direction in [0]: #range(2):
            if direction == 0:
                system = System.newTrivialWithSparseOperator(OO_LRs=[(NDArrayData.X,-NDArrayData.X),(NDArrayData.Y,-NDArrayData.Y),(NDArrayData.Z,NDArrayData.Z)])
            else:
                system = System.newTrivialWithSparseOperator(OO_UDs=[(NDArrayData.X,-NDArrayData.X),(NDArrayData.Y,-NDArrayData.Y),(NDArrayData.Z,NDArrayData.Z)])
            system.setPolicy("state compression",ConstantStateCompressionPolicy(1))
            system.setPolicy("sweep convergence",RelativeEstimatedOneSiteExpectationDifferenceThresholdConvergencePolicy(1e-5))
            system.setPolicy("run convergence",RelativeEstimatedOneSiteExpectationDifferenceThresholdConvergencePolicy(1e-4))
            system.setPolicy("bandwidth increase",AllDirectionsIncrementBandwidthIncreasePolicy())
            system.setPolicy("contraction",RepeatPatternContractionPolicy(range(4)))
            system.runUntilConverged()
            self.assertAlmostEqual(system.computeOneSiteExpectation()/4,-0.4431471805599,places=3)
    # }}}
# }}}
